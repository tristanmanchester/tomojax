"""Batched matched projectors shared by iterative reconstruction solvers.

Fixed-size view loops bound storage for parallel Joseph/ray and cone models.
Subsets preserve poses, source arrangements and padding weights; streamed
measurements remain host resident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from tomojax.backends.host_stream import read_views
from tomojax.core.backend_policy import cuda_backend
from tomojax.core.cone import ConeModel, cone_model, use_cuda_cone
from tomojax.core.devices import VIEWS, ViewSplit, pad_views
from tomojax.core.geometry.cone import is_cone_beam
from tomojax.core.operator_norm import unit
from tomojax.recon._projection_batches import ViewBatches

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Grid

type ProjectorModel = Literal["auto", "joseph", "ray"]
type ProjectorBackend = Literal["auto", "jax", "cuda"]


def resolve_projector(
    model: str,
    backend: str,
    *,
    det_grid: object = None,
    context: str,
) -> tuple[str, str]:
    """Resolve ``auto`` choices; explicit choices must be supported, never replaced.

    ``auto`` selects Joseph plane sampling with CUDA kernels on CUDA. Explicit
    detector grids need the ray model, and CUDA needs CUDA and the canonical
    detector grid.
    """
    if model not in {"auto", "joseph", "ray"}:
        raise ValueError(f"{context}: projector_model must be 'auto', 'joseph' or 'ray'")
    if backend not in {"auto", "jax", "cuda"}:
        raise ValueError(f"{context}: projector_backend must be 'auto', 'jax' or 'cuda'")
    if model == "auto":
        model = "ray" if det_grid is not None else "joseph"
    if model == "joseph" and det_grid is not None:
        raise ValueError(f"{context}: Joseph sampling requires the canonical detector grid")
    cuda = cuda_backend()
    if backend == "auto":
        backend = "cuda" if cuda and det_grid is None else "jax"
    if backend == "cuda" and (not cuda or det_grid is not None):
        raise ValueError(f"{context}: CUDA requires CUDA and the canonical detector grid")
    return model, backend


def resolve_geometry_projector(
    geometry: object,
    model: str,
    backend: str,
    *,
    detector: Detector,
    det_grid: object = None,
    context: str,
) -> tuple[str | ConeModel, str]:
    """Resolve the projector for a geometry: :class:`ConeModel` for cone beams.

    Cone beams use Joseph plane sampling (``model`` must be ``auto`` or
    ``joseph``) and the canonical detector grid; ``detector`` is the one the
    solver projects onto (a binned one, say). Their ``cuda`` backend is
    the CUDA cone kernels, chosen by ``auto`` when CuPy and a CUDA device are
    available.
    """
    if not is_cone_beam(geometry):
        return resolve_projector(model, backend, det_grid=det_grid, context=context)
    if det_grid is not None:
        raise ValueError(f"{context}: cone-beam geometry needs the canonical detector grid")
    if model not in {"auto", "joseph"}:
        raise ValueError(
            f"{context}: cone-beam geometry uses Joseph plane sampling; "
            "projector_model must be 'auto' or 'joseph'"
        )
    if backend not in {"auto", "jax", "cuda"}:
        raise ValueError(f"{context}: projector_backend must be 'auto', 'jax' or 'cuda'")
    cuda = use_cuda_cone()
    if backend == "auto":
        backend = "cuda" if cuda else "jax"
    if backend == "cuda" and not cuda:
        raise ValueError(f"{context}: the CUDA cone kernels need CuPy on a CUDA device")
    cone = cone_model(geometry, detector)
    assert cone is not None
    return cone, backend


def _on_devices[T](
    split: ViewSplit,
    ops: ViewBatches,
    local: Callable[..., T],
    *,
    shared: tuple[jax.Array, ...] = (),
    views: tuple[jax.Array, ...] = (),
    out_specs: PartitionSpec,
) -> T:
    """``local(share, *shared, *views)`` on every device of ``split``, with its views.

    ``shared`` arrays (a volume) are whole on every device; ``views`` stacks are
    cut into the devices' views, padded with zero views first if need be.
    """
    per_view = tuple(pad_views(a, split.total, repeat=True) for a in ops.per_view())

    def body(per_view: tuple[jax.Array, ...], counts: jax.Array, shared: tuple, views: tuple) -> T:
        return local(ops.share(per_view, counts), *shared, *views)

    cut, whole = PartitionSpec(VIEWS), PartitionSpec()
    # Unchecked: the projectors' loops start from zeros that the device's views
    # then change, which the check would make every loop declare.
    return jax.shard_map(
        body,
        mesh=split.mesh,
        in_specs=(cut, cut, whole, cut),
        out_specs=out_specs,
        check_vma=False,
    )(per_view, split.counted(), shared, tuple(pad_views(a, split.total) for a in views))


def _summed[T](
    split: ViewSplit | None,
    ops: ViewBatches,
    local: Callable[..., T],
    *,
    shared: tuple[jax.Array, ...] = (),
    views: tuple[jax.Array, ...] = (),
) -> T:
    """``local(ops, *shared, *views)``; with a ``split``, its sum over the devices' shares."""
    if split is None:
        return local(ops, *shared, *views)

    def on_device(share: ViewBatches, *arrays: jax.Array) -> T:
        return jax.lax.psum(local(share, *arrays), VIEWS)

    return _on_devices(split, ops, on_device, shared=shared, views=views, out_specs=PartitionSpec())


def chunk_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str | ConeModel,
) -> tuple[
    Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
]:
    """Matched canonical-detector block operators with full-scan coefficients.

    ``project(volume, start, poses)`` and ``backproject(images, start, poses,
    output)`` process one block. ``start`` is its offset in this full pose
    stack, and the block's ``poses`` must be the slice at that offset. Source
    arrangements stay attached to global view indices across segment boundaries.
    """
    ops = ViewBatches(
        poses, grid, detector, None, backend, batch_size, model, "linear",
        absolute_weights=False,
    )  # fmt: skip
    return ops.project, ops.backproject


def projection_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    backend: str,
    batch_size: int,
    model: str | ConeModel = "ray",
    joseph_interpolation: str = "linear",
    *,
    absolute_weights: bool = False,
    split: ViewSplit | None = None,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[..., jax.Array]]:
    """Return matched ``(forward, adjoint)`` over all views in fixed-size batches.

    ``absolute_weights`` returns ``abs(A).T`` instead of the adjoint, for
    cancellation bounds with negative interpolation lobes. ``adjoint(stacks,
    combine)`` applies ``combine`` to batches of several stacks, so a derived
    sinogram such as ``abs(a) + abs(b)`` is never stored whole; ``accumulate``
    adds the result to an existing volume, in place for Joseph on CUDA on one
    device. With a ``split``, the views are shared among its devices and the
    sinograms both take and return have its padded :attr:`ViewSplit.total` views.
    """
    ops = ViewBatches(
        poses,
        grid,
        detector,
        det_grid,
        backend,
        batch_size,
        model,
        joseph_interpolation,
        absolute_weights=absolute_weights,
    )

    def forward(volume: jax.Array) -> jax.Array:
        if split is None:
            return _forward_views(ops, volume)
        return _on_devices(
            split, ops, _forward_views, shared=(volume,), out_specs=PartitionSpec(VIEWS)
        )

    def adjoint(
        images: jax.Array | tuple[jax.Array, ...],
        combine: Callable[..., jax.Array] | None = None,
        accumulate: jax.Array | None = None,
    ) -> jax.Array:
        stacks = tuple(images) if isinstance(images, tuple | list) else (images,)
        if combine is None and len(stacks) != 1:
            raise ValueError("adjoint: several stacks need a combine")
        if split is None:
            initial = ops.zeros_volume() if accumulate is None else accumulate
            return _adjoint_views(ops, stacks, combine, initial)

        def local(share: ViewBatches, *stacks: jax.Array) -> jax.Array:
            return _adjoint_views(share, stacks, combine, share.zeros_volume())

        total = _summed(split, ops, local, views=stacks)
        return total if accumulate is None else accumulate + total

    return forward, adjoint


def subset_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    backend: str,
    batch_size: int,
    model: str | ConeModel,
    subsets: int,
    *,
    stream: bool = False,
):
    """A factory for interleaved subsets of a matched, nonnegative operator.

    ``select(index)`` returns ``(forward, adjoint, residual_adjoint)``. Each subset has
    ``ceil(views / subsets)`` entries; trailing repeated indices have zero
    operator weights. ``residual_adjoint(x, full_data, row_weights, row_support=mask)`` evaluates
    ``A.T @ (row_weights * (full_data[indices] - A*x))`` one batch at a time,
    without storing a predicted sinogram. Cone source arrangements and detector
    centres stay attached to their original views.
    With ``stream``, data names a registered host source. Passing ``weights=None``
    recomputes inverse row sums from ``row_support`` per batch instead of storing them;
    ``row_support=None`` denotes uniform unit support.
    """
    size = (int(poses.shape[0]) + subsets - 1) // subsets
    chunks = (size + batch_size - 1) // batch_size
    # Keep the callback count while minimizing the final shifted batch's overlap.
    balanced_batch = (size + chunks - 1) // chunks
    ops = ViewBatches(
        poses, grid, detector, det_grid, backend, balanced_batch, model, "linear",
        absolute_weights=False,
    )  # fmt: skip

    def select(index):
        indices = index + subsets * jnp.arange(size)
        valid = indices < ops.n
        indices = jnp.minimum(indices, ops.n - 1)
        selected = ops.share(tuple(a[indices] for a in ops.per_view()), valid)

        def forward(volume):
            return _forward_views(selected, volume)

        def adjoint(images):
            return _adjoint_views(selected, (images,), None, selected.zeros_volume())

        def residual_adjoint(
            volume, data, weights, *, row_support: jax.Array | None = None, return_column=False
        ):
            project = selected.projector(volume)

            def body(chunk, output):
                start, batch, counts = selected.select(chunk)
                view_ids = jax.lax.dynamic_slice_in_dim(indices, start, selected.size)
                measured = (
                    read_views(
                        data, start, (selected.size, detector.nv, detector.nu), indices=view_ids
                    )
                    if stream
                    else data[view_ids]
                )
                residual = measured - project(start, batch)
                if weights is None:
                    row = (
                        selected.project_ones(start, batch)
                        if row_support is None
                        else selected.project(row_support, start, batch)
                    )
                    residual = jnp.where(row > 0, residual / jnp.where(row > 0, row, 1.0), 0.0)
                else:
                    residual = residual * jax.lax.dynamic_slice(
                        weights, (start, 0, 0), (selected.size, weights.shape[1], detector.nu)
                    )
                residual = jnp.where(counts[:, None, None], residual, 0.0)
                if return_column:
                    ones = jnp.broadcast_to(
                        counts[:, None, None].astype(jnp.float32), residual.shape
                    )
                    return selected.backproject_pair((residual, ones), start, batch, output)
                return selected.backproject(residual, start, batch, output)

            initial = selected.zeros_volume()
            return jax.lax.fori_loop(
                0, selected.count, body, (initial, initial) if return_column else initial
            )

        return forward, adjoint, residual_adjoint

    return select


def _forward_views(ops: ViewBatches, volume: jax.Array) -> jax.Array:
    project = ops.projector(volume)

    def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
        start, batch, _ = ops.select(chunk)
        projected = project(start, batch)
        if ops.counts is not None:  # a split's padding projects to zero
            projected = jnp.where(ops.counted(start)[:, None, None], projected, 0.0)
        return jax.lax.dynamic_update_slice(output, projected, (start, 0, 0))

    empty = jnp.zeros((ops.n, ops.detector.nv, ops.detector.nu), dtype=jnp.float32)
    return jax.lax.fori_loop(0, ops.count, body, empty)


def _adjoint_views(
    ops: ViewBatches,
    stacks: tuple[jax.Array, ...],
    combine: Callable[..., jax.Array] | None,
    initial: jax.Array,
) -> jax.Array:
    def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
        start, batch, valid = ops.select(chunk)
        slices = [ops.images(a, start) for a in stacks]
        data = slices[0] if combine is None else combine(*slices)
        data = jnp.where(valid[:, None, None], data, 0.0)
        return ops.backproject(data, start, batch, output)

    return jax.lax.fori_loop(0, ops.count, body, initial)


def _measured(ops: ViewBatches, data: jax.Array, start: jax.Array, *, stream: bool) -> jax.Array:
    """The batch at ``start`` of ``data``, or of the host source ``data`` names."""
    if stream:
        return read_views(data, start, (ops.size, ops.detector.nv, ops.detector.nu))
    return ops.images(data, start)


def _residual_views(
    ops: ViewBatches, volume: jax.Array | None, data: jax.Array, *, stream: bool
) -> tuple[jax.Array, jax.Array]:
    """``0.5 ||A x - y||^2`` and ``A^T (A x - y)``, batch by batch; ``x = 0`` for None."""
    project = None if volume is None else ops.projector(volume)

    def body(chunk: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value, gradient = carry
        start, batch, valid = ops.select(chunk)
        measured = _measured(ops, data, start, stream=stream)
        projected = 0.0 if project is None else project(start, batch)
        residual = projected - measured
        residual = jnp.where(valid[:, None, None], residual, 0.0)
        value = value + 0.5 * jnp.vdot(residual, residual).real
        return value, ops.backproject(residual, start, batch, gradient)

    return jax.lax.fori_loop(0, ops.count, body, (jnp.float32(0), ops.zeros_volume()))


def least_squares_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str | ConeModel = "ray",
    joseph_interpolation: str = "linear",
    *,
    stream: bool = False,
    split: ViewSplit | None = None,
) -> Callable[[jax.Array | None, jax.Array], tuple[jax.Array, jax.Array]]:
    """Return the value and gradient of ``0.5 ||A x - y||^2``, ``(x, y) -> (value, A^T r)``.

    Each view batch is projected, compared with the data and transposed before
    the next, so no sinogram-sized intermediate is ever stored. ``x = None`` is
    zero, projected for free. With ``stream``, ``y`` is the key of a host array
    registered with :func:`host_source`, read one batch at a time. With a
    ``split``, the views are shared among its devices; ``y`` is best placed with
    :meth:`ViewSplit.place`. Streamed data cannot be split.
    """
    assert not (stream and split is not None), "streamed data cannot be split"
    ops = ViewBatches(
        poses,
        grid,
        detector,
        None,
        backend,
        batch_size,
        model,
        joseph_interpolation,
        absolute_weights=False,
    )

    def at(share: ViewBatches, *arrays: jax.Array) -> tuple[jax.Array, jax.Array]:
        *volume, data = arrays
        return _residual_views(share, volume[0] if volume else None, data, stream=stream)

    def value_and_gradient(
        volume: jax.Array | None, data: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        shared = () if volume is None else (volume,)
        return _summed(split, ops, at, shared=shared, views=(data,))

    return value_and_gradient


def normal_operator_norm(
    forward: Callable[[jax.Array], jax.Array],
    adjoint: Callable[[jax.Array], jax.Array],
    shape: tuple[int, int, int],
    *,
    iterations: int,
    mask: jax.Array | None = None,
    start: jax.Array | None = None,
) -> jax.Array:
    """Estimate ``||M A^T A M||`` by power iteration from ``start`` (default constant).

    Power iteration approaches the largest eigenvalue from below; callers
    needing a safe step size should add a margin.
    """

    def normal(x: jax.Array) -> jax.Array:
        x = x if mask is None else x * mask
        y = adjoint(forward(x))
        return y if mask is None else y * mask

    def step(_: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x, _ = carry
        return unit(normal(x))

    x0 = jnp.ones(shape, jnp.float32) if start is None else start
    x0 = x0 if mask is None else x0 * mask
    x0, _ = unit(x0)
    _, norm = jax.lax.fori_loop(0, max(1, iterations), step, (x0, jnp.float32(0)))
    return norm


def positive_normal_bound(
    forward: Callable[[jax.Array], jax.Array],
    adjoint: Callable[[jax.Array], jax.Array],
    shape: tuple[int, int, int],
    *,
    iterations: int,
) -> jax.Array:
    """Bound a nonnegative projector's squared norm, starting from a constant volume.

    For a nonnegative normal matrix N and positive q, the Collatz bound
    ``max((N q) / q)`` dominates its spectral radius even when q barely overlaps
    the leading mode; power passes tighten it. Zero coordinates after a pass
    belong to zero rows of N and do not enter the ratio. A small FP32 margin
    covers rounding in the passes.
    """

    def step(_: jax.Array, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        vector, _ = state
        normal = adjoint(forward(vector))
        ratio = normal / jnp.where(vector == 0, 1.0, vector)
        return unit(normal)[0], jnp.max(jnp.where(vector == 0, 0.0, ratio))

    vector = unit(jnp.ones(shape, jnp.float32))[0]
    _, upper = jax.lax.fori_loop(0, max(1, iterations), step, (vector, jnp.float32(0)))
    return upper * (1.0 + 8 * jnp.finfo(jnp.float32).eps)


def normal_equation_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str | ConeModel = "ray",
    joseph_interpolation: str = "linear",
) -> tuple[
    Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    Callable[[jax.Array | None, jax.Array], tuple[jax.Array, jax.Array]],
]:
    """Return ``(normal, gradient)`` streaming data from a :func:`host_source` key.

    ``normal(d)`` returns ``(||A d||^2, A^T A d)`` in one pass over view batches,
    touching no measured data. ``gradient(x, key)`` returns ``A^T (y - A x)`` and
    the cancellation scale ``|A|^T (|y| + |A x|)``, reading each batch of ``y``
    from the host. Only volume-sized arrays and one batch are ever on the device.
    ``gradient(None, key)`` skips the forward projection of a known-zero volume.
    """
    ops = ViewBatches(
        poses,
        grid,
        detector,
        None,
        backend,
        batch_size,
        model,
        joseph_interpolation,
        absolute_weights=False,
    )
    magnitude = (
        ViewBatches(
            poses,
            grid,
            detector,
            None,
            backend,
            batch_size,
            model,
            joseph_interpolation,
            absolute_weights=True,
        )
        if model == "joseph" and joseph_interpolation == "cubic"
        else ops
    )
    shape = (ops.size, detector.nv, detector.nu)

    def normal(direction: jax.Array) -> tuple[jax.Array, jax.Array]:
        project = ops.projector(direction)

        def body(
            chunk: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            energy, total = carry
            start, batch, valid = ops.select(chunk)
            projected = jnp.where(valid[:, None, None], project(start, batch), 0.0)
            energy = energy + jnp.vdot(projected, projected).real
            return energy, ops.backproject(projected, start, batch, total)

        return jax.lax.fori_loop(0, ops.count, body, (jnp.float32(0), ops.zeros_volume()))

    def gradient(volume: jax.Array | None, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        project = None if volume is None else ops.projector(volume)

        def body(
            chunk: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            total, scale = carry
            start, batch, valid = ops.select(chunk)
            measured = read_views(key, start, shape)
            predicted = 0.0 if project is None else project(start, batch)
            mask = valid[:, None, None]
            residual = jnp.where(mask, measured - predicted, 0.0)
            size = jnp.where(mask, jnp.abs(measured) + jnp.abs(predicted), 0.0)
            return (
                ops.backproject(residual, start, batch, total),
                magnitude.backproject(size, start, batch, scale),
            )

        zero = ops.zeros_volume()
        return jax.lax.fori_loop(0, ops.count, body, (zero, zero))

    return normal, gradient


def poisson_operators(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None,
    backend: str,
    batch_size: int,
    model: str | ConeModel,
    indices: jax.Array,
    valid: jax.Array,
    *,
    stream: bool = False,
) -> tuple[
    Callable[[jax.Array], jax.Array], Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
]:
    """Subset ``A.T 1`` and ``A.T (data / A volume)`` with batch-sized image buffers.

    ``indices`` is a (subsets, views) table into the original scan; ``valid``
    marks padding. The returned functions take the subset index, and the
    second also takes the volume and the original-order measurement stack.
    Source arrangements are gathered with their views, including cone segments.
    Only the current batch is gathered from the original-order measurement
    stack. With ``stream``, measurements are a host-source key instead.
    """
    size = int(indices.shape[1])
    chunks = (size + batch_size - 1) // batch_size
    balanced_batch = (size + chunks - 1) // chunks
    ops = ViewBatches(
        poses, grid, detector, det_grid, backend, balanced_batch, model, "linear",
        absolute_weights=False,
    )  # fmt: skip
    per_view = tuple(array[indices] for array in ops.per_view())

    def group(i: jax.Array) -> ViewBatches:
        return ops.share(tuple(array[i] for array in per_view), valid[i])

    return (
        lambda i: _sensitivity(group(i)),
        lambda i, volume, data: _correction(
            group(i),
            volume,
            data,
            indices=indices[i],
            stream=stream,
        ),
    )


def _sensitivity(ops: ViewBatches) -> jax.Array:
    def batch(k: jax.Array, output: jax.Array) -> jax.Array:
        start, poses, valid = ops.select(k)
        ones = jnp.broadcast_to(valid[:, None, None], (ops.size, ops.detector.nv, ops.detector.nu))
        return ops.backproject(ones.astype(jnp.float32), start, poses, output)

    return jax.lax.fori_loop(0, ops.count, batch, ops.zeros_volume())


def _correction(
    ops: ViewBatches,
    volume: jax.Array,
    data: jax.Array,
    *,
    indices: jax.Array,
    stream: bool,
) -> jax.Array:
    project = ops.projector(volume)

    def batch(k: jax.Array, output: jax.Array) -> jax.Array:
        start, poses, valid = ops.select(k)
        expected = project(start, poses)
        positive = expected > 0
        view_ids = jax.lax.dynamic_slice_in_dim(indices, start, ops.size)
        measured = (
            read_views(
                data,
                start,
                (ops.size, ops.detector.nv, ops.detector.nu),
                indices=view_ids,
            )
            if stream
            else data[view_ids]
        )
        ratio = measured / jnp.where(positive, expected, 1.0)
        ratio = jnp.where(valid[:, None, None] & positive, ratio, 0.0)
        return ops.backproject(ratio, start, poses, output)

    return jax.lax.fori_loop(0, ops.count, batch, ops.zeros_volume())
