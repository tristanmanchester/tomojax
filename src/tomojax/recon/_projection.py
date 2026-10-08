"""Batched projection operators shared by the iterative reconstruction solvers.

``projection_operators`` returns a matched forward projector and transpose over
all views, processed in fixed-size view batches inside one compiled loop.
``"joseph"`` samples voxel-centre planes along each ray's dominant axis and is
the fastest model on CUDA; ``"ray"`` is the trilinear ray marcher. Both use
physical units and agree with analytic line integrals to the same accuracy.
Cone-beam geometries use :class:`ConeModel`, Joseph sampling of rays from the
source, with CUDA kernels where the parallel models use Pallas.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from tomojax.core.cone import ConeModel, cone_backproject, cone_model, cone_project, use_cuda_cone
from tomojax.core.geometry.cone import is_cone_beam
from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T
from tomojax.recon._host_stream import read_views

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxlib._jax import Device  # jax.Device, as a type

    from tomojax.geometry import Detector, Grid

type ProjectorModel = Literal["auto", "joseph", "ray"]
type ProjectorBackend = Literal["auto", "jax", "pallas"]


def cuda_available() -> bool:
    """Return whether JAX's default device is a CUDA GPU."""
    return (
        jax.default_backend() == "gpu"
        and "cuda" in jax.devices()[0].client.platform_version.lower()
    )


def resolve_projector(
    model: str,
    backend: str,
    *,
    det_grid: object = None,
    context: str,
) -> tuple[str, str]:
    """Resolve ``auto`` choices; explicit choices must be supported, never replaced.

    ``auto`` selects Joseph plane sampling with Pallas kernels on CUDA. Explicit
    detector grids need the ray model, and Pallas needs CUDA and the canonical
    detector grid.
    """
    if model not in {"auto", "joseph", "ray"}:
        raise ValueError(f"{context}: projector_model must be 'auto', 'joseph' or 'ray'")
    if backend not in {"auto", "jax", "pallas"}:
        raise ValueError(f"{context}: projector_backend must be 'auto', 'jax' or 'pallas'")
    if model == "auto":
        model = "ray" if det_grid is not None else "joseph"
    if model == "joseph" and det_grid is not None:
        raise ValueError(f"{context}: Joseph sampling requires the canonical detector grid")
    cuda = cuda_available()
    if backend == "auto":
        backend = "pallas" if cuda and det_grid is None else "jax"
    if backend == "pallas" and (not cuda or det_grid is not None):
        raise ValueError(f"{context}: Pallas requires CUDA and the canonical detector grid")
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
    solver projects onto (a binned one, say). Their ``pallas`` backend is
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
    if backend not in {"auto", "jax", "pallas"}:
        raise ValueError(f"{context}: projector_backend must be 'auto', 'jax' or 'pallas'")
    cuda = use_cuda_cone()
    if backend == "auto":
        backend = "pallas" if cuda else "jax"
    if backend == "pallas" and not cuda:
        raise ValueError(f"{context}: the CUDA cone kernels need CuPy on a CUDA device")
    cone = cone_model(geometry, detector)
    assert cone is not None
    return cone, backend


_VIEWS = "views"


@dataclass(frozen=True)
class ViewSplit:
    """``n`` views shared evenly among several devices, each holding the whole volume.

    Each device projects its own views and backprojects them into its own volume,
    and the volumes are summed. So that the shares are equal, the views are padded
    to :attr:`total` with copies of the last one that count for nothing: their
    projections are zero and their images are ignored. Sinogram-sized arrays of
    a split have :attr:`total` views, the padding rows zero (see :meth:`place`).
    """

    mesh: Mesh
    n: int

    @property
    def total(self) -> int:
        return -(-self.n // self.mesh.size) * self.mesh.size

    def pad(self, stack: jax.Array) -> jax.Array:
        """``stack`` of ``n`` views, padded with zero views to :attr:`total`."""
        if int(stack.shape[0]) == self.total:
            return stack
        widths = [(0, self.total - int(stack.shape[0]))] + [(0, 0)] * (stack.ndim - 1)
        return jnp.pad(stack, widths)

    def place(self, stack: jax.Array | np.ndarray) -> jax.Array:
        """``stack`` padded and laid out with each device's views on that device."""
        sharding = NamedSharding(self.mesh, PartitionSpec(_VIEWS))
        return jax.device_put(self.pad(jnp.asarray(stack, jnp.float32)), sharding)

    def run[T](
        self,
        ops: _Batches,
        local: Callable[..., T],
        *,
        shared: tuple[jax.Array, ...] = (),
        views: tuple[jax.Array, ...] = (),
        out_specs: PartitionSpec,
    ) -> T:
        """``local(share, *shared, *views)`` on every device, with its share of the views.

        ``shared`` arrays (a volume) are whole on every device; ``views`` stacks
        are cut into the devices' views, padded first if need be.
        """
        per_view = tuple(_pad_edge(a, self.total) for a in ops.per_view())
        live = jnp.arange(self.total) < self.n

        def body(
            per_view: tuple[jax.Array, ...], live: jax.Array, shared: tuple, views: tuple
        ) -> T:
            return local(ops.restricted(per_view, live), *shared, *views)

        split, whole = PartitionSpec(_VIEWS), PartitionSpec()
        # Unchecked: the projectors' loops start from zeros that the device's views
        # then change, which the check would make every loop declare.
        return jax.shard_map(
            body,
            mesh=self.mesh,
            in_specs=(split, split, whole, split),
            out_specs=out_specs,
            check_vma=False,
        )(per_view, live, shared, tuple(self.pad(a) for a in views))


def _summed[T](
    split: ViewSplit | None,
    ops: _Batches,
    local: Callable[..., T],
    *,
    shared: tuple[jax.Array, ...] = (),
    views: tuple[jax.Array, ...] = (),
) -> T:
    """``local(ops, *shared, *views)``; with a ``split``, its sum over the devices' shares."""
    if split is None:
        return local(ops, *shared, *views)

    def on_device(share: _Batches, *arrays: jax.Array) -> T:
        return jax.lax.psum(local(share, *arrays), _VIEWS)

    return split.run(ops, on_device, shared=shared, views=views, out_specs=PartitionSpec())


def view_split(devices: Sequence[Device] | None, n_views: int) -> ViewSplit | None:
    """The split of ``n_views`` among ``devices``, or None for one device (or None)."""
    if devices is None or len(devices) < 2:
        return None
    return ViewSplit(Mesh(np.array(tuple(devices)), (_VIEWS,)), int(n_views))


def _pad_edge(array: jax.Array, total: int) -> jax.Array:
    """Per-view ``array`` padded to ``total`` views by repeating its last view."""
    widths = [(0, total - int(array.shape[0]))] + [(0, 0)] * (array.ndim - 1)
    return jnp.pad(array, widths, mode="edge")


class _Batches:
    """Fixed-size view batches over one pose stack, with matched batch operators."""

    def __init__(
        self,
        poses: jax.Array,
        grid: Grid,
        detector: Detector,
        det_grid: tuple[jax.Array, jax.Array] | None,
        backend: str,
        batch_size: int,
        model: str | ConeModel,
        joseph_interpolation: str,
        *,
        absolute_weights: bool,
    ) -> None:
        self.batch_size = batch_size
        self.n = int(poses.shape[0])
        self.size = min(batch_size, self.n)
        self.count = (self.n + self.size - 1) // self.size
        self.poses, self.grid, self.detector, self.det_grid = poses, grid, detector, det_grid
        self.live: jax.Array | None = None  # which views count, on a device's share
        self.coefficients: jax.Array | None = None
        self.backend, self.model = backend, model
        self.interpolation, self.absolute_weights = joseph_interpolation, absolute_weights
        if isinstance(model, ConeModel):
            # Cone weights are non-negative, so |A|^T is the transpose itself.
            self.coefficients = model.coefficients(poses, grid)
            self.cone_backend = "cuda" if backend == "pallas" else "jax"
        elif model == "joseph":
            from tomojax.core.joseph import plane_coefficients

            self.coefficients = plane_coefficients(poses, grid, detector)

    def per_view(self) -> tuple[jax.Array, ...]:
        """The per-view arrays a device's share of the views is cut from."""
        return (self.poses,) if self.coefficients is None else (self.poses, self.coefficients)

    def restricted(self, per_view: tuple[jax.Array, ...], live: jax.Array) -> _Batches:
        """These operators over the views of ``per_view``, of which ``live`` count."""
        share = copy.copy(self)
        share.poses = per_view[0]
        share.coefficients = per_view[1] if len(per_view) > 1 else None
        share.n = int(share.poses.shape[0])
        share.size = min(self.batch_size, share.n)
        share.count = (share.n + share.size - 1) // share.size
        share.live = live
        return share

    def select(self, chunk: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        start = jnp.minimum(chunk * self.size, self.n - self.size)
        batch = jax.lax.dynamic_slice(self.poses, (start, 0, 0), (self.size, 4, 4))
        # The final shifted batch overlaps earlier views. Its adjoint must count
        # only new views; forward writes the same overlapping values again.
        valid = start + jnp.arange(self.size) >= chunk * self.size
        if self.live is not None:
            valid = valid & self.counted(start)
        return start, batch, valid

    def counted(self, start: jax.Array) -> jax.Array:
        """Which of the batch's views count: all but a split's padding."""
        if self.live is None:
            return jnp.ones((self.size,), bool)
        return jax.lax.dynamic_slice(self.live, (start,), (self.size,))

    def images(self, stack: jax.Array, start: jax.Array) -> jax.Array:
        shape = (self.size, self.detector.nv, self.detector.nu)
        return jax.lax.dynamic_slice(stack, (start, 0, 0), shape)

    def _coefficients(self, start: jax.Array) -> jax.Array:
        assert self.coefficients is not None
        width = int(self.coefficients.shape[1])
        return jax.lax.dynamic_slice(self.coefficients, (start, 0), (self.size, width))

    def project(self, volume: jax.Array, start: jax.Array, batch: jax.Array) -> jax.Array:
        grid, detector = self.grid, self.detector
        if isinstance(self.model, ConeModel):
            return cone_project(
                volume, self._coefficients(start), grid, detector, backend=self.cone_backend
            )
        if self.model == "joseph":
            from tomojax.core.joseph import forward_project_planes

            return forward_project_planes(
                self._coefficients(start),
                volume,
                grid,
                detector,
                backend=self.backend,
                interpolation=self.interpolation,
            )
        if self.backend == "pallas":
            from tomojax.core.pallas.api import (
                PallasProjectorOptions,
                forward_project_views_T_pallas,
            )

            return forward_project_views_T_pallas(
                batch,
                grid,
                detector,
                volume,
                options=PallasProjectorOptions(tile_shape=(16, 4), num_warps=1),
            )
        return jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, volume, det_grid=self.det_grid)
        )(batch)

    def backproject(
        self, images: jax.Array, start: jax.Array, batch: jax.Array, output: jax.Array
    ) -> jax.Array:
        """Return ``output`` plus the batch transpose; Joseph on CUDA adds in place."""
        grid, detector = self.grid, self.detector
        if isinstance(self.model, ConeModel):
            return cone_backproject(
                images,
                self._coefficients(start),
                grid,
                detector,
                backend=self.cone_backend,
                accumulate=output,
            )
        if self.model == "joseph":
            from tomojax.core.joseph import sum_backproject_planes

            return sum_backproject_planes(
                self._coefficients(start),
                images,
                grid,
                detector,
                backend=self.backend,
                interpolation=self.interpolation,
                absolute_weights=self.absolute_weights,
                accumulate=output,
            )
        if self.backend == "pallas":
            from tomojax.core.pallas.api import sum_backproject_views_T_pallas

            update = sum_backproject_views_T_pallas(
                batch, grid, detector, images, tile_shape=(16, 4), num_warps=1
            )
        else:
            update = sum_backproject_views_T(batch, grid, detector, images, det_grid=self.det_grid)
        return output + update

    def zeros_volume(self) -> jax.Array:
        return jnp.zeros((self.grid.nx, self.grid.ny, self.grid.nz), dtype=jnp.float32)


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
    adds the result to an existing volume, in place for Joseph on CUDA. With a
    ``split``, the views are shared among its devices and the sinograms both
    take and return have its padded :attr:`ViewSplit.total` views.
    """
    ops = _Batches(
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
        return split.run(ops, _forward_views, shared=(volume,), out_specs=PartitionSpec(_VIEWS))

    def adjoint(
        images: jax.Array | tuple[jax.Array, ...],
        combine: Callable[..., jax.Array] | None = None,
        accumulate: jax.Array | None = None,
    ) -> jax.Array:
        stacks = tuple(images) if isinstance(images, tuple | list) else (images,)
        if split is None:
            initial = ops.zeros_volume() if accumulate is None else accumulate
            return _adjoint_views(ops, stacks, combine, initial)

        def local(share: _Batches, *stacks: jax.Array) -> jax.Array:
            return _adjoint_views(share, stacks, combine, share.zeros_volume())

        total = _summed(split, ops, local, views=stacks)
        return total if accumulate is None else accumulate + total

    return forward, adjoint


def _forward_views(ops: _Batches, volume: jax.Array) -> jax.Array:
    def body(chunk: jax.Array, output: jax.Array) -> jax.Array:
        start, batch, _ = ops.select(chunk)
        projected = ops.project(volume, start, batch)
        if ops.live is not None:  # a split's padding projects to zero
            projected = jnp.where(ops.counted(start)[:, None, None], projected, 0.0)
        return jax.lax.dynamic_update_slice(output, projected, (start, 0, 0))

    empty = jnp.zeros((ops.n, ops.detector.nv, ops.detector.nu), dtype=jnp.float32)
    return jax.lax.fori_loop(0, ops.count, body, empty)


def _adjoint_views(
    ops: _Batches,
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


def _measured(ops: _Batches, data: jax.Array, start: jax.Array, *, stream: bool) -> jax.Array:
    """The batch at ``start`` of ``data``, or of the host source ``data`` names."""
    if stream:
        return read_views(data, start, (ops.size, ops.detector.nv, ops.detector.nu))
    return ops.images(data, start)


def _residual_views(
    ops: _Batches, volume: jax.Array, data: jax.Array, *, stream: bool
) -> tuple[jax.Array, jax.Array]:
    """``0.5 ||A x - y||^2`` and ``A^T (A x - y)``, batch by batch."""

    def body(chunk: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value, gradient = carry
        start, batch, valid = ops.select(chunk)
        residual = ops.project(volume, start, batch) - _measured(ops, data, start, stream=stream)
        residual = jnp.where(valid[:, None, None], residual, 0.0)
        value = value + 0.5 * jnp.vdot(residual, residual).real
        return value, ops.backproject(residual, start, batch, gradient)

    return jax.lax.fori_loop(0, ops.count, body, (jnp.float32(0), ops.zeros_volume()))


def _data_views(ops: _Batches, data: jax.Array, *, stream: bool) -> tuple[jax.Array, jax.Array]:
    """``0.5 ||y||^2`` and ``A^T y``, batch by batch."""

    def body(chunk: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        value, backprojection = carry
        start, batch, valid = ops.select(chunk)
        measured = jnp.where(valid[:, None, None], _measured(ops, data, start, stream=stream), 0.0)
        value = value + 0.5 * jnp.vdot(measured, measured).real
        return value, ops.backproject(measured, start, batch, backprojection)

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
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Return the value and gradient of ``0.5 ||A x - y||^2``, ``(x, y) -> (value, A^T r)``.

    Each view batch is projected, compared with the data and transposed before
    the next, so no sinogram-sized intermediate is ever stored. With
    ``stream``, ``y`` is the key of a host array registered with
    :func:`host_source`, read one batch at a time. With a ``split``, the views
    are shared among its devices; ``y`` is best placed with :meth:`ViewSplit.place`.
    """
    if stream and split is not None:
        raise ValueError("streamed data cannot be split among devices")
    ops = _Batches(
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

    def value_and_gradient(volume: jax.Array, data: jax.Array) -> tuple[jax.Array, jax.Array]:
        local = partial(_residual_views, stream=stream)
        return _summed(split, ops, local, shared=(volume,), views=(data,))

    return value_and_gradient


def least_squares_at_zero(
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
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Return ``y -> (0.5 ||y||^2, A^T y)``, least squares and its negated gradient at zero.

    These are the value of ``0.5 ||A x - y||^2`` at ``x = 0`` and minus its
    gradient, with no projection of the zero volume. ``stream`` and ``split``
    are as for :func:`least_squares_operators`.
    """
    if stream and split is not None:
        raise ValueError("streamed data cannot be split among devices")
    ops = _Batches(
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

    def at_zero(data: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _summed(split, ops, partial(_data_views, stream=stream), views=(data,))

    return at_zero


def normal_operator_norm(
    forward: Callable[[jax.Array], jax.Array],
    adjoint: Callable[[jax.Array], jax.Array],
    shape: tuple[int, int, int],
    *,
    iters: int,
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
        y = normal(x)
        norm = jnp.sqrt(jnp.sum(y * y))
        return y / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny), norm

    x0 = jnp.ones(shape, jnp.float32) if start is None else start
    x0 = x0 if mask is None else x0 * mask
    x0 = x0 / jnp.sqrt(jnp.sum(x0**2) + 1e-30)
    _, norm = jax.lax.fori_loop(0, max(1, iters), step, (x0, jnp.float32(0)))
    return norm


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
    Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
]:
    """Return ``(normal, gradient)`` streaming data from a :func:`host_source` key.

    ``normal(d)`` returns ``(||A d||^2, A^T A d)`` in one pass over view batches,
    touching no measured data. ``gradient(x, key)`` returns ``A^T (y - A x)`` and
    the cancellation scale ``|A|^T (|y| + |A x|)``, reading each batch of ``y``
    from the host. Only volume-sized arrays and one batch are ever on the device.
    """
    ops = _Batches(
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
        _Batches(
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
        def body(
            chunk: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            energy, total = carry
            start, batch, valid = ops.select(chunk)
            projected = jnp.where(valid[:, None, None], ops.project(direction, start, batch), 0.0)
            energy = energy + jnp.vdot(projected, projected).real
            return energy, ops.backproject(projected, start, batch, total)

        return jax.lax.fori_loop(0, ops.count, body, (jnp.float32(0), ops.zeros_volume()))

    def gradient(volume: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        def body(
            chunk: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            total, scale = carry
            start, batch, valid = ops.select(chunk)
            measured = read_views(key, start, shape)
            predicted = ops.project(volume, start, batch)
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
