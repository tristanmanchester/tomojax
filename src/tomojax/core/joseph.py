"""Plane-sampled parallel-ray model and a JAX reference implementation.

Choose the fastest ray direction in voxel coordinates. Sample its voxel-centre
planes, interpolating the other two coordinates with zero padding. Bilinear
interpolation is the default; cubic convolution is an explicit alternative.
Multiply by the physical distance between planes. The resulting linear operator
differs from the ray-entry-phased trilinear sampler in ``projector.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.volume_layout import swap_private_axes

from ._plane_interpolation import cubic_weights_and_derivatives, validate_interpolation
from .compilation_cache import enable_persistent_compilation_cache
from .geometry.base import grid_volume_origin

if TYPE_CHECKING:
    from .geometry.base import Detector, Grid

enable_persistent_compilation_cache()


def validate_plane_geometry(poses: jax.Array, grid: Grid, detector: Detector) -> None:
    """Reject geometry that would make the plane map or gather bounds undefined."""
    if not np.isfinite((*grid_volume_origin(grid), *detector.center)).all():
        raise ValueError("cgls: Joseph grid origin and detector centre must be finite")
    host_poses = np.asarray(poses)
    rotations = host_poses[:, :3, :3]
    if not (
        np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-5)
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-5)
        and np.allclose(host_poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=2e-5)
    ):
        raise ValueError("cgls: Joseph sampling requires rigid homogeneous poses")


def select_axis(
    axis: jax.Array, x: jax.Array | int, y: jax.Array | int, z: jax.Array | int
) -> jax.Array:
    """Select an integer coordinate with explicit dtypes."""
    return jnp.where(
        axis == 0,
        jnp.asarray(x, jnp.int32),
        jnp.where(axis == 1, jnp.asarray(y, jnp.int32), jnp.asarray(z, jnp.int32)),
    )


def plane_coefficients(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Prepare a dynamic per-view affine mapping from detector pixels to planes.

    Columns are ``axis, ub, vb, kb, cb, uc, vc, kc, cc, ds, inv00, inv01,
    inv10, inv11``. At plane k, q_b = ub*u + vb*v + kb*k + cb, and similarly
    q_c. The inverse 2x2 map bounds the rays that contribute to each voxel.
    Inputs are rigid world-from-object transforms. ``det_grid`` gives flattened
    world (x, z) pixel positions, u fastest; it must be an affine lattice, such
    as a calibrated detector offset or roll, and is read from three pixels.
    """
    from ._plane_coefficients import coefficients

    return coefficients(poses, grid, detector, det_grid, xp=jnp)


def forward_jax(
    coefficients: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Reference model with ordinary AD; absolute weights are for roundoff bounds."""
    validate_interpolation(interpolation)
    shape = (grid.nx, grid.ny, grid.nz)
    u, v = jnp.meshgrid(jnp.arange(detector.nu), jnp.arange(detector.nv))

    def one_view(cf: jax.Array) -> jax.Array:
        axis = cf[0].astype(jnp.int32)
        ub, vb, kb, cb, uc, vc, kc, cc, weight = cf[1:10]
        na = select_axis(axis, *shape)
        nb, nc = (
            select_axis(axis, shape[1], shape[2], shape[0]),
            select_axis(axis, shape[2], shape[0], shape[1]),
        )

        def step(k: jax.Array, accumulator: jax.Array) -> jax.Array:
            qb, qc = (ub * u + vb * v) + (kb * k + cb), (uc * u + vc * v) + (kc * k + cc)
            ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
            wb, wc = qb - ib, qc - ic

            def sample(b: jax.Array, c: jax.Array) -> jax.Array:
                ix, iy, iz = (
                    select_axis(axis, k, c, b),
                    select_axis(axis, b, k, c),
                    select_axis(axis, c, b, k),
                )
                valid = (k < na) & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
                value = volume[
                    jnp.clip(iz, 0, shape[2] - 1),
                    jnp.clip(iy, 0, shape[1] - 1),
                    jnp.clip(ix, 0, shape[0] - 1),
                ]
                return jnp.where(valid, value, 0.0)

            if interpolation == "linear":
                value = (sample(ib, ic) * (1 - wc) + sample(ib, ic + 1) * wc) * (1 - wb) + (
                    sample(ib + 1, ic) * (1 - wc) + sample(ib + 1, ic + 1) * wc
                ) * wb
            else:
                weights_b, _ = cubic_weights_and_derivatives(wb)
                weights_c, _ = cubic_weights_and_derivatives(wc)
                if absolute_weights:
                    weights_b = tuple(jnp.abs(w) for w in weights_b)
                    weights_c = tuple(jnp.abs(w) for w in weights_c)
                value = jnp.zeros_like(accumulator)
                # Match the CUDA recurrence to keep near-zero residuals from
                # amplifying differences in summation order. The derivative
                # reference still uses ordinary JAX autodiff.
                for row in range(4):
                    row_value = jnp.zeros_like(accumulator)
                    for col in range(4):
                        row_value = row_value + sample(ib + row - 1, ic + col - 1) * weights_c[col]
                    value = value + row_value * weights_b[row]
            return accumulator + value * weight

        return jax.lax.fori_loop(
            0, max(shape), step, jnp.zeros((detector.nv, detector.nu), jnp.float32)
        )

    return jax.vmap(one_view)(coefficients)


def adjoint_jax(
    coefficients: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Reference transpose; ``absolute_weights`` selects abs(A).T for bounds."""
    zero = jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32)
    return jax.linear_transpose(
        lambda x: forward_jax(
            coefficients,
            x,
            grid,
            detector,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        ),
        zero,
    )(images)[0]


@dataclass(frozen=True)
class PrivateJoseph:
    """The plane model over solver-private CUDA volumes: (nx, ny, nz), z contiguous.

    Resident solves keep their volumes in this storage so the kernels read z
    contiguously, as rays of every view cross z planes; they convert at entry
    and exit. Coordinates and geometry are unchanged.
    """

    private_storage: ClassVar[bool] = True

    def volume_shape(self, grid: Grid) -> tuple[int, int, int]:
        """The buffer shape of a volume on ``grid``."""
        return (grid.nx, grid.ny, grid.nz)


def forward_kind(
    coefficients: jax.Array, grid: Grid, detector: Detector
) -> Literal["bands", "axial", "generic"]:
    """Which CUDA forward these views take, as one call over all of them.

    Aligned axial views take the four-row forward, which stages its own
    z-fast bands, so a private z-fast copy cannot help them. Other axial views
    take the scalar axial kernel and the rest the generic one, from z-fast
    storage: a copy or bands.
    """
    from ._cuda_axial import aligned_axial, axial_coefficients
    from ._cuda_joseph import use_cuda_gather

    if not use_cuda_gather(grid, axial=True) or not bool(axial_coefficients(coefficients)):
        return "generic"
    if grid.nz % 4 == 0 and detector.nv >= 4 and bool(aligned_axial(coefficients)):
        return "bands"
    return "axial"


def transpose_bands(
    coefficients: jax.Array, grid: Grid, *, batch: int
) -> Literal["axial", "gather"] | None:
    """How the CUDA transpose of these views fills its own z-fast bands, if it does.

    All-axial views take the separable transpose; other views gather, unless a
    ``batch`` of them is axial, which the operators give the separable transpose.
    Either fills 32 planes at a time, so a private z-fast copy cannot help them.
    """
    from ._cuda_axial import axial_coefficients
    from ._cuda_joseph import use_cuda_gather

    if not use_cuda_gather(grid, axial=True):
        return None
    if bool(axial_coefficients(coefficients)):
        return "axial"
    views = coefficients.shape[0]
    size = min(batch, views)
    # The operators' batches, the last one shifted back to end at the last view.
    starts = {min(first, views - size) for first in range(0, views, size)}
    axial = (bool(axial_coefficients(coefficients[s : s + size])) for s in starts)
    return "gather" if use_cuda_gather(grid) and not any(axial) else None


def forward_bands(
    coefficients: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    depth: int,
    axial: bool,
) -> jax.Array:
    """The linear CUDA forward of a (z, y, x) volume from z-fast bands of ``depth`` < nz planes.

    ``axial`` views take the scalar axial kernel, others the generic one (see
    :func:`forward_kind`), with the same bits as from a z-fast copy of the
    volume but one band of workspace. Off a GPU it is the JAX model.
    """
    from ._cuda_axial import forward_bands_cuda

    def on_cuda(c: jax.Array, x: jax.Array) -> jax.Array:
        return forward_bands_cuda(c, x, grid, detector, depth=depth, axial=axial)

    def on_jax(c: jax.Array, x: jax.Array) -> jax.Array:
        return forward_jax(c, x, grid, detector)

    return jax.lax.platform_dependent(coefficients, volume, cuda=on_cuda, default=on_jax)


def backproject_bands(
    coefficients: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    batch: int,
    axial: bool,
) -> jax.Array:
    """The CUDA transpose into a new (z, y, x) volume, 32 z-fast planes at a time.

    It adds ``batch`` views at a time, as one call per batch would, and stages
    each band of the volume once for all of them: ``axial`` views by the
    separable transpose, others by the gather. Off a GPU it is the JAX model.
    """
    from ._cuda_axial import transpose_axial_vu_cuda
    from ._cuda_joseph import gather_bands_cuda

    def on_cuda(c: jax.Array, y: jax.Array) -> jax.Array:
        if axial:
            return transpose_axial_vu_cuda(
                c, y, None, grid=grid, interpolation="linear", batch=batch
            )
        return gather_bands_cuda(c, y, grid, batch=batch)

    def on_jax(c: jax.Array, y: jax.Array) -> jax.Array:
        initial = jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=y.dtype)
        n = int(c.shape[0])
        if n == 0:
            return initial
        size = min(batch, n)

        def add_views(i: jax.Array, volume: jax.Array) -> jax.Array:
            start = jnp.minimum(i * size, n - size)
            cf = jax.lax.dynamic_slice_in_dim(c, start, size)
            images = jax.lax.dynamic_slice_in_dim(y, start, size)
            # The shifted last batch overlaps earlier views; count each once.
            valid = start + jnp.arange(size) >= i * size
            return volume + adjoint_jax(cf, images * valid[:, None, None], grid, detector)

        return jax.lax.fori_loop(0, (n + size - 1) // size, add_views, initial)

    return jax.lax.platform_dependent(coefficients, images, cuda=on_cuda, default=on_jax)


def forward_project_planes(
    coefficients: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
    absolute_weights: bool = False,
    z_fast: bool = False,
) -> jax.Array:
    """Apply a prepared plane model, or abs(A) for positive sensitivity bounds.

    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    The CUDA backend runs the CUDA kernels where the program runs on a GPU,
    and the JAX model elsewhere.
    """

    def on_jax(c: jax.Array, v: jax.Array) -> jax.Array:
        v = swap_private_axes(v, z_fast=z_fast)
        options = {"interpolation": interpolation, "absolute_weights": absolute_weights}
        return forward_jax(c, v, grid, detector, **options)

    if backend == "jax":
        return on_jax(coefficients, volume)
    if backend != "cuda":
        raise ValueError("Plane projection backend must be 'jax' or 'cuda'")

    def on_cuda(c: jax.Array, v: jax.Array) -> jax.Array:
        if absolute_weights:
            from .cuda_projector._joseph import forward_cuda

            options = {"interpolation": interpolation, "absolute_weights": True}
            return forward_cuda(c, v, grid, detector, **options, z_fast=z_fast)
        from .cuda_projector._joseph_autodiff import differentiable_forward

        return differentiable_forward(
            c, v, grid, detector, interpolation=interpolation, z_fast=z_fast
        )

    return jax.lax.platform_dependent(coefficients, volume, cuda=on_cuda, default=on_jax)


def sum_backproject_planes(
    coefficients: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
    absolute_weights: bool = False,
    accumulate: jax.Array | None = None,
    z_fast: bool = False,
) -> jax.Array:
    """Apply the matched transpose, or abs(A).T when estimating cancellation.

    A given ``accumulate`` volume is added to the result; CUDA adds in place.
    The volume is (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    The CUDA backend runs the CUDA kernels where the program runs on a GPU,
    and the JAX model elsewhere.
    """

    def on_jax(c: jax.Array, y: jax.Array, *initial: jax.Array) -> jax.Array:
        options = {"interpolation": interpolation, "absolute_weights": absolute_weights}
        volume = adjoint_jax(c, y, grid, detector, **options)
        volume = swap_private_axes(volume, z_fast=z_fast)
        return volume + initial[0] if initial else volume

    initial = () if accumulate is None else (accumulate,)
    if backend == "jax":
        return on_jax(coefficients, images, *initial)
    if backend != "cuda":
        raise ValueError("Plane projection backend must be 'jax' or 'cuda'")

    def on_cuda(c: jax.Array, y: jax.Array, *initial: jax.Array) -> jax.Array:
        from .cuda_projector._joseph import adjoint_cuda

        options = {"interpolation": interpolation, "absolute_weights": absolute_weights}
        accumulate = initial[0] if initial else None
        return adjoint_cuda(c, y, grid, detector, **options, accumulate=accumulate, z_fast=z_fast)

    return jax.lax.platform_dependent(coefficients, images, *initial, cuda=on_cuda, default=on_jax)


def plane_pose_columns(
    coefficients: jax.Array,
    volume: jax.Array,
    tangent: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
    z_fast: bool = False,
) -> jax.Array:
    """Values and directional derivatives, shaped ``(views, 1 + directions, nv, nu)``.

    ``tangent`` holds ``(views, directions, 14)`` coefficient derivatives.
    CUDA shares each voxel sample among all directions in one ray traversal.
    """
    if backend == "cuda":
        from ._cuda_joseph import cuda_gather_available
        from ._cuda_joseph_columns import pose_columns_cuda

        if cuda_gather_available():
            return pose_columns_cuda(
                coefficients,
                volume,
                tangent,
                grid,
                detector,
                interpolation=interpolation,
                z_fast=z_fast,
            )

    def project(cf: jax.Array) -> jax.Array:
        return forward_project_planes(
            cf, volume, grid, detector, backend=backend, interpolation=interpolation, z_fast=z_fast
        )

    value = project(coefficients)
    columns = jax.vmap(
        lambda dc: jax.jvp(project, (coefficients,), (dc,))[1], in_axes=1, out_axes=1
    )(tangent)
    return jnp.concatenate((value[:, None], columns), axis=1)


def plane_weighted_diagonal(
    coefficients: jax.Array,
    sqrtweights: jax.Array,
    accumulate: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
) -> jax.Array:
    """Add diag(A.T W A) of the linear plane projector to ``accumulate`` (nz, ny, nx).

    ``sqrtweights`` holds the square roots of the pixel weights, shaped
    ``(views, nv, nu)``. A Jacobi preconditioner input; the matched transpose
    is :func:`sum_backproject_planes`.
    """
    from ._weighted_diagonal import weighted_diagonal

    return weighted_diagonal(coefficients, sqrtweights, accumulate, grid, detector, backend=backend)


def plane_l2_value_and_grad(
    coefficients: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return half squared error, coefficient gradient and volume gradient."""
    if backend == "cuda":
        from .cuda_projector._joseph_derivatives import l2_loss_and_grads

        return l2_loss_and_grads(
            coefficients, volume, target, grid, detector, interpolation=interpolation
        )
    if backend == "jax":

        def loss(cf: jax.Array, x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(
                (forward_jax(cf, x, grid, detector, interpolation=interpolation) - target) ** 2
            )

        value, (coefficient_grad, volume_grad) = jax.value_and_grad(loss, argnums=(0, 1))(
            coefficients, volume
        )
        return value, coefficient_grad, volume_grad
    raise ValueError("Plane projection backend must be 'jax' or 'cuda'")


def plane_l2_normal_equations(
    coefficients: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    directions: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-view loss, directional gradient, Gauss-Newton matrix and residuals."""
    if backend == "cuda":
        from .cuda_projector._joseph_derivatives import pose_normal_equations

        return pose_normal_equations(
            coefficients, volume, target, directions, grid, detector, interpolation=interpolation
        )
    if backend != "jax":
        raise ValueError("Plane projection backend must be 'jax' or 'cuda'")

    def project(cf: jax.Array) -> jax.Array:
        return forward_jax(cf, volume, grid, detector, interpolation=interpolation)

    prediction = project(coefficients)
    # Direct JVPs keep ordinary forward mode; linearize would retain a large
    # per-plane residual tape before applying the tangent directions.
    jacobian = jax.vmap(
        lambda dc: jax.jvp(project, (coefficients,), (dc,))[1], in_axes=-1, out_axes=-1
    )(directions)
    jacobian = jacobian.reshape(coefficients.shape[0], -1, directions.shape[-1])
    residual = prediction - target
    flat = residual.reshape(coefficients.shape[0], -1)
    gradient = jnp.einsum("nmp,nm->np", jacobian, flat, precision=jax.lax.Precision.HIGHEST)
    matrix = jnp.einsum("nmp,nmq->npq", jacobian, jacobian, precision=jax.lax.Precision.HIGHEST)
    return 0.5 * jnp.sum(flat**2, axis=1), gradient, matrix, residual


def project_ones_native(coeff: jax.Array, grid: Grid, detector: Detector) -> jax.Array | None:
    """Project a uniform unit volume with the native CUDA gather, or return None.

    Parameters
    ----------
    coeff : jax.Array
        Prepared plane coefficients of the views, from ``plane_coefficients``.
    grid : Grid
        Volume grid.
    detector : Detector
        Detector the coefficients map onto.

    Returns
    -------
    jax.Array or None
        Projections of ones with the ordinary boundary weights, or None when the
        native CUDA gather does not apply to this grid.
    """
    from ._cuda_joseph import project_ones_cuda, use_cuda_gather

    return project_ones_cuda(coeff, grid, detector) if use_cuda_gather(grid) else None
