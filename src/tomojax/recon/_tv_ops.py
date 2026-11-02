"""Discrete isotropic-TV values, derivatives, and constrained proximal maps."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .types import Regulariser


def _forward_difference(u: jnp.ndarray, axis: int) -> jnp.ndarray:
    edge = list(u.shape)
    edge[axis] = 1
    return jnp.concatenate([jnp.diff(u, axis=axis), jnp.zeros(edge, u.dtype)], axis=axis)


def grad3(
    u: jnp.ndarray,
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Neumann forward differences along physical x, y and z.

    A volume is (z, y, x), or solver-private (x, y, z) with ``z_fast``.
    ``gradient_scale_xyz`` multiplies the physical-axis differences in either layout.
    """
    x, z = (0, 2) if z_fast else (2, 0)
    cx, cy, cz = gradient_scale_xyz
    return (
        cx * _forward_difference(u, x),
        cy * _forward_difference(u, 1),
        cz * _forward_difference(u, z),
    )


def gradient_norm_squared(
    shape: tuple[int, ...],
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
) -> float:
    """Exact squared norm of the Neumann forward differences on a finite grid.

    An n-voxel axis contributes ``c² * (2 + 2*cos(pi/n))`` (zero for a singleton),
    where c is its physical-axis gradient scale.
    The axis normal matrices form a Kronecker sum, so their maxima add.
    """
    scales = gradient_scale_xyz if z_fast else gradient_scale_xyz[::-1]
    return sum(
        c * c * (2 + 2 * math.cos(math.pi / n)) for n, c in zip(shape, scales, strict=True) if n > 1
    )


def _backward_difference(p: jnp.ndarray, axis: int) -> jnp.ndarray:
    n = p.shape[axis]
    if n == 1:
        return jnp.zeros_like(p)

    def take(first: int, stop: int) -> jnp.ndarray:
        return jax.lax.slice_in_dim(p, first, stop, axis=axis)

    middle = [take(1, n - 1) - take(0, n - 2)] if n > 2 else []
    return jnp.concatenate([take(0, 1), *middle, -take(n - 2, n - 1)], axis=axis)


def div3(
    px: jnp.ndarray,
    py: jnp.ndarray,
    pz: jnp.ndarray,
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
) -> jnp.ndarray:
    """Discrete divergence matching the TV updates: ``div3 = -grad3*``."""
    x, z = (0, 2) if z_fast else (2, 0)
    cx, cy, cz = gradient_scale_xyz
    return (
        cx * _backward_difference(px, x)
        + cy * _backward_difference(py, 1)
        + cz * _backward_difference(pz, z)
    )


def validate_regulariser(
    regulariser: str,
    huber_delta: float,
    *,
    context: str,
) -> Regulariser:
    """Validate a regulariser config and return its normalized name."""
    reg = str(regulariser).lower()
    if reg not in ("tv", "huber_tv"):
        raise ValueError(f"{context}: regulariser must be 'tv' or 'huber_tv'; got {regulariser!r}")
    if reg == "huber_tv":
        delta = float(huber_delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"{context}: huber_delta must be positive for huber_tv")
    return cast("Regulariser", reg)


def _gradient_norm_sq(
    u: jnp.ndarray,
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
) -> jnp.ndarray:
    gx, gy, gz = grad3(u, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz)
    return gx * gx + gy * gy + gz * gz


def _norm_with_zero_subgradient(squared: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a norm safely inside the smooth regions of Huber formulas.

    The enclosing gradient/prox is differentiable at zero even though the
    Euclidean norm alone is not. Mask before sqrt: masking only its result
    leaves an infinite derivative that can produce NaNs in JVPs and VJPs.
    Nonzero norms and nonfinite inputs retain their ordinary values.
    """
    zero = squared == 0
    safe = jnp.where(zero, jnp.ones_like(squared), squared)
    return jnp.where(zero, jnp.zeros_like(squared), jnp.sqrt(safe))


def isotropic_tv_value(
    u: jnp.ndarray,
    eps: float = 0.0,
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
) -> jnp.ndarray:
    """Return isotropic 3D TV using the repo's forward-difference operator."""
    sq = _gradient_norm_sq(u, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz)
    if eps == 0.0:
        return jnp.sum(jnp.sqrt(sq))
    return jnp.sum(jnp.sqrt(sq + jnp.asarray(eps, dtype=u.dtype)))


def huber_tv_value(
    u: jnp.ndarray,
    delta: float | jax.Array,
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
) -> jnp.ndarray:
    """Return isotropic Huber-TV with a quadratic basin of radius ``delta``."""
    delta_arr = jnp.asarray(delta, dtype=u.dtype)
    sq = _gradient_norm_sq(u, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz)
    tiny = jnp.asarray(jnp.finfo(u.dtype).tiny, dtype=u.dtype)
    norm = jnp.sqrt(jnp.maximum(sq, tiny))
    quadratic = 0.5 * sq / delta_arr
    linear = norm - 0.5 * delta_arr
    return jnp.sum(jnp.where(norm <= delta_arr, quadratic, linear))


def _huber_tv_grad(
    u: jnp.ndarray,
    delta: float | jax.Array,
    *,
    gradient_scale_xyz: tuple[float, float, float],
    z_fast: bool,
) -> jnp.ndarray:
    """Return the gradient of isotropic Huber-TV under ``grad3``/``div3``."""
    delta_arr = jnp.asarray(delta, dtype=u.dtype)
    gx, gy, gz = grad3(u, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz)
    norm = _norm_with_zero_subgradient(gx * gx + gy * gy + gz * gz)
    denom = jnp.maximum(norm, delta_arr)
    qx = gx / denom
    qy = gy / denom
    qz = gz / denom
    return -div3(qx, qy, qz, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz)


_HUBER_TILE_BYTES = 64 << 20


def huber_tv_grad(
    u: jnp.ndarray,
    delta: float | jax.Array,
    *,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_fast: bool = False,
    base: jnp.ndarray | None = None,
    scale: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """``scale`` times the Huber gradient, added to ``base`` when given.

    Scratch is bounded and derivatives exact: each output uses one neighbour along
    each axis, and haloed slabs preserve the Neumann boundary and mixed components
    without storing full gradient fields. A slab's terms are added into ``base`` as
    it goes, so a solver's data gradient takes the regulariser with no second volume.
    """
    scale = jnp.asarray(scale, u.dtype)
    if 5 * u.size * u.dtype.itemsize <= _HUBER_TILE_BYTES:
        scaled = scale * _huber_tv_grad(
            u, delta, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz
        )
        return scaled if base is None else base + scaled
    axis = max(range(3), key=u.shape.__getitem__)
    length = u.shape[axis]
    plane_bytes = u.dtype.itemsize * u.size // length
    core = min(length, max(1, _HUBER_TILE_BYTES // (8 * plane_bytes)))
    width = min(length, core + 2)
    shape = [1, 1, 1]
    shape[axis] = core
    offsets = jnp.arange(core).reshape(shape)

    def body(i, out):
        first = jnp.minimum(i * core, length - core)
        start = jnp.clip(first - 1, 0, length - width)
        tile = jax.lax.dynamic_slice_in_dim(u, start, width, axis=axis)
        gradient = _huber_tv_grad(tile, delta, z_fast=z_fast, gradient_scale_xyz=gradient_scale_xyz)
        local = scale * jax.lax.dynamic_slice_in_dim(gradient, first - start, core, axis=axis)
        if base is not None:
            # The final short slab overlaps its predecessor: add only the new planes.
            old = jax.lax.dynamic_slice_in_dim(out, first, core, axis=axis)
            local = jnp.where(first + offsets >= i * core, old + local, old)
        # Without a base, the overlap is overwritten with the same exact values.
        return jax.lax.dynamic_update_slice_in_dim(out, local, first, axis=axis)

    initial = jnp.zeros_like(u) if base is None else base
    return jax.lax.fori_loop(0, (length + core - 1) // core, body, initial)


def project_tv_duals(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    p3: jnp.ndarray,
    *,
    radius: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project vector fields onto their requested ball, without an absolute floor.

    Normalize before squaring to avoid norm underflow/overflow at small/large
    physical signal scales. A zero radius projects exactly to zero; positive
    radii retain the identity derivative at zero.
    """
    radius = jnp.maximum(jnp.asarray(radius, dtype=p1.dtype), 0.0)
    peak = jnp.maximum(jnp.maximum(jnp.abs(p1), jnp.abs(p2)), jnp.abs(p3))
    scale = jnp.maximum(peak, radius)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q1, q2, q3 = p1 / scale, p2 / scale, p3 / scale
    norm = _norm_with_zero_subgradient(q1 * q1 + q2 * q2 + q3 * q3)
    norm = jnp.where(norm == 0, jnp.ones_like(norm), norm)
    inside = norm <= radius / scale
    length = radius / norm
    return (
        jnp.where(inside, p1, q1 * length),
        jnp.where(inside, p2, q2 * length),
        jnp.where(inside, p3, q3 * length),
    )


def prox_huber_tv_conj(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    p3: jnp.ndarray,
    *,
    sigma: float | jnp.ndarray,
    lam: float | jnp.ndarray,
    delta: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prox for the conjugate of ``lam * huber_tv`` on vector-gradient duals."""
    sigma_arr = jnp.asarray(sigma, dtype=p1.dtype)
    lam_arr = jnp.asarray(lam, dtype=p1.dtype)
    delta_arr = jnp.asarray(delta, dtype=p1.dtype)
    scale = jnp.where(
        lam_arr > 0.0,
        lam_arr / jnp.maximum(lam_arr + sigma_arr * delta_arr, jnp.finfo(p1.dtype).tiny),
        0.0,
    )
    q1 = p1 * scale
    q2 = p2 * scale
    q3 = p3 * scale
    return project_tv_duals(q1, q2, q3, radius=lam_arr)


def tv_proximal(
    x: jnp.ndarray,
    lam_over_L: float | jax.Array,
    iterations: int = 20,
    *,
    nonnegative: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    huber_delta: float | jax.Array | None = None,
) -> jnp.ndarray:
    """Approximate the isotropic TV proximal by projected gradient on its dual.

    Solve ``min_u 0.5 ||u - x||^2 + lam TV(u)`` inside the optional box through
    the dual field ``p`` with ``|p| <= lam`` pointwise. The minimizing primal is
    ``u = project_box(x + div p)``; its gradient drives every dual update.
    Clipping only after an unconstrained solve is not the joint isotropic-TV
    prox: clipping can change the direction of a coupled gradient vector.
    Box projection is nonexpansive, so the step ``1 / ||grad3||^2`` remains
    valid. Use the exact finite-grid norm, not its asymptotic 3D bound of 12.
    Only three dual components persist between iterations. Large float32 CUDA
    volumes use stream-owned tiles with the same finite-iteration derivatives.
    ``huber_delta=None`` selects TV; a nonnegative physical threshold selects
    Huber TV (zero is its TV limit). Its conjugate quadratic is handled by a
    proximal dual shrink, so the gradient step does not depend on the threshold.
    """
    return _tv_execution(
        x,
        jnp.asarray(lam_over_L, dtype=x.dtype),
        None if huber_delta is None else jnp.asarray(huber_delta, dtype=x.dtype),
        int(iterations),
        nonnegative,
        lower_bound,
        upper_bound,
        gradient_scale_xyz,
    )


_MIN_CUDA_BYTES = 64 << 20


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6, 7))
def _tv_execution(
    x, lam, huber_delta, iterations, nonnegative, lower_bound, upper_bound, gradient_scale_xyz
):
    reference = functools.partial(
        _tv_proximal_jax,
        iterations=iterations,
        nonnegative=nonnegative,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gradient_scale_xyz=gradient_scale_xyz,
    )
    if iterations <= 0 or x.dtype != jnp.float32 or x.size * x.dtype.itemsize < _MIN_CUDA_BYTES:
        return reference(x, lam, huber_delta)
    import importlib.util
    import os

    if os.environ.get("TOMOJAX_CUDA_KERNELS") == "0" or importlib.util.find_spec("cupy") is None:
        return reference(x, lam, huber_delta)
    from ._cuda_tv import proximal

    lower = max(
        0.0 if nonnegative else -math.inf, lower_bound if lower_bound is not None else -math.inf
    )
    upper = upper_bound if upper_bound is not None else math.inf
    cuda = functools.partial(
        proximal,
        iterations=iterations,
        tau=1.0 / (gradient_norm_squared(x.shape, gradient_scale_xyz=gradient_scale_xyz) or 1.0),
        lower=lower,
        upper=upper,
        gradient_scale_xyz=gradient_scale_xyz,
    )
    return jax.lax.platform_dependent(x, lam, huber_delta, cuda=cuda, default=reference)


@_tv_execution.defjvp
def _tv_execution_jvp(
    iterations, nonnegative, lower_bound, upper_bound, gradient_scale_xyz, primals, tangents
):
    reference = functools.partial(
        _tv_proximal_jax,
        iterations=iterations,
        nonnegative=nonnegative,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gradient_scale_xyz=gradient_scale_xyz,
    )
    _, tangent = jax.jvp(reference, primals, tangents)
    return _tv_execution(
        *primals, iterations, nonnegative, lower_bound, upper_bound, gradient_scale_xyz
    ), tangent


def _tv_proximal_jax(
    x: jnp.ndarray,
    lam_over_L: float | jax.Array,
    huber_delta: float | jax.Array | None = None,
    iterations: int = 20,
    *,
    nonnegative: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    gradient_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> jnp.ndarray:
    """Reference finite-iteration TV algebra, also used for all derivatives."""
    lam = jnp.asarray(lam_over_L, dtype=x.dtype)
    tau = jnp.asarray(
        1.0 / (gradient_norm_squared(x.shape, gradient_scale_xyz=gradient_scale_xyz) or 1.0),
        dtype=x.dtype,
    )

    project = functools.partial(
        project_box,
        nonnegative=nonnegative,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    def prox_impl(lam_val: jnp.ndarray) -> jnp.ndarray:
        def body(
            p: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: object
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
            gx, gy, gz = grad3(
                project(x + div3(*p, gradient_scale_xyz=gradient_scale_xyz)),
                gradient_scale_xyz=gradient_scale_xyz,
            )
            q1, q2, q3 = p[0] + tau * gx, p[1] + tau * gy, p[2] + tau * gz
            if huber_delta is None:
                return project_tv_duals(q1, q2, q3, radius=lam_val), None
            return prox_huber_tv_conj(q1, q2, q3, sigma=tau, lam=lam_val, delta=huber_delta), None

        zeros = jnp.zeros_like(x)
        p, _ = jax.lax.scan(body, (zeros, zeros, zeros), None, length=int(iterations))
        return project(x + div3(*p, gradient_scale_xyz=gradient_scale_xyz))

    return jax.lax.cond(lam > 0, prox_impl, lambda _: project(x), lam)


def project_box(
    x: jnp.ndarray,
    *,
    nonnegative: bool,
    lower_bound: float | None,
    upper_bound: float | None,
) -> jnp.ndarray:
    """Project onto optional elementwise physical constraints."""
    if lower_bound is not None:
        x = jnp.maximum(x, jnp.asarray(lower_bound, dtype=x.dtype))
    if nonnegative:
        x = jnp.maximum(x, jnp.asarray(0.0, dtype=x.dtype))
    if upper_bound is not None:
        x = jnp.minimum(x, jnp.asarray(upper_bound, dtype=x.dtype))
    return x
