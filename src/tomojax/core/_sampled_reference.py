"""Portable reference for sampled-ray CUDA launches and cached traversal state."""

import jax
import jax.numpy as jnp

from .geometry.base import Detector, Grid
from .projector import _projector_traversal_state, _trilinear_gather, _trilinear_scatter_add


def _traversal(poses, params):
    shape = tuple(params[k] for k in ("nx", "ny", "nz"))
    voxel = tuple(params[k] for k in ("vx", "vy", "vz"))
    origin = tuple(params["vol_origin_" + a] for a in "xyz")
    grid = Grid(*shape, *voxel, vol_origin=origin)
    detector = Detector(
        params["nu"],
        params["nv"],
        params["du"],
        params["dv"],
        (params["det_center_x"], params["det_center_z"]),
    )
    return jax.vmap(
        lambda t: _projector_traversal_state(
            t, grid, detector, step_size=params["step_size"], n_steps=params["n_steps"]
        )[:7]
    )(poses)


def _tile_sum(values, params):
    n, nv, nu = values.shape
    tv, tu = params["tile_v"], params["tile_u"]
    pv, pu = -(-nv // tv), -(-nu // tu)
    padded = jnp.pad(values, ((0, 0), (0, pv * tv - nv), (0, pu * tu - nu)))
    return padded.reshape(n, pv, tv, pu, tu).sum(axis=(2, 4))


def run(poses, volume, target, weights, initial, *, params, mode, state=()):
    """Execute the same recurrence without device-specific primitives."""
    nx, ny, nz = (params[k] for k in ("nx", "ny", "nz"))
    n, nv, nu = poses.shape[0], params["nv"], params["nu"]
    if state:
        ix, iy, iz, count, dx, dy, dz = state
        ix, iy, iz, count = (x.reshape(n, -1) for x in (ix, iy, iz, count))
        dx, dy, dz = (x.reshape(n, 1) for x in (dx, dy, dz))
    else:
        ix, iy, iz, dx, dy, dz, count = _traversal(poses, params)
    starts = tuple(x.ravel() for x in (ix, iy, iz))
    deltas = tuple(jnp.broadcast_to(x, ix.shape).ravel() for x in (dx, dy, dz))
    count = count.ravel()
    step = jnp.float32(params["step_size"])
    unroll = params.get("unroll")
    unroll = 1 if unroll is None else unroll

    def forward(k, carry):
        acc, x, y, z = carry
        sample_z = jnp.floor(z + 0.5) if params.get("kernel_variant_id", 0) == 1 else z
        value = _trilinear_gather(volume.astype(jnp.float32).ravel(), x, y, sample_z, nx, ny, nz)
        return (
            acc + jnp.where(k < count, value * step, 0),
            x + deltas[0],
            y + deltas[1],
            z + deltas[2],
        )

    if mode != 1:
        projection = jax.lax.fori_loop(
            0, params["n_steps"], forward, (jnp.zeros_like(starts[0]), *starts), unroll=unroll
        )[0].reshape(n, nv, nu)
        if mode == 0:
            return projection
        residual = projection - target
        if mode == 3:
            return _tile_sum(residual * residual, params)
        weighted = residual * weights.reshape(n, 1, 1)
        losses = _tile_sum(0.5 * weighted * weighted, params)
        if not params.get("compute_loss", True):
            losses = jnp.zeros_like(losses)
        target = weighted * weights.reshape(n, 1, 1)

    def adjoint(k, carry):
        grad, x, y, z = carry
        values = jnp.where(k < count, target.ravel() * step, 0)
        sample_z = jnp.floor(z + 0.5) if params.get("kernel_variant_id", 0) == 1 else z
        grad = _trilinear_scatter_add(grad, values, x, y, sample_z, nx, ny, nz)
        return grad, x + deltas[0], y + deltas[1], z + deltas[2]

    gradient = jax.lax.fori_loop(
        0, params["n_steps"], adjoint, (initial.ravel(), *starts), unroll=unroll
    )[0]
    return (losses, gradient) if mode == 2 else gradient
