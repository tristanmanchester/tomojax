from __future__ import annotations

import jax.numpy as jnp


def grad3(u: jnp.ndarray):
    dx = jnp.concatenate(
        [jnp.diff(u, axis=0), jnp.zeros((1, u.shape[1], u.shape[2]), u.dtype)],
        axis=0,
    )
    dy = jnp.concatenate(
        [jnp.diff(u, axis=1), jnp.zeros((u.shape[0], 1, u.shape[2]), u.dtype)],
        axis=1,
    )
    dz = jnp.concatenate(
        [jnp.diff(u, axis=2), jnp.zeros((u.shape[0], u.shape[1], 1), u.dtype)],
        axis=2,
    )
    return dx, dy, dz


def div3(px: jnp.ndarray, py: jnp.ndarray, pz: jnp.ndarray):
    """Discrete divergence matching the TV updates: ``div3 = -grad3*``."""
    if px.shape[0] == 1:
        dx = jnp.zeros_like(px)
    else:
        first_x = px[0:1, :, :]
        if px.shape[0] > 2:
            mid_x = px[1:-1, :, :] - px[0:-2, :, :]
            dx = jnp.concatenate([first_x, mid_x, -px[-2:-1, :, :]], axis=0)
        else:
            dx = jnp.concatenate([first_x, -px[-2:-1, :, :]], axis=0)

    if py.shape[1] == 1:
        dy = jnp.zeros_like(py)
    else:
        first_y = py[:, 0:1, :]
        if py.shape[1] > 2:
            mid_y = py[:, 1:-1, :] - py[:, 0:-2, :]
            dy = jnp.concatenate([first_y, mid_y, -py[:, -2:-1, :]], axis=1)
        else:
            dy = jnp.concatenate([first_y, -py[:, -2:-1, :]], axis=1)

    if pz.shape[2] == 1:
        dz = jnp.zeros_like(pz)
    else:
        first_z = pz[:, :, 0:1]
        if pz.shape[2] > 2:
            mid_z = pz[:, :, 1:-1] - pz[:, :, 0:-2]
            dz = jnp.concatenate([first_z, mid_z, -pz[:, :, -2:-1]], axis=2)
        else:
            dz = jnp.concatenate([first_z, -pz[:, :, -2:-1]], axis=2)
    return dx + dy + dz
