"""Independent rigid poses for projector derivative checks."""

import jax.numpy as jnp


def pose_update(nominal, parameters, *, translation_frame="detector"):
    """Apply Ry(beta) Rx(alpha) Rz(phi), with physical x/z/y translations."""
    alpha, beta, phi, dx, dz, dy = parameters
    sx, sy, sz = jnp.sin(jnp.stack([alpha, beta, phi]))
    cx, cy, cz = jnp.cos(jnp.stack([alpha, beta, phi]))
    rotation = jnp.array(
        [
            [cy * cz + sy * sx * sz, -cy * sz + sy * sx * cz, sy * cx],
            [cx * sz, cx * cz, -sx],
            [-sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, cy * cx],
        ]
    )
    shift = jnp.stack([dx, dy, dz])
    if translation_frame == "object":
        shift = nominal[:3, :3] @ shift
    return nominal.at[:3, :3].set(nominal[:3, :3] @ rotation).at[:3, 3].add(shift)
