"""Phase corrections for complex transmission images."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.fft import dctn, idctn


@jax.jit
def unwrap_phase(projections: jax.Array) -> jax.Array:
    """Integrate wrapped phase differences independently on each detector image.

    Accepts complex transmission or wrapped phase in radians, with detector
    axes last: ``(..., rows, columns)``. The result is a real JAX array of the
    same shape, in radians, with zero mean on each image. Use
    :func:`remove_phase_ramp` to set its phase origin from air.

    Solves the least-squares fit to nearest-neighbour wrapped phase gradients
    with nonperiodic (Neumann) boundaries, using a discrete cosine transform.
    Smooth phase is recovered up to a constant when adjacent true phase
    differences are less than pi. With phase residues the gradients are
    inconsistent: this returns their least-squares integral, not an exact
    pointwise unwrapping. Missing or low-amplitude pixels should be cropped
    before this operation; they are not inferred from neighbouring views.
    """
    values = jnp.asarray(projections)
    phase = jnp.angle(values) if jnp.iscomplexobj(values) else values.astype(jnp.float32)
    du, dv = jnp.diff(phase, axis=-1), jnp.diff(phase, axis=-2)
    du, dv = jnp.arctan2(jnp.sin(du), jnp.cos(du)), jnp.arctan2(jnp.sin(dv), jnp.cos(dv))
    # D.T @ wrapped(D phase), including the one-sided boundary differences.
    padding = [(0, 0)] * phase.ndim
    left, right = padding.copy(), padding.copy()
    left[-1], right[-1] = (1, 0), (0, 1)
    rhs = jnp.pad(du, left) - jnp.pad(du, right)
    left[-1] = right[-1] = (0, 0)
    left[-2], right[-2] = (1, 0), (0, 1)
    rhs = rhs + jnp.pad(dv, left) - jnp.pad(dv, right)
    rows, columns = phase.shape[-2:]
    ev = 4 * jnp.sin(jnp.pi * jnp.arange(rows) / (2 * rows)) ** 2
    eu = 4 * jnp.sin(jnp.pi * jnp.arange(columns) / (2 * columns)) ** 2
    eigenvalues = ev[:, None] + eu[None, :]
    spectrum = dctn(rhs, axes=(-2, -1), norm="ortho")
    spectrum = spectrum / jnp.where(eigenvalues > 0, eigenvalues, 1)
    return idctn(spectrum.at[..., 0, 0].set(0), axes=(-2, -1), norm="ortho")


@jax.jit
def remove_phase_ramp(phase: jax.Array, *, air_mask: jax.Array) -> jax.Array:
    """Subtract each image's phase plane fitted only to its known air pixels.

    ``phase`` has detector axes last, ``(..., rows, columns)``, in radians.
    ``air_mask`` is a boolean mask or nonnegative weights broadcastable to its
    shape. It must cover enough noncollinear air pixels to determine a plane.
    A common mask can describe air strips at both sides of the sample; per-view
    masks can follow a drifting sample. Unreliable border pixels must not be
    included. No sample pixels should contribute: this correction cannot
    distinguish the sample's own low-frequency signal from a phase ramp.

    Returns a real JAX array of the same shape and units. For a reconstruction
    in pixel units, ``-phase * wavelength / (2*pi*pixel_size)`` then gives line
    integrals of the refractive-index decrement. Wavelength and pixel size
    must have the same units.
    """
    phase = jnp.asarray(phase)
    weights = jnp.broadcast_to(jnp.asarray(air_mask, dtype=phase.dtype), phase.shape)
    rows, columns = phase.shape[-2:]
    v, u = jnp.meshgrid(jnp.linspace(-1, 1, rows), jnp.linspace(-1, 1, columns), indexing="ij")
    basis = jnp.stack((jnp.ones_like(u), u, v), axis=-1)
    gram = jnp.einsum("...vu,vui,vuj->...ij", weights, basis, basis, precision="highest")
    rhs = jnp.einsum("...vu,vui->...i", weights * phase, basis, precision="highest")
    coefficients = jnp.linalg.solve(gram, rhs[..., None])[..., 0]
    return phase - jnp.einsum("...i,vui->...vu", coefficients, basis, precision="highest")
