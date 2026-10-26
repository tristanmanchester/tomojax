"""Shared gather precision and volume layout for JAX and CUDA projectors."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def resolve_gather_target(gather_dtype: str) -> jnp.dtype:
    """Resolve the gather/interpolation dtype for the forward projector."""
    if not isinstance(gather_dtype, str):
        raise ValueError(f"gather_dtype must be a string; got {type(gather_dtype).__name__}")
    gd = gather_dtype.lower()
    if gd == "auto":
        platform = jax.default_backend()
        if platform in ("gpu", "tpu"):
            return jnp.bfloat16 if platform == "tpu" else jnp.float16
        return jnp.float32
    if gd in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if gd in ("fp16", "float16", "half"):
        return jnp.float16
    if gd in ("fp32", "float32", "single"):
        return jnp.float32
    raise ValueError(
        "gather_dtype must be one of 'auto', 'fp32', 'float32', 'single', "
        "'bf16', 'bfloat16', 'fp16', 'float16', or 'half'; "
        f"got {gather_dtype!r}"
    )


def prepare_volume_for_gather(volume: jnp.ndarray, gather_dtype: str) -> jnp.ndarray:
    """Cast gather samples and flatten the volume in (z, y, x) storage order."""
    target = resolve_gather_target(gather_dtype)
    vol_cast = volume if volume.dtype == target else volume.astype(target)
    return jnp.ravel(vol_cast, order="C")
