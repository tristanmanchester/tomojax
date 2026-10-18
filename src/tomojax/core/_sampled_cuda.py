"""CUDA sampled-ray forward, transpose and fused weighted residual gradient."""

from functools import cache, partial
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import jax.numpy as jnp
import numpy as np

from .cuda import initialize_buffer, launch_kernel


@cache
def _source(constants):
    defines = "\n".join(f"#define {key} {value}" for key, value in constants)
    source = Path(__file__).with_name("_sampled_kernels.cu").read_text()
    # CUDA12 wheels carry the matching half/bfloat headers. Their version joins
    # the source cache key so a header upgrade cannot reuse an old binary.
    headers = files("nvidia.cuda_runtime") / "include"
    release = version("nvidia-cuda-runtime-cu12")
    return f"// CUDA runtime headers {release}\n{defines}\n{source}", str(headers)


def _constants(params, dtype, mode, cached):
    constants = {key.upper(): str(params[key]) for key in ("nx", "ny", "nz", "nu", "nv")}
    tv, tu = params["tile_v"], params["tile_u"]
    block = max(32 * params["num_warps"], 1 << (tu * tv - 1).bit_length())
    unroll = params.get("unroll")
    if unroll is None or unroll is False:
        unroll = 1
    elif unroll is True or unroll == 0:
        unroll = params["n_steps"]
    elif unroll < 0:
        raise ValueError("unroll must be non-negative")
    constants.update(
        TU=str(tu),
        TV=str(tv),
        BLOCK=str(block),
        MODE=str(mode),
        CACHED=str(int(cached)),
        NSTEPS=str(params["n_steps"]),
        UNROLL=str(unroll),
        COMPUTE_LOSS=str(int(params.get("compute_loss", True))),
        LAYOUT=str(params.get("layout_variant_id", 0)),
        ZINTEGER=str(int(params.get("kernel_variant_id", 0) == 1)),
    )
    for key, name in (
        ("step_size", "STEP"),
        ("du", "DU"),
        ("dv", "DV"),
        ("det_center_x", "CU"),
        ("det_center_z", "CV"),
        ("vol_origin_x", "OX"),
        ("vol_origin_y", "OY"),
        ("vol_origin_z", "OZ"),
        ("vx", "VX"),
        ("vy", "VY"),
        ("vz", "VZ"),
    ):
        constants[name] = f"{float(np.float32(params.get(key, 1))):.9e}f"
    for axis, key in zip("XYZ", ("nx", "ny", "nz"), strict=True):
        origin = params.get("vol_origin_" + axis.lower(), 0)
        voxel = params.get("v" + axis.lower(), 1)
        constants["LO" + axis] = f"{float(np.float32(origin - voxel)):.9e}f"
        constants["HI" + axis] = f"{float(np.float32(origin + params[key] * voxel)):.9e}f"
    constants["STORAGE"] = {
        jnp.dtype("float32"): "float",
        jnp.dtype("float16"): "half",
        jnp.dtype("bfloat16"): "__nv_bfloat16",
    }[dtype]
    return tuple(constants.items()), block


def _launch(
    context,
    outputs,
    poses,
    volume,
    targets,
    weights,
    initial,
    *state: Any,
    constants,
    block,
    tiles,
    mode,
):
    out, gradient = outputs if mode == 2 else (outputs, outputs)
    if mode in (1, 2):
        initialize_buffer(gradient, stream=context.stream, source=initial)
    source, headers = _source(constants)
    launch_kernel(
        source,
        "integrate",
        options=("--fmad=false", "-I" + headers),
        stream=context.stream,
        grid=(poses.shape[0] * tiles,),
        block=(block,),
        arguments=(
            poses,
            volume,
            targets,
            weights,
            out,
            gradient,
            np.int32(poses.shape[0]),
            *state,
        ),
    )


def run(poses, volume, target=None, weights=None, initial=None, *, mode, params, state=()):
    """Launch inline or precomputed rays; adjoints accumulate into ``initial``."""
    if params.get("interpret", False):
        from ._sampled_reference import run as reference

        return reference(
            poses, volume, target, weights, initial, mode=mode, params=params, state=state
        )
    n, nu, nv = poses.shape[0], params["nu"], params["nv"]
    pu, pv = -(-nu // params["tile_u"]), -(-nv // params["tile_v"])
    constants, block = _constants(params, volume.dtype, mode, bool(state))
    volume_size = params["nx"] * params["ny"] * params["nz"]
    shape = (n, nv, nu) if mode == 0 else (volume_size,) if mode == 1 else (n, pv, pu)
    output = jax.ShapeDtypeStruct(shape, jnp.float32)
    if mode == 2:
        output = (output, jax.ShapeDtypeStruct((volume_size,), jnp.float32))
    aliases = {4: int(mode == 2)} if mode in (1, 2) else {}
    call = buffer_callback(
        partial(_launch, constants=constants, block=block, tiles=pu * pv, mode=mode),
        output,
        input_output_aliases=aliases,
        vmap_method="sequential",
        command_buffer_compatible=True,
    )
    return call(
        poses,
        volume,
        poses if target is None else target,
        poses if weights is None else weights,
        poses if initial is None else initial,
        *state,
    )
