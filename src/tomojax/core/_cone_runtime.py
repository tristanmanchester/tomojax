"""CUDA availability and shared kernel compilation for cone operators."""

from functools import cache
from importlib.resources import files
import os
from typing import Any

# Threads per block of the plane-tile transpose kernels (PLANE_THREADS in the source).
PLANE_THREADS = 256
# One warp scans the list of detector runs, with one lane per view.
_PLANE_VIEWS = 32
_PLANE_KERNELS = ("plane_adjoint", "plane_adjoint_wide", "pair_plane_adjoint")


def plane_kernel(name: str, views: int) -> str:
    """Select the tile that can keep all its detector footprints in one run list."""
    return f"{name}<{'true' if views <= _PLANE_VIEWS else 'false'}>"


def use_cuda_cone() -> bool:
    """Return whether cone projections run as CUDA kernels (``TOMOJAX_CUDA_KERNELS=0`` disables)."""
    if os.environ.get("TOMOJAX_CUDA_KERNELS", "auto") == "0":
        return False
    from tomojax.core._cuda_joseph import cuda_gather_available

    return cuda_gather_available()


@cache
def cuda_module(*, z_fast: bool = False) -> Any:
    """Shared cone CUDA kernels, compiled lazily on the active device."""
    import cupy as cp

    source = files("tomojax.core").joinpath("cone_kernels.cu").read_text(encoding="utf-8")
    options = (
        f"-DVOLUME_Z_FAST={int(z_fast)}",
        f"-DPLANE_THREADS={PLANE_THREADS}",
        f"-DVMAX={_PLANE_VIEWS}",
    )
    expressions = tuple(
        f"{name}<{small}>" for name in _PLANE_KERNELS for small in ("true", "false")
    )
    return cp.RawModule(code=source, options=options, name_expressions=expressions)
