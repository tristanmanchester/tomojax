"""Choose the matched projector or backprojection backend for the actual placement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tomojax._parameters import check_backend
from tomojax.core.backend_policy import cuda_backend
from tomojax.core.cone import use_cuda_cone
from tomojax.errors import InvalidParameter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tomojax._typed_arrays import Device


def reconstruction_backend(
    backend: str,
    *,
    cone: bool,
    context: str,
    devices: Iterable[Device] | None = None,
) -> Literal["cuda", "jax"]:
    """Resolve an explicit or automatic backend without allocating a projection or volume."""
    check_backend(backend, context)
    if backend == "jax":
        return "jax"
    if cone:
        available = use_cuda_cone()
    elif devices is None:
        available = cuda_backend()
    else:
        available = all(cuda_backend(d.platform) for d in devices)
    if available:
        return "cuda"
    if backend == "cuda":
        if cone:
            detail = "the CUDA cone kernels need CuPy on a CUDA device"
        else:
            detail = "backend='cuda' needs a CUDA device"
        message = {
            "fbp": "CUDA FBP requires CUDA arrays",
            "fbp_host": "fbp_host: CUDA requires CUDA",
            "fdk": "fdk: the CUDA kernel needs CuPy on a CUDA device",
        }.get(context, f"{context}: {detail}")
        raise InvalidParameter("config.backend", message)
    return "jax"
