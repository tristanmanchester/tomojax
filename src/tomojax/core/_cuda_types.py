"""Types shared by the optional CUDA backend."""

from typing import Any, Protocol

# CuPy does not ship type stubs for its arrays, kernels, or CUDA handles.
type CuPyValue = Any


class CallbackBuffer(Protocol):
    # JAX exports this extension type via a factory, without a stub for its class.
    @property
    def shape(self) -> tuple[int, ...]: ...

    __cuda_array_interface__: dict[str, object]


class CallbackContext(Protocol):
    @property
    def stream(self) -> int: ...
