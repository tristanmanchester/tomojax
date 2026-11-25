"""Types for the optional CUDA backend's untyped values."""

from typing import Any, Protocol

# CuPy does not ship type stubs for its arrays, kernels, or CUDA handles.
type CuPyValue = Any


class CallbackBuffer(Protocol):
    """A device buffer JAX passes to a CUDA callback; JAX's class has no stub."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Buffer shape."""
        ...

    __cuda_array_interface__: dict[str, object]


class CallbackContext(Protocol):
    """The execution context JAX passes to a CUDA callback."""

    @property
    def stream(self) -> int:
        """CUDA stream handle the callback runs on."""
        ...
