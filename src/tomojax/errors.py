"""Parameter paths and exception classification shared by numerical and client interfaces."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from typing import TYPE_CHECKING

from tomojax.wire.progress import Error

if TYPE_CHECKING:
    from collections.abc import Iterator


class InvalidParameter(ValueError):
    """A domain or cross-field constraint failed at a named parameter.

    Parameters
    ----------
    path : str
        Field name relative to the current constructor, or a nested operation path.
    message : str
        Human-readable reason the supplied value is invalid.
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path


class Cancelled(Exception):
    """The operation stopped at a cooperative checkpoint."""


@contextmanager
def within(path: str) -> Iterator[None]:
    """Attach the enclosing field or list element once at its construction boundary."""
    try:
        yield
    except InvalidParameter as exc:
        raise InvalidParameter(f"{path}.{exc.path}", str(exc)) from exc


def _causes(exc: BaseException) -> Iterator[BaseException]:
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        yield cause
        cause = cause.__cause__


def _device_allocation_failed(error: BaseException) -> bool:
    # An exception implies its defining module is already loaded. Checking one
    # must not import JAX or CuPy, nor initialise CUDA.
    cupy_oom = getattr(sys.modules.get("cupy.cuda.memory"), "OutOfMemoryError", None)
    if cupy_oom is not None and isinstance(error, cupy_oom):
        return True
    jax_errors = sys.modules.get("jax.errors")
    if jax_errors is None or not isinstance(error, jax_errors.JaxRuntimeError):
        return False
    if getattr(error, "error_code", None) == 8:  # absl::RESOURCE_EXHAUSTED
        return True
    message = str(error).lower()
    return (
        "resource_exhausted" in message
        or "out of memory" in message
        or ("fft_plan != nullptr" in message and "cufft batched plan with scratch" in message)
    )


def device_out_of_memory(exc: BaseException) -> bool:
    """Whether a device allocation failed anywhere along the cause chain of ``exc``.

    XLA reports allocator exhaustion as ``RESOURCE_EXHAUSTED``, but some device
    allocations (kernel images, cuFFT plan scratch) fail as ``INTERNAL`` errors that
    only the message identifies. CuPy raises its own allocator error. A host
    ``MemoryError`` is not a device failure: smaller device batches do not help it.

    Parameters
    ----------
    exc : BaseException
        Exception to inspect, with its ``__cause__`` chain.

    Returns
    -------
    bool
        True if any exception in the chain is a failed device allocation.
    """
    return any(_device_allocation_failed(cause) for cause in _causes(exc))


def classify(exc: Exception) -> Error:
    """Classify a failure by exception type and status along its cause chain."""
    kind = "internal"
    for cause in _causes(exc):
        if isinstance(cause, InvalidParameter):
            return Error(2, "invalid_parameter", str(exc), cause.path)
        if isinstance(cause, Cancelled):
            return Error(2, "cancelled", str(exc))
        if isinstance(cause, MemoryError) or _device_allocation_failed(cause):
            return Error(2, "out_of_memory", str(exc))
        if isinstance(cause, OSError | ValueError | KeyError):
            kind = "input"
    return Error(2, kind, str(exc))
