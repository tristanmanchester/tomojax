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


class HardwareUnavailable(InvalidParameter):
    """The selected backend or device is unavailable on this executor."""


class Canceled(Exception):
    """The operation stopped at a work boundary.

    Parameters
    ----------
    message : str, optional
        Why and where it stopped; ``"canceled"`` by default.
    checkpoint : str or None, optional
        The directory holding the stopped run's saved state: the same call with
        this checkpoint resumes. None when the run had no checkpoint.
    """

    def __init__(self, message: str = "canceled", *, checkpoint: str | None = None) -> None:
        super().__init__(message)
        self.checkpoint = checkpoint


@contextmanager
def within(path: str) -> Iterator[None]:
    """Attach the enclosing field or list element once at its construction boundary."""
    try:
        yield
    except InvalidParameter as exc:
        raise type(exc)(f"{path}.{exc.path}", str(exc)) from exc


def _causes(exc: BaseException) -> Iterator[BaseException]:
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        yield cause
        cause = cause.__cause__


def _is_jax_runtime_error(error: BaseException) -> bool:
    runtime_error = getattr(sys.modules.get("jax.errors"), "JaxRuntimeError", None)
    return runtime_error is not None and isinstance(error, runtime_error)


def _device_allocation_failed(error: BaseException) -> bool:
    # An exception implies its defining module is already loaded. Checking one
    # must not import JAX or CuPy, nor initialize CUDA.
    cupy_oom = getattr(sys.modules.get("cupy.cuda.memory"), "OutOfMemoryError", None)
    if cupy_oom is not None and isinstance(error, cupy_oom):
        return True
    if not _is_jax_runtime_error(error):
        return False
    if getattr(error, "error_code", None) == 8:  # absl::RESOURCE_EXHAUSTED
        return True
    message = str(error).lower()
    return "resource_exhausted" in message or "out of memory" in message


@contextmanager
def explain_cuda_failure() -> Iterator[None]:
    """Translate fatal CUDA errors at the reconstruction boundary, preserving their cause."""
    try:
        yield
    except RuntimeError as exc:
        message = str(exc).lower()
        is_jax = _is_jax_runtime_error(exc)
        is_cupy = type(exc).__module__.startswith("cupy_backends.cuda.")
        if not (is_jax or is_cupy):
            raise
        lost = "illegal_address" in message or "an illegal memory access" in message
        runtime = sys.modules.get("cupy_backends.cuda.api.runtime")
        memory_info = getattr(runtime, "memGetInfo", None)
        if is_jax and "failed to create cufft batched plan" in message and memory_info is not None:
            # XLA can surface an earlier asynchronous fault at plan creation.
            # CUDA reports a sticky illegal address on subsequent runtime calls.
            try:
                memory_info()
            except RuntimeError as probe:
                lost = lost or getattr(probe, "status", None) == 700  # cudaErrorIllegalAddress
        if lost:
            raise RuntimeError("The CUDA context is lost. Restart the process.") from exc
        raise


def device_out_of_memory(exc: BaseException) -> bool:
    """Whether a device allocation failed anywhere along the cause chain of ``exc``.

    XLA reports allocator exhaustion as ``RESOURCE_EXHAUSTED``, but some device
    allocations (such as kernel images) fail as ``INTERNAL`` errors that
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
        if isinstance(cause, HardwareUnavailable):
            return Error(2, "unavailable", str(exc), cause.path)
        if isinstance(cause, InvalidParameter):
            return Error(2, "invalid_parameter", str(exc), cause.path)
        if isinstance(cause, Canceled):
            return Error(2, "canceled", str(exc), checkpoint=cause.checkpoint)
        if isinstance(cause, MemoryError) or _device_allocation_failed(cause):
            return Error(2, "out_of_memory", str(exc))
        if isinstance(cause, OSError | ValueError | KeyError):
            kind = "input"
    return Error(2, kind, str(exc))
