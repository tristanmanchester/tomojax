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


def classify(exc: Exception) -> Error:
    """Classify a failure by exception type and structured status along its cause chain."""
    import jax

    # A CuPy exception implies its defining module is already loaded. Classifying
    # an unrelated exception must not initialise CUDA or require optional CuPy.
    memory_module = sys.modules.get("cupy.cuda.memory")
    cupy_oom = getattr(memory_module, "OutOfMemoryError", MemoryError)
    cause: BaseException | None = exc
    seen: set[int] = set()
    kind = "internal"
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, InvalidParameter):
            return Error(2, "invalid_parameter", str(exc), cause.path)
        if isinstance(cause, Cancelled):
            return Error(2, "cancelled", str(exc))
        if isinstance(cause, MemoryError | cupy_oom) or (
            isinstance(cause, jax.errors.JaxRuntimeError)
            and getattr(cause, "error_code", None) == 8  # absl::RESOURCE_EXHAUSTED
        ):
            return Error(2, "out_of_memory", str(exc))
        if isinstance(cause, OSError | ValueError | KeyError):
            kind = "input"
        cause = cause.__cause__
    return Error(2, "input" if kind == "input" else "internal", str(exc))
