"""File input tracking and atomic output replacement."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


# A worker executes one Python job at a time. Its code can load data on any thread.
_input_recorders: dict[object, set[Path]] = {}


@contextmanager
def record_inputs(paths: set[Path]) -> Iterator[None]:
    """Collect loader inputs across all threads for the duration of a worker call."""
    token = object()
    _input_recorders[token] = paths
    try:
        yield
    finally:
        del _input_recorders[token]


def read_path[T: (str, Path)](path: T) -> T:
    """Record the resolved input at the point of use, before the caller can change directory."""
    for paths in tuple(_input_recorders.values()):
        paths.add(Path(path).resolve())
    return path


@contextmanager
def replaced_when_complete(path: str | os.PathLike[str], parts: str) -> Iterator[Path]:
    """A temporary file beside ``path`` that replaces it only once the write is complete.

    The temporary file is in the same directory, so on the same filesystem; on
    success it reaches the disk and moves onto ``path`` in one step, so a file
    already at ``path`` survives until the new one is whole. On any failure the
    temporary file is removed and nothing at ``path`` changes. A full disk raises
    ``OSError`` saying how much was written, the room left and ``parts``, what was
    being written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.name}.partial-", dir=target.parent)
    os.close(handle)
    mask = os.umask(0)
    os.umask(mask)
    temporary = Path(name)
    temporary.chmod(0o666 & ~mask)  # the mode an ordinary new file gets, not mkstemp's 0600
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)  # one step: the old file or the new, never neither
    except BaseException as err:
        written = temporary.stat().st_size if temporary.exists() else 0
        temporary.unlink(missing_ok=True)
        if _disk_full(err):
            free = shutil.disk_usage(target.parent).free
            raise OSError(
                errno.ENOSPC,
                f"saving {target} stopped after {written / 2**30:.1f} GiB written: "
                f"{target.parent} has {free / 2**30:.1f} GiB free. It was writing {parts}; "
                "free some space or save elsewhere",
            ) from err
        raise


def _disk_full(err: BaseException) -> bool:
    """Whether ``err``, or what caused it, is a full disk (HDF5 reports errno 28 in text)."""
    current: BaseException | None = err
    while current is not None:
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return True
        text = str(current)
        if "No space left on device" in text or "errno = 28" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = ["replaced_when_complete"]
