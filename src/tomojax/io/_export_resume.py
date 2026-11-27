"""Durable plane counts for the three export layouts with restartable payloads."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import fcntl
import json
import os
from pathlib import Path
import shutil
from time import monotonic

import h5py
import numpy as np

from tomojax.core.files import replaced_when_complete
from tomojax.core.host_volume import mapped_storage
from tomojax.core.input_files import check_output, publishing
from tomojax.errors import Canceled, InvalidParameter

INTERVAL = 2.0
_MARKER = "tomojax-export-1"


def mapped_identity(pixels: object) -> tuple[int, ...] | None:
    """Identify the mapped bytes, including a sliced view's actual offset and strides."""
    mapping = mapped_storage(pixels)
    if (
        not isinstance(pixels, np.ndarray)
        or mapping is None
        or mapping.filename is None
        or mapping.mode == "c"
    ):
        return None
    try:
        stat = Path(mapping.filename).stat()
        with Path("/proc/self/maps").open() as regions:
            for line in regions:
                bounds, _, offset, device, inode, *_ = line.split(maxsplit=5)
                low, high = (int(value, 16) for value in bounds.split("-"))
                if low <= pixels.ctypes.data < high:
                    major, minor = (int(value, 16) for value in device.split(":"))
                    if os.makedev(major, minor) == stat.st_dev and int(inode) == stat.st_ino:
                        return (
                            stat.st_dev,
                            stat.st_ino,
                            stat.st_size,
                            stat.st_mtime_ns,
                            int(offset, 16) + pixels.ctypes.data - low,
                            *pixels.strides,
                        )
                    break
    except OSError:
        pass
    return None


def hdf5_identity(data: h5py.Dataset) -> tuple[int, ...] | None:
    """Identify one HDF5 signal. Several backing files need a fresh export."""
    file = data.file
    if file.driver != "sec2" or data.is_virtual or data.id.get_create_plist().get_external_count():
        return None
    stat = os.fstat(file.id.get_vfd_handle())
    offset = data.id.get_offset()
    if offset is None:
        offset = h5py.h5o.get_info(data.id).addr
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, offset


def partial_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial")


def sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass
class State:
    marker: str
    request: dict
    values: dict
    planes: int = 0
    pages: int = 0
    below: int = 0
    above: int = 0
    finished: bool = False


@contextmanager
def locked(path: Path, *, create: bool) -> Iterator[Path | None]:
    """Lock the partial directory inode. Reject another exporter before reading its journal."""
    root = partial_path(path)
    check_output(root)
    if create:
        with publishing(root):
            root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        yield None
        return
    fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FileExistsError(f"another export is writing {path}") from None
        if not root.exists() or not os.path.samestat(os.fstat(fd), root.stat()):
            raise InvalidParameter("path", f"another export just finished at {path}; retry")
        yield root
    finally:
        os.close(fd)


def saved(root: Path) -> State | None:
    journal = root / "journal.json"
    if not journal.exists():
        if any(not child.name.startswith(".journal.json.partial-") for child in root.iterdir()):
            raise InvalidParameter("path", f"{root} contains files without an export journal")
        return None
    try:
        state = State(**json.loads(journal.read_bytes()))
        if state.marker == _MARKER:
            return state
    except (TypeError, ValueError):
        pass
    raise InvalidParameter("path", f"{journal} is not a recognized export journal")


def matching(state: State | None, request: dict) -> bool:
    return state is not None and request["source"] is not None and state.request == request


def prior_state(path: Path, request: dict) -> State | None:
    with locked(path, create=False) as root:
        state = saved(root) if root else None
        return state if matching(state, request) else None


@dataclass
class Journal:
    root: Path
    state: State
    dirty: set[Path] = field(default_factory=set)
    last: float = 0.0

    def save(self, *, force: bool = False) -> None:
        if not force and monotonic() - self.last < INTERVAL:
            return
        for path in self.dirty:
            sync(path)
        for parent in {p.parent for p in self.dirty}:
            sync(parent)
        with replaced_when_complete(self.root / "journal.json", "export journal") as temporary:
            temporary.write_text(json.dumps(asdict(self.state), allow_nan=False), encoding="utf-8")
        sync(self.root)
        self.dirty.clear()
        self.last = monotonic()

    def advanced(self, planes: int, below: int, above: int) -> None:
        self.state.planes, self.state.below, self.state.above = planes, below, above
        self.save()

    def page(self, count: int) -> None:
        self.state.pages = count
        self.save()


@contextmanager
def continuing(path: Path, request: dict, values: dict) -> Iterator[Journal]:
    with locked(path, create=True) as root:
        assert root is not None
        state = saved(root)
        if not matching(state, request):
            with publishing(root):
                if state is not None:
                    for child in root.iterdir():
                        if child.name != "journal.json":
                            if child.is_dir() and not child.is_symlink():
                                shutil.rmtree(child)
                            else:
                                child.unlink()
                state = State(_MARKER, request, values)
                journal = Journal(root, state)
                journal.save(force=True)
        else:
            assert state is not None
            journal = Journal(root, state)
        try:
            yield journal
        except (Canceled, KeyboardInterrupt) as exc:
            journal.save(force=True)
            action = "continue" if request["source"] is not None else "restart"
            message = f"export stopped; rerun the same export to {action} at {root}"
            if isinstance(exc, Canceled):
                exc.args = (message,)
                raise
            raise Canceled(message) from exc
        except BaseException:
            with publishing(root):
                shutil.rmtree(root)
            raise
        else:
            with publishing(root):
                shutil.rmtree(root)
