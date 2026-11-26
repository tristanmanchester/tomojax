"""Saved state that lets a long run stop and resume where it stopped.

A run's checkpoint is a directory. ``checkpoint.json`` names the run (digests of
its operation, settings and input) and lists the saved entries; each entry is a
``state-NNNNNN`` directory of ``.npy`` arrays and a record of scalars in the
manifest. A save writes a new entry directory, replaces ``checkpoint.json`` in
one step and only then deletes the entry it superseded, so a kill at any moment
leaves the previous state usable.

A checkpointed loop saves at a boundary (an iteration, a slab, an alignment step)
once :data:`INTERVAL` seconds, and :data:`SAVE_SHARE` times the last save's
duration, have passed since the last save. After SIGTERM it saves at its next
boundary and raises :class:`~tomojax.errors.Canceled`. SIGTERM outside a
checkpointed loop ends the process as it would without a checkpoint. A finished
slab is saved when it finishes. The owner of a checkpoint removes it when the run
completes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import signal
import threading
from time import monotonic
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypedDict, runtime_checkable
import zlib

import numpy as np

from tomojax.core.files import replaced_when_complete
from tomojax.core.host_volume import mapped_storage
from tomojax.errors import Canceled, InvalidParameter

if TYPE_CHECKING:
    from types import FrameType

    from tomojax.core.json_values import JsonValue

INTERVAL = 600.0
"""Least seconds between two periodic saves."""
SAVE_SHARE = 20.0
"""Least multiple of the last save's duration between two periodic saves (5% overhead)."""

_FORMAT = 1
_MANIFEST = "checkpoint.json"
_LOCK = "checkpoint.lock"
_ENTRY = "state-"
_MAPPED = 2**20  # arrays larger than this many bytes are memory-mapped when loaded
_SAMPLE = 2**20  # elements of each sampled array slice in a digest


class _Entry(TypedDict):
    directory: str
    arrays: int
    record: dict[str, JsonValue]


class _Manifest(TypedDict):
    format: int
    identity: dict[str, str] | None
    entries: dict[str, _Entry]
    next: int


@dataclass(frozen=True)
class Saved:
    """A saved entry: its arrays (read-only, large ones memory-mapped) and its scalars."""

    arrays: tuple[np.ndarray, ...]
    record: dict[str, JsonValue]

    def scalars[T](self, kind: Callable[..., T]) -> T:
        """The saved scalars as ``kind``, the named tuple they were saved from."""
        return kind(**self.record)


def _host(value: object) -> np.ndarray:
    """A host array for a NumPy, JAX or CuPy array or scalar."""
    if isinstance(value, np.ndarray):
        return value
    get = getattr(value, "get", None) if type(value).__module__.startswith("cupy") else None
    array: np.ndarray = np.asarray(value if get is None else get())
    return array


def _scalar(value: object) -> object:
    """A NumPy or JAX scalar in a record as the Python number it holds."""
    item = getattr(value, "item", None)
    if item is None:
        raise TypeError(f"a checkpoint record cannot hold {type(value).__name__}")
    return item()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Session:
    """One checkpoint directory, locked by this process while the run is open."""

    def __init__(self, path: Path) -> None:
        manifest = path / _MANIFEST
        if path.is_dir() and not manifest.exists() and any(p.name != _LOCK for p in path.iterdir()):
            raise InvalidParameter(
                "checkpoint",
                f"{path} holds files but no TomoJAX checkpoint: choose a new or empty directory",
            )
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = os.open(path / _LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock)
            raise InvalidParameter("checkpoint", f"{path} is in use by another process") from None
        self.manifest: _Manifest = (
            json.loads(manifest.read_text())
            if manifest.exists()
            else {"format": _FORMAT, "identity": None, "entries": {}, "next": 0}
        )
        if self.manifest.get("format") != _FORMAT:
            os.close(self._lock)
            raise InvalidParameter("checkpoint", f"{path} holds a checkpoint of another format")
        referenced = {entry["directory"] for entry in self.manifest["entries"].values()}
        for entry in path.glob(_ENTRY + "*"):
            if entry.name not in referenced:  # written by a save a kill interrupted
                shutil.rmtree(entry, ignore_errors=True)
        self.guard = threading.RLock()
        self.kept = False
        self.loops = 0
        self.stop: int | None = None
        self.saved_at, self.save_seconds = monotonic(), 0.0
        self.counters: dict[str, int] = {}

    def bind(self, identity: Mapping[str, str]) -> None:
        """Adopt ``identity`` for a new checkpoint, or check that it is the saved run's."""
        with self.guard:
            known = self.manifest["identity"]
            if known is None:
                self.manifest["identity"] = dict(identity)
                self._write_manifest()
            elif known != dict(identity):
                differ = [
                    k
                    for k in sorted(known.keys() | identity.keys())
                    if known.get(k) != identity.get(k)
                ]
                raise InvalidParameter(
                    "checkpoint",
                    f"{self.path} holds the checkpoint of another run (its {', '.join(differ)} "
                    "differ): remove it, or choose another directory",
                )

    def claim(self, kind: str) -> str:
        """The next key of this kind: the same sequence in every run of the same work."""
        with self.guard:
            count = self.counters.get(kind, 0)
            self.counters[kind] = count + 1
        return f"{kind}-{count}"

    def keys(self, prefix: str) -> list[str]:
        """Saved keys starting with ``prefix``."""
        return [key for key in self.manifest["entries"] if key.startswith(prefix)]

    def load(self, key: str) -> Saved | None:
        """The entry saved under ``key``, if any."""
        entry = self.manifest["entries"].get(key)
        if entry is None:
            return None
        directory = self.path / entry["directory"]
        arrays: list[np.ndarray] = []
        for index in range(entry["arrays"]):
            file = directory / f"{index}.npy"
            mapped = file.stat().st_size > _MAPPED
            arrays.append(np.load(file, mmap_mode="r" if mapped else None, allow_pickle=False))
        return Saved(tuple(arrays), entry["record"])

    def save(self, key: str, arrays: Sequence[object], record: Mapping[str, JsonValue]) -> None:
        """Replace the entry ``key``: new files first, then the manifest, then the old files."""
        start = monotonic()
        with self.guard:
            name = f"{_ENTRY}{self.manifest['next']:06d}"
            self.manifest["next"] += 1
            directory = self.path / name
            directory.mkdir()
            previous = self.manifest["entries"].get(key)
            try:
                for index, value in enumerate(arrays):
                    host = _host(value)
                    file = directory / f"{index}.npy"
                    with (
                        replaced_when_complete(file, "a checkpoint") as part,
                        part.open("wb") as stream,
                    ):
                        np.lib.format.write_array(stream, host, allow_pickle=False)
                    del host
                _fsync_directory(directory)
                entry: _Entry = {"directory": name, "arrays": len(arrays), "record": dict(record)}
                self.manifest["entries"][key] = entry
                self._write_manifest()
            except BaseException:
                if previous is None:
                    _ = self.manifest["entries"].pop(key, None)
                else:
                    self.manifest["entries"][key] = previous
                shutil.rmtree(directory, ignore_errors=True)
                raise
            if previous is not None:
                shutil.rmtree(self.path / previous["directory"], ignore_errors=True)
        self.saved_at = monotonic()
        self.save_seconds = self.saved_at - start

    def discard(self, key: str) -> None:
        """Remove the entry ``key``."""
        with self.guard:
            entry = self.manifest["entries"].pop(key)
            self._write_manifest()
            shutil.rmtree(self.path / entry["directory"], ignore_errors=True)

    def _write_manifest(self) -> None:
        with replaced_when_complete(self.path / _MANIFEST, "a checkpoint record") as part:
            _ = part.write_text(json.dumps(self.manifest, default=_scalar))
        _fsync_directory(self.path)

    def due(self) -> bool:
        """Whether a loop at a boundary should save now."""
        wait = max(INTERVAL, SAVE_SHARE * self.save_seconds)
        return self.stop is not None or monotonic() - self.saved_at >= wait

    def stop_if_requested(self) -> None:
        """Raise :class:`Canceled` once the state is saved after SIGTERM."""
        if self.stop is not None:
            name = signal.Signals(self.stop).name
            raise Canceled(f"stopped by {name}: run again with checkpoint {self.path} to resume")

    @contextmanager
    def looping(self) -> Iterator[None]:
        """Within the block, SIGTERM waits for the loop's next boundary."""
        with self.guard:
            self.loops += 1
        try:
            yield
        finally:
            with self.guard:
                self.loops -= 1

    def close(self) -> None:
        """Release the directory, leaving the checkpoint for a later run."""
        os.close(self._lock)

    def remove(self) -> None:
        """Delete the checkpoint's files and, when nothing else is in it, the directory."""
        for entry in self.manifest["entries"].values():
            shutil.rmtree(self.path / entry["directory"], ignore_errors=True)
        (self.path / _MANIFEST).unlink(missing_ok=True)
        (self.path / _LOCK).unlink(missing_ok=True)
        self.close()
        with suppress(OSError):
            self.path.rmdir()


@dataclass(frozen=True)
class _Place:
    session: Session
    prefix: str


_place = ContextVar[_Place | None]("tomojax_checkpoint", default=None)
type _Handler = Callable[[int, FrameType | None], object] | int | signal.Handlers | None


def active() -> bool:
    """Whether the calling context is in a checkpointed run."""
    return _place.get() is not None


def _deliver(handler: _Handler, signum: int, frame: FrameType | None) -> None:
    """Act on ``signum`` as ``handler``, the one installed before the checkpoint's, would."""
    if callable(handler):
        _ = handler(signum, frame)
    elif handler != signal.SIG_IGN:
        signal.raise_signal(signum)


@contextmanager
def opened(path: str | os.PathLike[str]) -> Iterator[Session]:
    """Own the checkpoint at ``path`` for the block, and remove it when the block completes.

    In the main thread, SIGTERM inside a checkpointed loop is held for the loop's
    next boundary, which saves and stops; elsewhere it acts as before. A SIGTERM no
    loop acted on is delivered when the block ends. The checkpoint is kept when
    the block fails, is stopped or calls :func:`keep_checkpoint`.
    """
    session = Session(Path(path).resolve())
    main = threading.current_thread() is threading.main_thread()
    previous: _Handler = signal.getsignal(signal.SIGTERM) if main else None
    handling = main and previous != signal.SIG_IGN

    def terminate(signum: int, frame: FrameType | None) -> None:
        if session.loops:
            session.stop = signum
            return
        _ = signal.signal(signum, previous)
        _deliver(previous, signum, frame)

    if handling:
        _ = signal.signal(signal.SIGTERM, terminate)
    token = _place.set(_Place(session, ""))
    completed = False
    try:
        yield session
        completed = True
    finally:
        _place.reset(token)
        if handling:
            _ = signal.signal(signal.SIGTERM, previous)
        unused = session.manifest["identity"] is None  # failed before its run began
        if unused or (completed and session.stop is None and not session.kept):
            session.remove()
        else:
            session.close()
    if session.stop is not None and handling:
        _deliver(previous, session.stop, None)


@contextmanager
def checkpointing(path: str | os.PathLike[str] | None, **run: object) -> Iterator[None]:
    """Run the block in the checkpoint at ``path`` (None: none) of the run ``run`` describes.

    ``run`` holds the parts of the run's :func:`identity`. A checkpoint already
    open at ``path`` (as the command line opens one around loading and saving)
    is joined; otherwise this block owns it.
    """
    if path is None:
        yield
        return
    place = _place.get()
    if place is not None and place.session.path == Path(path).resolve():
        place.session.bind(identity(**run))
        yield
        return
    with opened(path) as session:
        session.bind(identity(**run))
        yield


def keep_checkpoint() -> None:
    """Keep the open checkpoint when its run ends: the work it records is not complete."""
    place = _place.get()
    if place is not None:
        place.session.kept = True


@contextmanager
def scope(name: str) -> Iterator[None]:
    """Keys claimed in the block belong to ``name``, so they do not depend on what ran before."""
    place = _place.get()
    if place is None:
        yield
        return
    token = _place.set(_Place(place.session, f"{place.prefix}{name}/"))
    try:
        yield
    finally:
        _place.reset(token)


@contextmanager
def suspended() -> Iterator[None]:
    """Loops in the block do not checkpoint: their caller saves what they make."""
    token = _place.set(None)
    try:
        yield
    finally:
        _place.reset(token)


class Loop:
    """A checkpointed loop: the state saved by an earlier run, and saves at its boundaries.

    At each boundary the loop asks :meth:`due` and, when it is, calls :meth:`save`
    with its arrays and a named tuple of its scalars, which :meth:`Saved.scalars`
    gives back.
    """

    def __init__(self, session: Session | None, key: str) -> None:
        self.session, self.key = session, key
        self.saved = None if session is None else session.load(key)

    def due(self) -> bool:
        """Whether the loop should save its state at this boundary."""
        return self.session is not None and self.session.due()

    def save(self, arrays: Sequence[object], scalars: NamedTuple | None = None) -> None:
        """Save the loop's state (without a checkpoint, nothing), then stop if SIGTERM arrived."""
        if self.session is None:
            return
        record: dict[str, JsonValue] = {} if scalars is None else scalars._asdict()
        self.session.save(self.key, arrays, record)
        self.session.stop_if_requested()


class Completed(NamedTuple):
    """The scalars of a loop whose state is its arrays and how many steps they hold."""

    steps: int


IDLE = Loop(None, "")
"""The loop of a run without a checkpoint: nothing saved, nothing to save."""


@contextmanager
def loop(kind: str = "solve") -> Iterator[Loop]:
    """The checkpointed loop of this kind at this place in the run (idle without a checkpoint)."""
    place = _place.get()
    if place is None:
        yield IDLE
        return
    session = place.session
    with session.looping():
        yield Loop(session, session.claim(place.prefix + kind))


def _fingerprint(block: np.ndarray) -> int:
    flat = block.reshape(-1)
    return zlib.crc32(np.ascontiguousarray(flat[:: max(1, flat.size // _SAMPLE)]).tobytes())


def _flush(block: np.ndarray) -> None:
    """Write a slab of a file mapping to its file."""
    owner = mapped_storage(block)
    mapping: mmap.mmap | None = getattr(owner, "_mmap", None)
    if mapping is None:
        return
    first = block.ctypes.data - np.frombuffer(mapping, np.uint8).ctypes.data
    page = first - first % mmap.ALLOCATIONGRANULARITY
    mapping.flush(page, first + block.nbytes - page)


class Slabs:
    """The finished z ranges of one slab loop, kept in the checkpoint.

    With an ``output`` volume the planes are restored into it when the tracker
    opens; a named file mapping keeps its own planes, and the checkpoint keeps a
    fingerprint to check them by. Without one the caller keeps what it made of
    each finished slab.
    """

    def __init__(self, session: Session | None, key: str, output: np.ndarray | None) -> None:
        self.session, self.key, self.output = session, key, output
        self.done: list[tuple[int, int]] = []
        if session is None:
            return
        for name in session.keys(f"{key}/"):
            start, stop = map(int, re.findall(r"\d+", name.rsplit("/", 1)[1]))
            saved = session.load(name)
            assert saved is not None
            if output is not None:
                planes = output[start:stop]
                if "fingerprint" not in saved.record:
                    np.copyto(planes, saved.arrays[0])
                elif _fingerprint(planes) != saved.record["fingerprint"]:
                    session.discard(name)  # the file no longer holds these planes
                    continue
            self.done.append((start, stop))

    @property
    def prefix(self) -> int:
        """The end of the finished planes ``[0, prefix)``: where a z-ordered loop resumes."""
        end = 0
        for low, high in sorted(self.done):
            if low > end:
                break
            end = max(end, high)
        return end

    def finished(self, start: int, stop: int) -> bool:
        """Whether planes ``[start, stop)`` all belong to finished slabs."""
        for low, high in sorted(self.done):
            if low <= start < high:
                start = high
            if start >= stop:
                return True
        return False

    def add(self, start: int, stop: int) -> None:
        """Record planes ``[start, stop)`` as finished, then stop if SIGTERM arrived."""
        session = self.session
        if session is None:
            return
        arrays: tuple[np.ndarray, ...] = ()
        record: dict[str, JsonValue] = {}
        if self.output is not None:
            planes = self.output[start:stop]
            owner = mapped_storage(planes)  # a named file outlives the process; scratch does not
            name = None if owner is None or not owner.flags.writeable else owner.filename
            if name is not None and Path(name).exists():
                _flush(planes)
                record = {"fingerprint": _fingerprint(planes)}
            else:
                arrays = (planes,)
        session.save(f"{self.key}/z{start}-{stop}", arrays, record)
        self.done.append((start, stop))
        session.stop_if_requested()


_IDLE_SLABS = Slabs(None, "", None)


@contextmanager
def slabs(output: np.ndarray | None = None) -> Iterator[Slabs]:
    """The finished slabs of the slab loop at this place in the run (none without a checkpoint)."""
    place = _place.get()
    if place is None:
        yield _IDLE_SLABS
        return
    session = place.session
    with session.looping():
        yield Slabs(session, session.claim(place.prefix + "slabs"), output)


@runtime_checkable
class _Array(Protocol):
    """NumPy, JAX and CuPy arrays, and arrays read lazily from files."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[np.generic]: ...

    def __getitem__(self, index: int, /) -> object: ...


def _array_digest(value: _Array) -> str:
    """The shape, type and contents of an array, large ones by a sample of eight slices."""
    shape = tuple(value.shape)
    digest = hashlib.sha256(f"{shape}{np.dtype(value.dtype).str}".encode())
    if not shape or math.prod(shape) <= 4 * _SAMPLE:
        rows: list[object] = [value]
    else:
        count = min(shape[0], 8)
        rows = [value[i] for i in sorted({(shape[0] - 1) * k // (count - 1) for k in range(count)})]
    for row in rows:
        flat = _host(row).reshape(-1)
        digest.update(np.ascontiguousarray(flat[:: max(1, flat.size // _SAMPLE)]).tobytes())
    return digest.hexdigest()


def _canonical(value: object) -> JsonValue:
    """A JSON value standing for ``value``: its fields, items, or array contents by digest."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, _Array):
        return {"array": _array_digest(value)}
    if is_dataclass(value) and not isinstance(value, type):
        named = {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
        return {"type": type(value).__name__, "fields": named}
    if isinstance(value, Mapping):
        mapping: Mapping[object, object] = value
        return {str(key): _canonical(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple | set | frozenset):
        collection: Iterable[object] = value
        items = [_canonical(item) for item in collection]
        return sorted(items, key=json.dumps) if isinstance(value, set | frozenset) else items
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))  # devices and the like


def identity(**parts: object) -> dict[str, str]:
    """Digests naming a run, one for each part: settings exactly, large arrays by a sample.

    Arrays of more than four million elements are sampled: eight evenly spaced
    slices along the first axis, each at most a million evenly spaced elements.
    """
    from tomojax._version import __version__

    named: dict[str, object] = {"version": __version__, **parts}
    canonical = {
        name: json.dumps(_canonical(value), sort_keys=True) for name, value in named.items()
    }
    return {
        name: hashlib.sha256(text.encode()).hexdigest()[:32] for name, text in canonical.items()
    }


__all__ = [
    "IDLE",
    "INTERVAL",
    "SAVE_SHARE",
    "Completed",
    "Loop",
    "Saved",
    "Session",
    "Slabs",
    "active",
    "checkpointing",
    "identity",
    "keep_checkpoint",
    "loop",
    "opened",
    "scope",
    "slabs",
    "suspended",
]
