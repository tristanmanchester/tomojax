"""Physical input files retained by live acquisitions and their derived results."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import gc
from itertools import chain
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal
from weakref import WeakSet

from tomojax.errors import InvalidParameter

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_lock = RLock()
_pins: Counter[InputFiles] = Counter()
_python_results: WeakSet[InputFiles] = WeakSet()
_owns_results = ContextVar("tomojax_owns_load_results", default=False)
_current: ContextVar[InputFiles | None] = ContextVar("tomojax_input_files", default=None)


class InputFiles:
    """Keep the names and filesystem identities discovered during one read."""

    def __init__(self, parents: Iterable[InputFiles] = ()) -> None:
        self.parents = tuple(parents)
        self.paths: dict[Path, Path] = {}
        self.identities: dict[tuple[int, int], Path] = {}
        self.containers: set[tuple[Path, Path]] = set()
        self.ancestors: dict[Path, Path] = {}

    def lineage(self) -> set[InputFiles]:
        """Include parents that may still discover more files after issuing a preview."""
        groups: set[InputFiles] = set()
        pending: list[InputFiles] = [self]
        while pending:
            group = pending.pop()
            if group not in groups:
                groups.add(group)
                pending.extend(group.parents)
        return groups

    @contextmanager
    def pinned(self) -> Iterator[None]:
        """Hold the input and its ancestry until this operation or resource ends."""
        groups = self.lineage()
        with _lock:
            _pins.update(groups)
        try:
            yield
        finally:
            with _lock:
                for group in groups:
                    _pins[group] -= 1
                    if not _pins[group]:
                        del _pins[group]

    def include(self, child: InputFiles) -> None:
        """Keep files a Python request opens for the handles and arrays it publishes."""
        with _lock:
            self.paths.update(child.paths)
            self.identities.update(child.identities)
            self.ancestors.update(child.ancestors)

    def add(self, path: Path, owner: Path) -> None:
        """Snapshot both names and identity before consuming this file."""
        path = path.absolute()
        with _lock:
            if path in self.paths:
                return
            resolved = path.resolve()
            self.paths[path] = self.paths[resolved] = owner
            for parent in chain(path.parents, resolved.parents):
                _ = self.ancestors.setdefault(parent, path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                pass  # A VDS can supply fill values until its source appears.
            else:
                self.identities[stat.st_dev, stat.st_ino] = path

    def read(self, path: Path) -> None:
        """Resolve a container once, retaining the spelling used for relative links."""
        import h5py

        from tomojax.core.files import hdf5_sources

        path = path.absolute()
        if not path.exists():
            # Inspection may first restore a missing primary from its previous directory.
            return
        pending = [path]
        while pending:
            here = pending.pop()
            self.add(here, path)
            identity = here.resolve(), here.parent.resolve()
            if identity in self.containers:
                continue
            self.containers.add(identity)
            if here.is_file() and h5py.is_hdf5(here):
                with h5py.File(here, "r") as file:
                    linked, raw = hdf5_sources(file)
                for storage in raw:
                    self.add(storage, path)
                pending.extend(linked)


def current_inputs() -> tuple[InputFiles, ...]:
    """Input lifetimes inherited by a result constructed inside a reader."""
    group = _current.get()
    return () if group is None else (group,)


@contextmanager
def reading(
    *paths: Path,
    group: InputFiles | None = None,
    owner: Literal["operation", "request", "result"] = "operation",
) -> Iterator[None]:
    """Pin a read scope and transfer ownership only for a successful public Python load."""
    outer = _current.get()
    public_result = owner == "result" and not _owns_results.get()
    inputs = InputFiles() if public_result else group or outer or InputFiles()
    token = _current.set(inputs)
    ownership = _owns_results.set(owner != "operation" or _owns_results.get())
    try:
        with inputs.pinned():
            with _lock:
                for path in paths:
                    inputs.read(path)
            yield
            if public_result:
                with _lock:
                    _python_results.add(inputs)
                    if outer is not None:
                        outer.include(inputs)
    finally:
        _owns_results.reset(ownership)
        _current.reset(token)


@contextmanager
def python_workspace() -> Iterator[None]:
    """Let user Python own its load results while recording dependencies for the job."""
    token = _owns_results.set(False)
    try:
        yield
    finally:
        _owns_results.reset(token)


def expose_to_python(groups: Iterable[InputFiles]) -> None:
    """Retain a worker value exposed to user Python for as long as it stays reachable."""
    with _lock:
        for group in groups:
            _python_results.update(group.lineage())


def input_path(path: Path, *, container: bool = False) -> Path:
    """Retain a file selected by a reader, including vendor frame and metadata files."""
    if (inputs := _current.get()) is not None:
        if container:
            with _lock:
                inputs.read(path)
        else:
            inputs.add(path, path)
    return path


def check_output(path: Path, *, field: str = "path") -> None:
    """Refuse replacement of a live input, any alias of it, or a directory containing it."""
    with publishing(path, field=field):
        pass


def _collision(path: Path, field: str, groups: tuple[InputFiles, ...]) -> InvalidParameter | None:
    if not any(group.paths for group in groups):
        return None
    resolved = path.resolve()
    lexical = path.absolute()
    for group in groups:
        for name in (resolved, lexical):
            source = group.ancestors.get(name, name)
            if owner := group.paths.get(source):
                return _conflict(path, source, owner, field)
    # A destination directory can contain a hard link under a different name.
    entries = path.walk() if path.is_dir() else ()
    members = (root / name for root, dirs, files in entries for name in dirs + files)
    for entry in chain((path,), members):
        try:
            stat = entry.stat()
        except FileNotFoundError:
            continue
        for group in groups:
            if source := group.identities.get((stat.st_dev, stat.st_ino)):
                return _conflict(path, source, group.paths[source], field)
    return None


def check_checkpoint(path: str | Path | None) -> None:
    """Check the whole directory before opening or resuming a checkpoint."""
    if path is not None:
        check_output(Path(path), field="checkpoint")


@contextmanager
def publishing(*paths: Path, field: str = "path") -> Iterator[None]:
    """Hold the input registry stable from the last destination check through publication."""
    # Only a returned Python value needs a reachability decision. The collector may
    # run arbitrary finalizers, so do that before taking the publication lock.
    with _lock:
        python_conflict = any(_collision(path, field, tuple(_python_results)) for path in paths)
    if python_conflict:
        _ = gc.collect()
    with _lock:
        for path in paths:
            if error := _collision(path, field, (*_pins, *_python_results)):
                raise error
        yield


def _conflict(path: Path, source: Path, owner: Path, field: str) -> InvalidParameter:
    dependency = str(source) if source == owner else f"{source} (used by {owner})"
    return InvalidParameter(
        field,
        f"output path must differ from input path: {path} would replace input {dependency}. "
        "Files stay protected while a running operation, open handle, unreleased artifact or "
        "reachable Python result depends on them. Choose a different destination.",
    )
