"""Synchronous progress events and cooperative cancellation."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import wraps
import inspect
from itertools import count
import logging
from math import isfinite
import os
from pathlib import Path
import tempfile
from threading import Condition, RLock
from time import monotonic
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args
import warnings

import msgspec
import numpy as np

from tomojax.errors import Canceled, classify
from tomojax.wire.arrays import Array, Axis, Column, Coordinates, Fixed, Npy
from tomojax.wire.metrics import Metric
from tomojax.wire.progress import NoteLevel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping


__all__ = [
    "Advanced",
    "Canceled",
    "Control",
    "Event",
    "Finished",
    "Note",
    "Paused",
    "Placed",
    "Preview",
    "Resumed",
    "Started",
    "to_json",
    "watch",
]

type _Metric = float | int | str | None
type _Name = Literal[
    "load",
    "correction",
    "reconstruct",
    "align",
    "level",
    "refine",
    "metadata",
    "simulate",
    "save",
    "project",
    "backproject",
    "inspect",
    "export",
    "estimate_region",
    "opposite-views",
    "image",
    "prepare",
    "step",
    "linearize",
    "solve",
    "retry",
    "line_search",
    "write",
]
type _Status = Literal["done", "failed", "canceled"]


@dataclass(frozen=True, slots=True)
class Started:
    """An operation entered by its calling thread.

    Parameters
    ----------
    task : int
        Identifier unique to this watch.
    parent : int or None
        Enclosing task, if any.
    name : str
        The operation or numerical stage identifier.
    label : str
        Display text for the task, preserving file names.
    total : int or None
        Expected number of units, if known.
    unit : str
        Unit of completed work.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    parent: int | None
    name: _Name
    label: str
    total: int | None
    unit: str
    time: float


@dataclass(frozen=True, slots=True)
class Advanced:
    """Completed units and a snapshot of the associated diagnostics.

    Parameters
    ----------
    task : int
        Task identifier.
    done : int
        Number of completed units.
    total : int or None
        Current estimate of the total units, if known.
    metrics : mapping
        Snapshot of scalar diagnostics for the completed work.
    time : float
        Monotonic seconds since the watch began.
    eta : float or None, optional
        Remaining active seconds at the measured completion rate.
    """

    task: int
    done: int
    total: int | None
    metrics: Mapping[str, _Metric]
    time: float
    eta: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class Finished:
    """The outcome of an operation.

    Parameters
    ----------
    task : int
        Task identifier.
    status : str
        One of done, failed or canceled.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    status: _Status
    time: float


@dataclass(frozen=True, slots=True)
class Placed:
    """The coordinator resolved the devices participating in a task.

    Parameters
    ----------
    task : int
        Active task identifier.
    devices : tuple of str
        Participating device IDs in coordinator order.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    devices: tuple[str, ...]
    time: float


@dataclass(frozen=True, slots=True)
class Paused:
    """A task reached a work boundary while paused.

    Parameters
    ----------
    task : int
        Innermost active task.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    time: float


@dataclass(frozen=True, slots=True)
class Resumed:
    """A paused task resumed computation.

    Parameters
    ----------
    task : int
        Innermost active task.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    time: float


@dataclass(frozen=True, slots=True)
class Note:
    """A message from the watched code.

    Parameters
    ----------
    task : int or None
        Innermost task ID or None between operations.
    level : str
        Severity of the message.
    message : str
        Formatted log or warning text.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int | None
    level: NoteLevel
    message: str
    time: float


@dataclass(frozen=True, slots=True)
class Sample:
    """A small host image and its registration at a completed work boundary."""

    data: np.ndarray
    axes: tuple[Axis, ...]
    unit: str
    domain: str
    fixed: tuple[Fixed, ...] = ()
    columns: tuple[Column, ...] = ()
    coordinates: tuple[tuple[int, np.ndarray], ...] = ()

    def store(self, write: Callable[[np.ndarray], Npy]) -> Array:
        axes = list(self.axes)
        for index, values in self.coordinates:
            axes[index] = msgspec.structs.replace(
                axes[index], coordinates=Coordinates(write(values))
            )
        return Array(
            write(self.data),
            self.data.shape,
            self.data.dtype.str,
            tuple(axes),
            self.unit,
            self.domain,
            self.fixed,
            self.columns,
        )


@dataclass(frozen=True, slots=True)
class Preview:
    """Coherent small arrays copied at one completed work boundary.

    Parameters
    ----------
    task : int
        Owning task identifier.
    arrays : mapping
        Named host images with axes, fixed coordinates and value units.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    arrays: Mapping[str, Sample]
    time: float


type Event = Started | Placed | Advanced | Finished | Paused | Resumed | Note | Preview
type _State = Literal["running", "pause-requested", "paused"]


class Control:
    """Coordinate pause, resume and cancellation between an owner and a controller."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._state: _State = "running"
        self._canceled = False
        self._deferring = 0

    def pause(self) -> None:
        """Request a pause at the next work boundary."""
        with self._condition:
            if self._state == "running" and not self._canceled:
                self._state = "pause-requested"

    def resume(self) -> None:
        """Wake the paused operation or withdraw a pending pause."""
        with self._condition:
            self._state = "running"
            self._condition.notify_all()

    def cancel(self) -> None:
        """Request cancellation and wake a paused operation."""
        with self._condition:
            self._canceled = True
            self._condition.notify_all()

    @property
    def state(self) -> _State:
        """Return the cooperative state of the active operation.

        Returns
        -------
        str
            Running, pause-requested or paused.
        """
        with self._condition:
            return self._state

    @property
    def canceled(self) -> bool:
        """Whether cancellation has been requested.

        Returns
        -------
        bool
            True once cancellation has been requested.
        """
        with self._condition:
            return self._canceled

    @contextmanager
    def deferring(self) -> Iterator[None]:
        """Within, a cancel waits for the code that saves state at its own boundary.

        A checkpointed loop enters this block: on cancel it saves at its next
        boundary and raises :class:`Canceled` itself, so the work boundaries inside
        it let the cancel pass. Outside the block a cancel stops the work at the
        next boundary.

        Yields
        ------
        None
            The extent of the deferral.
        """
        with self._condition:
            self._deferring += 1
        try:
            yield
        finally:
            with self._condition:
                self._deferring -= 1

    def boundary(self) -> None:
        """Mark a work boundary: block while paused, or raise when canceled.

        Raises
        ------
        Canceled
            If the operation was canceled, unless a checkpointed loop defers it.
        """
        while True:
            with self._condition:
                if self._canceled:
                    if self._deferring:
                        return
                    raise Canceled()
                if self._state == "running":
                    return
                self._state = "paused"
            active = current_task.get()
            start = monotonic()
            if active is not None:
                active.watcher.emit(Paused, active.id)
            with self._condition:
                self._condition.wait_for(lambda: self._canceled or self._state != "paused")
                if self._canceled and not self._deferring:
                    raise Canceled()
            if active is not None:
                active.watcher.pause_time += monotonic() - start
                active.watcher.emit(Resumed, active.id)


def to_json(event: Event, *, directory: str | Path | None = None) -> dict[str, object]:
    """Return a versioned JSON-compatible record for an event.

    Parameters
    ----------
    event : Event
        Progress event to encode.
    directory : str or pathlib.Path or None, optional
        Consumer-owned directory for the NumPy arrays in a preview.

    Returns
    -------
    dict
        Versioned record suitable for strict JSON encoding.
    """
    if isinstance(event, Preview):
        if directory is None:
            raise ValueError("encoding a preview requires its consumer directory")
        files = []
        try:

            def write(data: np.ndarray) -> Npy:
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as f:
                    temporary = Path(f.name)
                    files.append(temporary)
                    np.save(f, data, allow_pickle=False)
                path = temporary.with_suffix(".npy")
                os.link(temporary, path)
                files.append(path)
                temporary.unlink()
                return Npy(str(path))

            arrays = {
                name: msgspec.to_builtins(sample.store(write))
                for name, sample in event.arrays.items()
            }
        except BaseException:
            for path in files:
                path.unlink(missing_ok=True)
            raise
        return {
            "schema": 2,
            "event": "preview",
            "task": event.task,
            "time": event.time,
            "arrays": arrays,
        }
    record = {field.name: getattr(event, field.name) for field in fields(event)}
    if isinstance(event, Advanced):
        record["metrics"] = {
            key: None if isinstance(value, float) and not isfinite(value) else value
            for key, value in event.metrics.items()
        }
    return {"schema": 2, "event": type(event).__name__.lower(), **record}


class _Watch:
    def __init__(
        self, handler: Callable[[Event], None], control: Control | None, *, previews: bool
    ) -> None:
        self.previews = previews
        self.last_preview = float("-inf")
        self.handler = handler
        self.control = control
        self.pause_time = 0.0
        self.ids = count(1)
        self.start = monotonic()
        self.lock = RLock()
        self.error: BaseException | None = None
        self.logged: set[tuple[int, str]] = set()  # each log message is one note per watch

    def emit(self, kind: Callable[..., Event], *args: object, **kwargs: object) -> None:
        with self.lock:
            if self.error is not None:
                raise self.error
            try:
                self.handler(kind(*args, time=monotonic() - self.start, **kwargs))
            except BaseException as exc:
                self.error = exc
                raise


_watch = ContextVar[_Watch | None]("tomojax_watch", default=None)


class _Task:
    def __init__(self, watcher: _Watch, name: _Name, unit: str, total: int | None) -> None:
        self.name, self.unit = name, unit
        self.watcher = watcher
        self.id = next(watcher.ids)
        self.status: _Status = "done"
        self.start = monotonic() - watcher.pause_time
        self.done = 0
        self.previewed = -1
        self.devices: tuple[str, ...] = ()
        self.total = total

    def advance(self, done: int, metrics: Mapping[str, _Metric]) -> None:
        if done < self.done or (self.total is not None and done > self.total):
            raise ValueError("completed work outside task bounds")
        if unknown := metrics.keys() - Metric.__members__.keys():
            raise ValueError(f"undeclared progress metrics: {sorted(unknown)}")
        self.done = done
        elapsed = monotonic() - self.watcher.pause_time - self.start
        eta = (self.total - done) * elapsed / done if self.total is not None and done else None
        self.watcher.emit(Advanced, self.id, done, self.total, metrics, eta=eta)


current_task = ContextVar[_Task | None]("tomojax_task", default=None)


def previewing() -> bool:
    watcher = _watch.get()
    return watcher is not None and watcher.previews


@contextmanager
def previews() -> Iterator[None]:
    """Deliver preview arrays to the active watch for the duration of the block.

    A client that wants previews of one stage only (the solve, not the loading
    before it) watches without them and turns them on around that stage. Task
    identifiers, parents and cancellation span the whole watch.
    """
    watcher = _watch.get()
    if watcher is None:
        yield
        return
    before, watcher.previews = watcher.previews, True
    try:
        yield
    finally:
        watcher.previews = before


def preview(build: Callable[[], dict[str, Sample]], *, final: bool = False) -> None:
    """Copy coherent arrays at a work boundary only when a preview will be delivered."""
    active = current_task.get()
    if active is None or not active.watcher.previews or active.previewed == active.done:
        return
    watcher = active.watcher
    now = monotonic() - watcher.pause_time
    if not final and now - watcher.last_preview < 1:
        return
    try:
        check_canceled()
        arrays = build()
        check_canceled()
    except Canceled as exc:
        if handler_failed(exc) or not (final or active.done == active.total):
            raise
        return
    watcher.last_preview = monotonic() - watcher.pause_time
    active.previewed = active.done
    watcher.emit(Preview, active.id, MappingProxyType(arrays))


def watching() -> bool:
    """Whether the calling context has an active watch."""
    return _watch.get() is not None


def for_unit(unit: str) -> _Task | None:
    """The active task when its owner is counting this unit of work."""
    active = current_task.get()
    return active if active is not None and active.unit == unit else None


def remaining(unit: str, count: int) -> None:
    """Update the remaining work after planning a batch or an allocation retry."""
    active = for_unit(unit)
    if active is not None:
        active.total = active.done + count


def step(unit: str, *, remaining: int | None = None) -> None:
    """Complete one unit, updating the total if the remaining plan is known."""
    active = for_unit(unit)
    if active is not None:
        done = active.done + 1
        if remaining is not None:
            active.total = done + remaining
        active.advance(done, {})


def preparing(label: str, first: int = 0) -> AbstractContextManager[None]:
    """Mark an owned program's first dispatch or explicit compilation on its caller."""
    return task("prepare", label, None, "") if first == 0 and watching() else nullcontext()


def chunks(total: int) -> Iterator[tuple[int, int]]:
    """Group existing work units between owner work boundaries, using completed chunk time."""
    if not watching():
        if total:
            yield 0, total
        return
    first, length = 0, 1
    while first < total:
        check_canceled()
        stop = min(total, first + length)
        start = monotonic()
        yield first, stop
        if first:  # The first dispatch can include compilation.
            length = max(1, round(0.25 * (stop - first) / max(monotonic() - start, 1e-9)))
        first = stop


class Runtime:
    """Devices and allocation retries observed by one workflow's coordinators."""

    def __init__(self) -> None:
        self.devices: dict[str, None] = {}
        self.retries: dict[str, int] = {}
        self.lock = RLock()

    def record(self, *, slabbed: bool) -> dict[str, object]:
        """Summarize decisions after all workers have joined."""
        return {
            "devices": tuple(self.devices),
            "strategy": "slabbed" if slabbed else "resident",
            "retries": dict(self.retries),
        }


current_runtime: ContextVar[Runtime | None] = ContextVar("tomojax_runtime", default=None)


@contextmanager
def runtime() -> Iterator[Runtime]:
    """Collect actual placement without installing a progress handler or moving work boundaries."""
    value = Runtime()
    token = current_runtime.set(value)
    try:
        yield value
    finally:
        current_runtime.reset(token)


def tracking_placement() -> bool:
    """Whether an event or a run result needs the participating device IDs."""
    return current_task.get() is not None or current_runtime.get() is not None


def retried(unit: str) -> None:
    """Count allocation failures that actually led to another attempt."""
    value = current_runtime.get()
    if value is not None:
        with value.lock:
            value.retries[unit] = value.retries.get(unit, 0) + 1


def placed(devices: tuple[str, ...]) -> None:
    """Report a change of placement within the current coordinator's task."""
    run = current_runtime.get()
    if run is not None:
        with run.lock:
            run.devices.update(dict.fromkeys(devices))
    active = current_task.get()
    if active is not None and active.devices != devices:
        active.devices = devices
        active.watcher.emit(Placed, active.id, devices)


def note(level: NoteLevel, message: str) -> None:
    """Deliver a message to the innermost watch in the calling context."""
    watcher = _watch.get()
    if watcher is not None:
        active = current_task.get()
        watcher.emit(Note, None if active is None else active.id, level, message)


def warn(message: str, *, stacklevel: int) -> None:
    """Issue a TomoJAX UserWarning at its public caller and report visible warnings.

    Leave the interpreter in charge of filtering, repeat suppression, promotion to
    exceptions, and delivery. Snapshot the two repeat keys because the first
    delivery writes them into the caller's registry.
    """
    if _watch.get() is None:
        warnings.warn(message, stacklevel=stacklevel + 1)
        return
    frame = inspect.currentframe()
    for _ in range(stacklevel):
        frame = frame.f_back if frame else None
    namespace = frame.f_globals if frame else {}
    lineno = frame.f_lineno if frame else 1
    location = (message, UserWarning, lineno)
    module_key = (message, UserWarning)
    registry = namespace.setdefault("__warningregistry__", {})
    version = registry.get("version")
    seen_here, seen_module = registry.get(location), registry.get(module_key)
    # Python 3.14 may keep catch_warnings filters in a context variable.
    filters = getattr(warnings, "_get_filters", lambda: warnings.filters)()
    module = namespace.get("__name__", "")
    action = next(
        (
            action
            for action, text, category, mod, line in filters
            if issubclass(UserWarning, category)
            and (not text or text.match(message))
            and (not mod or mod.match(module))
            and line in (0, lineno)
        ),
        getattr(warnings, "defaultaction", "default"),
    )
    warnings.warn(message, stacklevel=stacklevel + 1)
    repeated = seen_here or (action in ("module", "once") and seen_module)
    if action != "ignore" and (version != registry.get("version") or not repeated):
        note("warning", message)


class _LogNotes(logging.NullHandler):
    """Observe the TomoJAX logger hierarchy without serializing independent watches."""

    def handle(self, record: logging.LogRecord) -> bool:
        # Adding a handler normally disables logging's last-resort stderr handler.
        # Preserve that delivery when this observer is the only reachable handler.
        logger: logging.Logger | None = logging.getLogger(record.name)
        while logger is not None:
            if any(handler is not self for handler in logger.handlers):
                break
            logger = logger.parent if logger.propagate else None
        else:
            fallback = logging.lastResort
            if fallback is not None and record.levelno >= fallback.level:
                fallback.handle(record)
        watcher = _watch.get()
        message = record.getMessage()
        # Readers that open a file more than once in one operation log the same finding.
        if watcher is not None and _logging_watch.get() is not watcher:
            with watcher.lock:
                repeated = (record.levelno, message) in watcher.logged
                watcher.logged.add((record.levelno, message))
            if repeated:
                return True
            token = _logging_watch.set(watcher)
            try:
                level = min(5, max(1, record.levelno // 10))
                note(_LEVELS[level - 1], message)
            finally:
                _logging_watch.reset(token)
        return True


_LEVELS = get_args(NoteLevel.__value__)
_logging_watch = ContextVar[_Watch | None]("tomojax_logging_watch", default=None)
_log_notes = _LogNotes()


@contextmanager
def watch(
    handler: Callable[[Event], None], *, control: Control | None = None, previews: bool = False
) -> Iterator[None]:
    """Deliver events synchronously to a fast handler in the calling context.

    Parameters
    ----------
    handler : callable
        Receives each event synchronously and must return quickly. An exception stops
        delivery and propagates to the caller.
    control : Control or None, optional
        Token which another thread or the handler may set to stop the operation.
    previews : bool, optional
        Deliver registered preview arrays from completed numerical work boundaries.

    Yields
    ------
    None
        The dynamic extent of this watch in the current context.
    """
    logging.getLogger("tomojax").addHandler(_log_notes)
    token = _watch.set(_Watch(handler, control, previews=previews))
    parent = current_task.set(None)
    try:
        yield
    finally:
        current_task.reset(parent)
        _watch.reset(token)


def handler_failed(exc: BaseException) -> bool:
    """Identify a handler exception at an internal recovery boundary."""
    watcher = _watch.get()
    return watcher is not None and watcher.error is exc


def active_control() -> Control | None:
    """The control of the innermost watch in the calling context, if it has one.

    Returns
    -------
    Control or None
        The token a controller uses to pause or cancel the watched work.
    """
    watcher = _watch.get()
    return None if watcher is None else watcher.control


def check_canceled() -> None:
    watcher = _watch.get()
    if watcher is not None and watcher.control is not None:
        watcher.control.boundary()


@contextmanager
def outcome() -> Iterator[None]:
    """Retain iterator failures until the owning operation's context exits."""
    active = current_task.get()
    try:
        yield
    except GeneratorExit:
        raise
    except BaseException as exc:
        if active is not None:
            active.status = "canceled" if isinstance(exc, Canceled) else "failed"
        raise


@contextmanager
def task(name: _Name, label: str, total: int | None, unit: str) -> Iterator[None]:
    watcher = _watch.get()
    if watcher is None:
        yield
        return
    parent = current_task.get()
    if parent is not None:
        check_canceled()
    # Initializers, slab solves and alignment image updates belong to their owner.
    if name == "correction" and parent is not None and parent.name != "load":
        yield
        return
    if (
        name == "reconstruct"
        and parent is not None
        and parent.name in {"reconstruct", "level", "refine"}
    ):
        yield
        return
    task = _Task(watcher, name, unit, total)
    token = current_task.set(task)
    try:
        if name in ("correction", "align", "level", "refine", "reconstruct"):
            label = label.replace("_", " ").lower()
        watcher.emit(Started, task.id, parent.id if parent else None, name, label, total, unit)
        try:
            with outcome():
                check_canceled()
                yield
        finally:
            if watcher.error is None:
                watcher.emit(Finished, task.id, task.status)
    finally:
        current_task.reset(token)


def reports_iterations[F: Callable[..., Any]](default: int) -> Callable[[F], F]:
    """Start an expert solver's task when it is called directly."""

    def decorate(solve: F) -> F:
        @wraps(solve)
        def run(*args: Any, **kwargs: Any) -> Any:
            iterations = getattr(kwargs.get("config"), "iterations", default)
            with task(
                "reconstruct",
                solve.__name__,
                default if iterations is None else iterations,
                "iterations",
            ):
                return solve(*args, **kwargs)

        return cast("F", run)

    return decorate


def error_record(error: Exception) -> dict[str, object]:
    """Encode a command failure using the same schema as progress events."""
    import msgspec

    return msgspec.to_builtins(classify(error))
