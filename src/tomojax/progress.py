"""Synchronous progress events and cooperative cancellation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import wraps
from itertools import count
from math import isfinite
from threading import Event as ThreadEvent, RLock
from time import monotonic
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = ["Advanced", "Cancel", "Cancelled", "Event", "Finished", "Started", "to_json", "watch"]

type _Metric = float | int | str | None
type _Name = Literal["load", "correction", "reconstruct", "align", "level", "refine"]
type _Status = Literal["done", "failed", "cancelled"]


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
        One of load, correction, reconstruct, align, level, or refine.
    label : str
        Lower-case human text describing this task.
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
    """

    task: int
    done: int
    total: int | None
    metrics: Mapping[str, _Metric]
    time: float

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
        One of done, failed or cancelled.
    time : float
        Monotonic seconds since the watch began.
    """

    task: int
    status: _Status
    time: float


type Event = Started | Advanced | Finished


class Cancelled(Exception):
    """The operation stopped at a cancellation boundary."""


class Cancel:
    """Request cancellation from any thread."""

    def __init__(self) -> None:
        self._signal = ThreadEvent()

    def cancel(self) -> None:
        """Request cancellation at the next work boundary."""
        self._signal.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested.

        Returns
        -------
        bool
            True once the token has been set.
        """
        return self._signal.is_set()


def to_json(event: Event) -> dict[str, object]:
    """Return a versioned JSON-compatible record for an event.

    Parameters
    ----------
    event : Event
        Progress event to encode.

    Returns
    -------
    dict
        Versioned record suitable for strict JSON encoding.
    """
    record = {field.name: getattr(event, field.name) for field in fields(event)}
    if isinstance(event, Advanced):
        record["metrics"] = {
            key: None if isinstance(value, float) and not isfinite(value) else value
            for key, value in event.metrics.items()
        }
    return {"schema": 1, "event": type(event).__name__.lower(), **record}


class _Watch:
    def __init__(self, handler: Callable[[Event], None], cancel: Cancel | None) -> None:
        self.handler = handler
        self.cancel = cancel
        self.ids = count(1)
        self.start = monotonic()
        self.lock = RLock()
        self.error: BaseException | None = None

    def emit(self, kind: Callable[..., Event], *args: object) -> None:
        with self.lock:
            if self.error is not None:
                raise self.error
            try:
                self.handler(kind(*args, time=monotonic() - self.start))
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
        self.done = 0
        self.total = total

    def advance(self, done: int, metrics: Mapping[str, _Metric]) -> None:
        if done < self.done or (self.total is not None and done > self.total):
            raise ValueError("completed work outside task bounds")
        self.done = done
        self.watcher.emit(Advanced, self.id, done, self.total, metrics)


current_task = ContextVar[_Task | None]("tomojax_task", default=None)


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


@contextmanager
def watch(handler: Callable[[Event], None], *, cancel: Cancel | None = None) -> Iterator[None]:
    """Deliver events synchronously to a fast handler in the calling context.

    Parameters
    ----------
    handler : callable
        Receives each event synchronously and must return quickly. An exception stops
        delivery and propagates to the caller.
    cancel : Cancel or None, optional
        Token which another thread or the handler may set to stop the operation.

    Yields
    ------
    None
        The dynamic extent of this watch in the current context.
    """
    token = _watch.set(_Watch(handler, cancel))
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


def check_cancelled() -> None:
    watcher = _watch.get()
    if watcher is not None and watcher.cancel is not None and watcher.cancel.cancelled:
        raise Cancelled()


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
            active.status = "cancelled" if isinstance(exc, Cancelled) else "failed"
        raise


@contextmanager
def task(name: _Name, label: str, total: int | None, unit: str) -> Iterator[None]:
    watcher = _watch.get()
    if watcher is None:
        yield
        return
    check_cancelled()
    parent = current_task.get()
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
        label = label.replace("_", " ").lower()
        watcher.emit(Started, task.id, parent.id if parent else None, name, label, total, unit)
        try:
            with outcome():
                check_cancelled()
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
            with task("reconstruct", solve.__name__, iterations, "iterations"):
                return solve(*args, **kwargs)

        return cast("F", run)

    return decorate


def error_record(error: Exception) -> dict[str, object]:
    """Encode a command failure using the same schema as progress events."""
    return {"schema": 1, "event": "error", "type": type(error).__name__, "message": str(error)}
