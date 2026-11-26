"""Drive one donated JAX loop from the operation's calling thread."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from contextvars import ContextVar
from functools import partial
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

import jax

from tomojax import progress

type Metrics = Mapping[str, float | int | str | None]


if TYPE_CHECKING:

    class IterationProgress(Protocol):
        def advance(self, done: int, metrics: Metrics) -> None: ...


type Drive[S] = Callable[
    [
        S,
        int,
        Callable[[S, int], S],
        Callable[[S], tuple[int, bool, Metrics]],
        Callable[[S], dict[str, progress.Sample]],
    ],
    S,
]


# None outside an axial partition, false in its final slab, true while later slabs remain.
partial_sweep = ContextVar[bool | None]("tomojax_partial_sweep", default=None)


def run_chunks[S](
    state: S,
    end: int,
    advance: Callable[[S, int], S],
    completed: Callable[[S], tuple[int, bool, Metrics]],
    sample: Callable[[S], dict],
    *,
    task: IterationProgress | None,
) -> S:
    if task is None:
        return advance(state, end)
    jax.block_until_ready(state)
    done, stopped, metrics = completed(state)
    if done:
        task.advance(done, metrics)
        if partial_sweep.get() is None:
            progress.preview(lambda state=state: sample(state), final=stopped or done == end)
    length = 1
    compiled = False
    while done < end and not stopped:
        progress.check_canceled()
        start = monotonic()
        with progress.preparing("first update", int(compiled)):
            state = jax.block_until_ready(advance(state, min(end, done + length)))
        elapsed = monotonic() - start
        previous = done
        done, stopped, metrics = completed(state)
        if not partial_sweep.get():
            task.advance(done, metrics)
            if partial_sweep.get() is None:
                progress.preview(lambda state=state: sample(state), final=stopped or done == end)
        if compiled:
            length = max(1, round(0.25 * (done - previous) / max(elapsed, 1e-9)))
        compiled = True
    return state


class Chunked[R]:
    """Fuse all phases when unwatched. Only watched solves materialize chunk state."""

    def __init__(self, program: Callable[..., R], *, static_argnames: tuple[str, ...]) -> None:
        self.program = program
        self.unwatched = jax.jit(
            partial(program, drive=partial(run_chunks, task=None)), static_argnames=static_argnames
        )

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        task = progress.for_unit("iterations")
        if task is None:
            return self.unwatched(*args, **kwargs)
        progress.check_canceled()
        with ExitStack() as setup:
            setup.enter_context(progress.preparing("initialize solver"))

            def drive[S](
                state: S,
                end: int,
                advance: Callable[[S, int], S],
                completed: Callable[[S], tuple[int, bool, Metrics]],
                sample: Callable[[S], dict[str, progress.Sample]],
            ) -> S:
                setup.close()
                return run_chunks(state, end, advance, completed, sample, task=task)

            return self.program(*args, **kwargs, drive=drive)


def fixed_updates[S](
    state: S,
    end: int,
    advance: Callable[[S, int, int], S],
    drive: Drive[tuple[int, S]],
    sample: Callable[[S], dict[str, progress.Sample]],
    metrics: Callable[[S, int], Metrics] | None = None,
) -> S:
    """Report completed full sweeps with their absolute position in the original schedule."""

    def update(carry: tuple[int, S], stop: int) -> tuple[int, S]:
        first, state = carry
        return stop, advance(state, first, stop)

    def completed(carry: tuple[int, S]) -> tuple[int, bool, Metrics]:
        done, state = carry
        return int(done), False, {} if metrics is None else metrics(state, int(done))

    return drive((0, state), end, update, completed, lambda carry: sample(carry[1]))[1]
