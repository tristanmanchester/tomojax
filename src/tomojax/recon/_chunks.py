"""Drive one donated JAX loop from the operation's calling thread."""

from collections.abc import Callable
from contextvars import ContextVar
from functools import partial
from time import monotonic
from typing import Any

import jax

from tomojax import progress

# Until the last independent axial partition, no whole-volume sweep is finished.
partial_sweep = ContextVar("tomojax_partial_sweep", default=False)


def run_chunks[S](
    state: S,
    end: int,
    advance: Callable[[S, int], S],
    completed: Callable[[S], tuple[int, bool, dict[str, float]]],
    *,
    task,
) -> S:
    if task is None:
        return advance(state, end)
    jax.block_until_ready(state)
    done, stopped, metrics = completed(state)
    if done:
        task.advance(done, metrics)
    length = 1
    compiled = False
    while done < end and not stopped:
        progress.check_cancelled()
        start = monotonic()
        state = jax.block_until_ready(advance(state, min(end, done + length)))
        elapsed = monotonic() - start
        previous = done
        done, stopped, metrics = completed(state)
        if not partial_sweep.get():
            task.advance(done, metrics)
        if compiled:
            length = max(1, round(0.25 * (done - previous) / max(elapsed, 1e-9)))
        compiled = True
    return state


class Chunked[R]:
    """Fuse all phases when unwatched. Only watched solves materialise chunk state."""

    def __init__(self, program: Callable[..., R], *, static_argnames: tuple[str, ...]) -> None:
        self.program = program
        self.unwatched = jax.jit(
            partial(program, drive=partial(run_chunks, task=None)), static_argnames=static_argnames
        )

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        task = progress.for_unit("iterations")
        if task is None:
            return self.unwatched(*args, **kwargs)
        progress.check_cancelled()
        return self.program(*args, **kwargs, drive=partial(run_chunks, task=task))


def fixed_updates(state, end, advance, drive, metrics=None):
    """Report completed full sweeps with their absolute position in the original schedule."""

    def update(carry, stop):
        first, state = carry
        return stop, advance(state, first, stop)

    def completed(carry):
        done, state = carry
        return int(done), False, {} if metrics is None else metrics(state, int(done))

    return drive((0, state), end, update, completed)[1]
