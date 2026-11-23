"""Drive one donated JAX loop from the operation's calling thread."""

from collections.abc import Callable
from functools import partial
from time import monotonic
from typing import Any

import jax

from tomojax import progress


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
        task = progress.current_task.get()
        if task is None:
            return self.unwatched(*args, **kwargs)
        return self.program(*args, **kwargs, drive=partial(run_chunks, task=task))
