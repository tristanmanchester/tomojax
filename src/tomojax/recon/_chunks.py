"""Drive one donated JAX loop from the operation's calling thread."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from contextvars import ContextVar
from functools import partial
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import jax
import numpy as np

from tomojax import progress
from tomojax.core import checkpoint

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


def restored[S](state: S, arrays: Sequence[np.ndarray]) -> S:
    """``state`` with the saved values of its leaves, on the same devices and of the same types."""
    leaves, tree = jax.tree.flatten(state)
    if len(leaves) != len(arrays):
        raise ValueError("the checkpoint's solver state does not match this solve")
    values = []
    for leaf, value in zip(leaves, arrays, strict=True):
        dtype = leaf.dtype if hasattr(leaf, "dtype") else np.result_type(leaf)
        if np.shape(leaf) != value.shape or dtype != value.dtype:
            raise ValueError("the checkpoint's solver state does not match this solve")
        if isinstance(leaf, jax.Array):
            values.append(jax.device_put(np.asarray(value), leaf.sharding))
        elif isinstance(leaf, np.ndarray):
            values.append(np.array(value))
        else:
            values.append(type(leaf)(value.item()))
    return jax.tree.unflatten(tree, values)


def run_chunks[S](
    state: S,
    end: int,
    advance: Callable[[S, int], S],
    completed: Callable[[S], tuple[int, bool, Metrics]],
    sample: Callable[[S], dict[str, progress.Sample]],
    *,
    task: IterationProgress | None,
    saves: checkpoint.Loop | None = None,
) -> S:
    """Advance ``state`` to ``end`` in chunks reported to ``task`` and saved by ``saves``.

    Without either, the whole loop is one call of ``advance``. A chunk lasts
    about a quarter of a second once compiled; its boundaries are where the
    state is reported and saved.
    """
    if task is None and saves is None:
        return advance(state, end)
    if saves is not None and saves.saved is not None:
        state = restored(state, saves.saved.arrays)
    _ = jax.block_until_ready(state)
    done, stopped, metrics = completed(state)
    if done and task is not None:
        task.advance(done, metrics)
        if partial_sweep.get() is None:
            progress.preview(lambda state=state: sample(state), final=stopped or done == end)
    length = 1
    compiled = False
    while done < end and not stopped:
        progress.check_canceled()
        start = monotonic()
        with progress.preparing("first update", int(compiled)):
            state = advance(state, min(end, done + length))
            _ = jax.block_until_ready(state)
        elapsed = monotonic() - start
        previous = done
        done, stopped, metrics = completed(state)
        if task is not None and not partial_sweep.get():
            task.advance(done, metrics)
            if partial_sweep.get() is None:
                progress.preview(lambda state=state: sample(state), final=stopped or done == end)
        if saves is not None and saves.due():
            saves.save(jax.tree.leaves(state))
        if compiled:
            length = max(1, round(0.25 * (done - previous) / max(elapsed, 1e-9)))
        compiled = True
    return state


class Chunked[R]:
    """Fuse all phases when unwatched. Watched or checkpointed solves materialize chunk state."""

    def __init__(self, program: Callable[..., R], *, static_argnames: tuple[str, ...]) -> None:
        self.program = program
        self.unwatched: Callable[..., R] = jax.jit(
            partial(program, drive=partial(run_chunks, task=None)), static_argnames=static_argnames
        )

    def __call__(self, *args: object, **kwargs: object) -> R:
        task = progress.for_unit("iterations")
        if task is None and not checkpoint.active():
            return self.unwatched(*args, **kwargs)
        progress.check_canceled()
        with ExitStack() as setup, checkpoint.loop() as saves:
            _ = setup.enter_context(progress.preparing("initialize solver"))

            def drive[S](
                state: S,
                end: int,
                advance: Callable[[S, int], S],
                completed: Callable[[S], tuple[int, bool, Metrics]],
                sample: Callable[[S], dict[str, progress.Sample]],
            ) -> S:
                setup.close()
                return run_chunks(state, end, advance, completed, sample, task=task, saves=saves)

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
