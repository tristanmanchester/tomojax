from __future__ import annotations

from typing import Callable, Iterable


LossCallback = Callable[[int, float], None]


def emit_loss_callback_endpoints(
    callback: LossCallback | None,
    step_loss_pairs: Iterable[tuple[int, float]],
) -> None:
    """Emit first/last recorded loss samples without duplicating the same step."""
    if callback is None:
        return

    seen_steps: set[int] = set()
    for step, loss in step_loss_pairs:
        step_i = int(step)
        if step_i in seen_steps:
            continue
        seen_steps.add(step_i)
        callback(step_i, float(loss))
