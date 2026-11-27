"""How a checkpointed Slurm job resumes after an early SIGTERM and after a hard kill.

    uv run python examples/gallery/checkpoint_timeline.py checkpoint-timeline.png

A schematic of the rules in the large-scans guide, not a measurement: periodic
saves at least 10 minutes apart; on SIGTERM the run finishes its step, saves
and exits with status 130, so the requeued job continues from that save; after
a hard kill no final save runs, and the requeued job repeats the work done
since the last periodic save.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import save, style
import matplotlib.pyplot as plt

RUN, LOST, SAVE = "#2a78d6", "#eb6834", "#0b0b0b"
INK, MUTED = "#0b0b0b", "#52514e"
LIMIT = 8  # the time limit of one job segment, in save intervals
QUEUE = 1.9  # time spent waiting in the queue after a requeue
HEIGHT = 0.34


def lane(ax: Any, y: float, stop: float, *, final_save: bool, note: str, resumed: str) -> None:
    """One scenario: a segment stopped at ``stop``, the queue, then the resumed segment."""
    saves = [float(t) for t in range(1, int(stop) + 1)] + ([stop] if final_save else [])
    lost = stop - saves[-1]
    ax.barh(y, stop, height=HEIGHT, color=RUN)
    ax.barh(y, lost, left=saves[-1], height=HEIGHT, color=LOST)
    for t in saves:
        ax.plot([t, t], [y - 0.25, y + 0.25], color=SAVE, lw=1.2)
    ax.text(stop, y - 0.3, note, ha="right", va="top", color=INK)
    restart = stop + QUEUE
    ax.annotate("", xy=(restart, y), xytext=(stop + 0.1, y),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})  # fmt: skip
    ax.text(stop + QUEUE / 2, y + 0.06, "requeued", ha="center", va="bottom", color=MUTED)
    ax.barh(y, lost, left=restart, height=HEIGHT, color=LOST)
    ax.barh(y, 3.2, left=restart + lost, height=HEIGHT, color=RUN)
    ax.text(restart, y - 0.3, resumed, ha="left", va="top", color=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    style()
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    lane(ax, 1.0, LIMIT - 0.3, final_save=True,
         note="SIGTERM: finish the step, save,\nexit with status 130",
         resumed="resumes from\nthe final save")  # fmt: skip
    lane(ax, -0.2, 5.45, final_save=False,
         note="hard kill: no final save", resumed="resumes from the last\nperiodic save and\n"
         "repeats the lost work")  # fmt: skip
    ax.plot([LIMIT, LIMIT], [1.22, 1.45], color=MUTED, lw=0.8, ls=":")
    ax.text(LIMIT, 1.45, "time limit", ha="center", va="bottom", color=MUTED)
    ax.annotate("", xy=(2, 1.3), xytext=(1, 1.3),
                arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 0.8})  # fmt: skip
    ax.text(1.5, 1.34, "≥ 10 min", ha="center", va="bottom", color=MUTED)
    ax.set_yticks([1.0, -0.2], ["Early SIGTERM\n(--signal=TERM@300)", "Node failure\nor SIGKILL"])
    ax.set_ylim(-1.15, 1.7)
    ax.set_xlim(0, LIMIT + QUEUE + 3.4)
    ax.set_xticks([])
    ax.set_xlabel("Wall-clock time · black ticks are saves · orange is work done twice",
                  color=MUTED)  # fmt: skip
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout()
    save(fig, args.output, palette=False)


if __name__ == "__main__":
    main()
