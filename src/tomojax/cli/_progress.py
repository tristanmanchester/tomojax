"""Progress display and interrupt handling shared by long-running commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import signal
import sys
from types import FrameType

from tomojax import progress


def add_progress(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "json", "none"),
        default="auto",
        help="Show progress on stderr or write JSON Lines to stdout (auto uses a terminal bar)",
    )


class _Bar:
    def __init__(self) -> None:
        self.path: dict[int, str] = {}

    def __call__(self, event: progress.Event) -> None:
        detail = ""
        if isinstance(event, progress.Started):
            self.path[event.task] = event.label
        elif isinstance(event, progress.Advanced):
            count = str(event.done) if event.total is None else f"{event.done}/{event.total}"
            metric = next(iter(event.metrics.items()), None)
            detail = f"  {count}"
            if metric is not None:
                detail += f"  {metric[0]}={metric[1]}"
        else:
            del self.path[event.task]
        line = " > ".join(self.path.values())
        if line:
            line += f"{detail}  {event.time:.1f}s"
        print(f"\r\033[2K{line}", end="", file=sys.stderr, flush=True)


def run_command(command: str, mode: str, operation: Callable[[], None]) -> int:
    """Run the command under one watch, restoring the caller's SIGINT policy."""
    token = progress.Cancel()

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        if token.cancelled:
            raise KeyboardInterrupt
        token.cancel()

    def deliver(event: progress.Event) -> None:
        if mode == "json":
            print(json.dumps(progress.to_json(event), allow_nan=False), flush=True)

    handler = _Bar() if mode == "bar" or (mode == "auto" and sys.stderr.isatty()) else deliver
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        with progress.watch(handler, cancel=token):
            operation()
    except progress.Cancelled:
        return 130
    except (OSError, ValueError, KeyError) as exc:
        if mode == "json":
            print(json.dumps(progress.error_record(exc)), flush=True)
        else:
            print(f"tomojax {command}: error: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
    return 0
