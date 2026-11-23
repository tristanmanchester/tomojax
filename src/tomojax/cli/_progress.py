"""Progress display and interrupt handling shared by long-running commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import signal
import sys
from types import FrameType
from typing import NoReturn

import msgspec

from tomojax import progress
from tomojax.wire.progress import Error, Result


class ProgressParser(argparse.ArgumentParser):
    """Render usage failures in the requested output mode as well as on stderr."""

    invocation: list[str] | None = None

    @property
    def json_output(self) -> bool:
        args = self.invocation if self.invocation is not None else sys.argv[1:]
        mode = self.get_default("progress")
        for index, option in enumerate(args):
            if option == "--":
                break
            if option == "--progress" and index + 1 < len(args):
                mode = args[index + 1]
            elif option.startswith("--progress="):
                mode = option.removeprefix("--progress=")
        return mode == "json"

    def set_invocation(self, argv: list[str]) -> None:
        self.invocation = argv
        if self.json_output and any(flag in argv for flag in ("--help", "-h", "--config-keys")):
            self.error("help and configuration listings cannot be combined with --progress json")

    def error(self, message: str) -> NoReturn:
        if self.json_output:
            print(msgspec.json.encode(Error(2, "invalid_parameter", message)).decode(), flush=True)
        super().error(message)


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
        self.counts: dict[int, str] = {}

    def __call__(self, event: progress.Event) -> None:
        if isinstance(event, progress.Started):
            self.path[event.task] = event.label
            self.counts[event.task] = ""
        elif isinstance(event, progress.Advanced):
            count = str(event.done) if event.total is None else f"{event.done}/{event.total}"
            metric = next(iter(event.metrics.items()), None)
            detail = f"  {count}"
            if metric is not None:
                detail += f"  {metric[0]}={metric[1]}"
            self.counts[event.task] = detail
        elif isinstance(event, progress.Finished):
            del self.path[event.task], self.counts[event.task]
        line = " > ".join(self.path.values())
        if line:
            line += f"{self.counts[next(reversed(self.path))]}  {event.time:.1f}s"
        print(f"\r\033[2K{line}", end="", file=sys.stderr, flush=True)


def _handler(mode: str) -> Callable[[progress.Event], None]:
    if mode == "bar" or (mode == "auto" and sys.stderr.isatty()):
        return _Bar()
    if mode == "json":
        return lambda e: print(json.dumps(progress.to_json(e), allow_nan=False), flush=True)
    return lambda _e: None


def _write_result(result: Result | int | None, command: str) -> None:
    if not isinstance(result, Result):
        raise RuntimeError(f"{command} did not return a result")
    print(msgspec.json.encode(result).decode(), flush=True)


def run_command(command: str, mode: str, operation: Callable[[], Result | int | None]) -> int:
    """Run the command under one watch, restoring the caller's SIGINT policy."""
    token = progress.Control()

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        if token.cancelled:
            raise KeyboardInterrupt
        token.cancel()

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        with progress.watch(_handler(mode), control=token):
            result = operation()
            if mode == "json":
                _write_result(result, command)
            elif isinstance(result, int):
                return result
    except Exception as exc:
        record = progress.error_record(exc)
        if mode == "json":
            print(json.dumps(record), flush=True)
        elif record["kind"] != "cancelled":
            print(f"tomojax {command}: error: {exc}", file=sys.stderr)
        if record["kind"] == "internal":
            raise
        return 130 if record["kind"] == "cancelled" else 1
    finally:
        signal.signal(signal.SIGINT, previous)
    return 0
