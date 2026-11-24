"""Progress display, usage errors and interrupt handling shared by the commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
import json
import logging
from pathlib import Path
import signal
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any, NoReturn
import warnings

import msgspec

from tomojax import progress
from tomojax.cli import _console
from tomojax.cli._display import Display
from tomojax.wire.progress import Error, Result

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomojax.cli._watch import WatchDirectory

_PACKAGE = Path(progress.__file__).resolve().parent


class ProgressParser(argparse.ArgumentParser):
    """A command's parser: short usage errors with hints, in JSON too when asked.

    A usage error prints the message, a hint where the fix is known (a close match
    for a mistyped choice or option, ``--force`` for an existing output) and the
    command's one-line usage. With ``--progress json`` it also writes the error as a
    JSON record on stdout, whose message never carries the hint.
    """

    invocation: list[str] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from rich_argparse import RawDescriptionRichHelpFormatter

        kwargs["formatter_class"] = RawDescriptionRichHelpFormatter
        prog = kwargs.get("prog")
        if isinstance(prog, str) and "usage" not in kwargs:
            kwargs["usage"] = _console.usage_line(prog)
        super().__init__(*args, **kwargs)
        self._hint: str | None = None
        self._display: str | None = None

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

    def _check_value(self, action: argparse.Action, value: Any) -> None:
        choices = action.choices
        if choices is not None and value not in choices:
            close = _console.suggest(str(value), [str(choice) for choice in choices])
            if close is not None:
                self._hint = f"did you mean {close}?"
        super()._check_value(action, value)

    def fail(
        self, message: str, *, hint: str | None = None, display: str | None = None
    ) -> NoReturn:
        """A usage error with a hint; ``display`` replaces ``message`` on the terminal only."""
        self._hint, self._display = hint, display
        self.error(message)

    def error(self, message: str) -> NoReturn:
        unrecognized = message.removeprefix("unrecognized arguments: ")
        if unrecognized != message and self._hint is None:  # suggest the option meant
            known = [s for action in self._actions for s in action.option_strings]
            for extra in unrecognized.split():
                close = _console.suggest(extra.split("=", 1)[0], known)
                if extra.startswith("--") and close is not None:
                    self._hint = f"did you mean {close}?"
                    break
        if self.json_output:
            print(msgspec.json.encode(Error(2, "invalid_parameter", message)).decode(), flush=True)
        hint, display = self._hint, self._display
        self._hint = self._display = None
        _console.report_usage_error(self.prog, display or message, hint=hint)
        self.exit(2)


def add_progress(parser: argparse.ArgumentParser, *, watch_dir: bool = False) -> None:
    """Add ``--progress`` and, for a command with previews, ``--watch-dir``."""
    if watch_dir:
        _ = parser.add_argument(
            "--watch-dir",
            metavar="DIR",
            help="Rewrite PNGs of the central planes in DIR at each checkpoint of the solve",
        )
    parser.add_argument(
        "--progress",
        choices=("auto", "bar", "json", "none"),
        default="auto",
        help="Show progress on stderr or write JSON Lines to stdout (auto uses a terminal bar)",
    )


def _handler(mode: str, token: progress.Control) -> tuple[Callable[[progress.Event], None], Any]:
    """The watch handler for ``mode``, and the context that runs its display."""
    if mode == "json":

        def printer(event: progress.Event) -> None:
            print(json.dumps(progress.to_json(event), allow_nan=False), flush=True)

        return printer, nullcontext()
    if mode == "bar" or (mode == "auto" and sys.stderr.isatty()):
        display = Display(token, live=sys.stderr.isatty())
        return display, display

    def quiet(event: progress.Event) -> None:
        """Without a display, notes of warning level and above still reach stderr."""
        if isinstance(event, progress.Note) and event.level not in ("debug", "info"):
            _console.report_warning(event.message, level=event.level)

    return quiet, nullcontext()


class _ForeignRecords(logging.Handler):
    """Log records from outside TomoJAX (TomoJAX's own arrive as watch notes)."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "tomojax" or record.name.startswith("tomojax."):
            return
        try:
            message = record.getMessage()
        except Exception:
            self.handleError(record)
            return
        _console.report_warning(message, level=record.levelname.lower())


@contextmanager
def _owned_messages(mode: str) -> Iterator[None]:
    """For one command, show each log record and warning once, then restore the settings.

    TomoJAX's log records and warnings become watch notes, which the display shows
    (``--progress json`` writes them as records). Records and warnings from other
    packages go to stderr.
    """
    package = logging.getLogger("tomojax")
    level = package.level
    package.setLevel(logging.INFO)
    root = logging.getLogger()
    foreign = _ForeignRecords(logging.WARNING)
    root.addHandler(foreign)
    shown = warnings.showwarning

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        if Path(filename).resolve().is_relative_to(_PACKAGE):
            return  # issued through progress.warn, which delivered it as a note
        if mode == "json":
            shown(message, category, filename, lineno, file, line)
            return
        _console.report_warning(f"{message} ({category.__name__})")

    warnings.showwarning = show
    try:
        yield
    finally:
        warnings.showwarning = shown
        root.removeHandler(foreign)
        package.setLevel(level)


def _write_result(result: Result | int | None, command: str) -> None:
    if not isinstance(result, Result):
        raise RuntimeError(f"{command} did not return a result")
    print(msgspec.json.encode(result).decode(), flush=True)


def _failed(error: Exception, mode: str, prog: str) -> int:
    """Report a command's failure; return its exit status, or re-raise a TomoJAX bug."""
    record = progress.error_record(error)
    kind = record["kind"]
    if mode == "json":
        print(json.dumps(record), flush=True)
    if kind == "internal":
        raise error
    if kind == "cancelled":
        _console.report_cancelled(prog)
        return 130
    # A KeyError's message without the quotes str() adds.
    unquoted = isinstance(error, KeyError) and len(error.args) == 1
    message, path = str(error.args[0] if unquoted else error), record.get("path")
    named = isinstance(path, str) and path and path not in message
    _console.report_error(prog, message, hint=f"setting: {path}" if named else None)
    return 1


def run_command(
    command: str,
    mode: str,
    operation: Callable[[], Result | int | None],
    *,
    watch: WatchDirectory | None = None,
) -> int:
    """Run the command under one watch, restoring the caller's SIGINT policy.

    With ``watch``, preview checkpoints go to its images and not to the display or
    the JSON stream.
    """
    token = progress.Control()
    prog = f"tomojax {command}"

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        if token.cancelled:
            raise KeyboardInterrupt
        token.cancel()

    previous = signal.signal(signal.SIGINT, interrupt)
    shown, display = _handler(mode, token)

    def handler(event: progress.Event) -> None:
        if watch is not None:
            watch.observe(event)
            if isinstance(event, progress.Preview):
                return
        shown(event)

    try:
        with (
            display,
            watch or nullcontext(),
            _owned_messages(mode),
            progress.watch(handler, control=token),
        ):
            result = operation()
            if mode == "json":
                _write_result(result, command)
            elif isinstance(result, int):
                return result
    except Exception as exc:
        return _failed(exc, mode, prog)
    finally:
        signal.signal(signal.SIGINT, previous)
    return 0
