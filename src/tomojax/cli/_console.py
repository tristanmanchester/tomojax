"""Human output of the ``tomojax`` command: errors with hints, suggestions and tracebacks.

Display goes to stderr and data to stdout. Rich decides colour from the terminal
(``NO_COLOR``, ``FORCE_COLOR``, ``TERM=dumb``, not a TTY). Text is never wrapped or
cut, and user-supplied text is never read as markup.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

ISSUES_URL = "https://github.com/tristanmanchester/tomojax/issues"

# The command forms of `tomojax --help`, as each command's one-line usage.
USAGE = {
    "inspect": "INPUT [options]",
    "correct": "INPUT -o OUTPUT [options]",
    "reconstruct": "INPUT -o OUTPUT [options]",
    "align": "INPUT -o OUTPUT [options]",
    "export": "INPUT -o OUTPUT [options]",
    "simulate": "-o OUTPUT [options]",
    "": "<command> [options]",
}

# Frames from these packages are collapsed in an internal error's traceback.
_LIBRARIES = ("jax", "jaxlib", "numpy", "msgspec", "h5py")


def console(*, stderr: bool = True) -> Console:
    """A console on the current stderr (or stdout); made per use, so redirection applies."""
    return Console(stderr=stderr, markup=False, highlight=False, emoji=False, soft_wrap=True)


def suggest(word: str, candidates: Iterable[str]) -> str | None:
    """The candidate closest to a mistyped ``word``, if one is close."""
    matches = difflib.get_close_matches(word, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def usage_line(prog: str) -> str:
    """The one-line usage of ``prog`` (``tomojax <command>``)."""
    command = prog.removeprefix("tomojax").strip()
    form = USAGE.get(command, "[options]")
    return f"{prog} {form}"


def report_error(prog: str, message: str, *, hint: str | None = None) -> None:
    """Print ``prog: error: message`` and an optional hint on stderr."""
    out = console()
    out.print(Text.assemble((f"{prog}: ", "bold"), ("error: ", "bold red"), message))
    if hint:
        out.print(Text.assemble("  ", ("hint: ", "cyan"), hint))


def report_usage_error(prog: str, message: str, *, hint: str | None = None) -> None:
    """Print a usage error, its hint and the command's one-line usage on stderr."""
    report_error(prog, message, hint=hint)
    command = prog.split()[-1]
    help_command = prog if command in USAGE else "tomojax"
    console().print(
        Text.assemble(
            ("usage: ", "dim"),
            (usage_line(prog), "dim"),
            ("   (", "dim"),
            (f"{help_command} --help", "dim"),
            (")", "dim"),
        )
    )


def report_cancelled(prog: str) -> None:
    """Say on stderr that the command stopped at the user's request."""
    console().print(Text(f"{prog}: cancelled", style="dim"))


def report_warning(message: str, *, level: str = "warning") -> None:
    """Print a warning or another note level on stderr."""
    style = "yellow" if level == "warning" else "bold red" if level in ("error", "critical") else ""
    if level == "info":
        console().print(Text(message, style="dim"))
        return
    console().print(Text.assemble((f"{level}: ", style), message))


def report_internal(prog: str, error: BaseException) -> None:
    """Print a short traceback of a TomoJAX bug, and where to report it, on stderr."""
    from rich.traceback import Traceback

    suppress: list[str | ModuleType] = [
        module for name in _LIBRARIES if (module := sys.modules.get(name)) is not None
    ]
    # The command plumbing every traceback passes through.
    here = Path(__file__).resolve().parent
    suppress += [str(here / "main.py"), str(here / "_progress.py")]
    out = console()
    if not out.is_terminal:  # a log or a pipe keeps every file name and line in full
        import traceback

        out.print("".join(traceback.format_exception(error)).rstrip())
    else:
        out.print(
            Traceback.from_exception(
                type(error),
                error,
                error.__traceback__,
                extra_lines=0,
                show_locals=False,
                suppress=suppress,
                max_frames=12,
            )
        )
    from tomojax import __version__

    out.print(
        Text.assemble(
            (f"{prog}: ", "bold"),
            ("internal error: ", "bold red"),
            f"this is a bug in TomoJAX {__version__}. Please report it at {ISSUES_URL} "
            "with the command you ran.",
        )
    )


__all__ = [
    "ISSUES_URL",
    "USAGE",
    "console",
    "report_cancelled",
    "report_error",
    "report_internal",
    "report_usage_error",
    "report_warning",
    "suggest",
    "usage_line",
]
