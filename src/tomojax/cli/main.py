"""The ``tomojax`` command: dispatch to a subcommand."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import sys
from typing import TYPE_CHECKING, cast

from tomojax.cli import _console
from tomojax.cli.api import PRODUCT_COMMANDS

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from tomojax.cli._progress import ProgressParser

_MODULES = {c.name: c.name for c in PRODUCT_COMMANDS}


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``tomojax <command> ...``; return the exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _build_parser().print_help()
        return 0
    if args[0] == "--version":
        from tomojax import __version__

        print(f"tomojax {__version__}")
        return 0
    command, *tail = args
    if command not in _MODULES:
        close = _console.suggest(command, _MODULES)
        hint = f"did you mean {close}?" if close else "tomojax --help lists the commands"
        _build_parser().fail(f"unknown command {command!r}", hint=hint)
    module = importlib.import_module(f"tomojax.cli.{_MODULES[command]}")
    run_module = cast("Callable[[], int | None]", module.main)
    with _temporary_argv([f"tomojax {command}", *tail]):
        try:
            return int(run_module() or 0)
        except (OSError, ValueError, KeyError) as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and len(exc.args) == 1 else exc
            _console.report_error(f"tomojax {command}", str(message))
            return 1


def run() -> int:
    """The ``tomojax`` console script: :func:`main`, with a TomoJAX bug reported briefly."""
    args = sys.argv[1:]
    prog = f"tomojax {args[0]}" if args and args[0] in _MODULES else "tomojax"
    try:
        return main()
    except KeyboardInterrupt:
        _console.report_cancelled(prog)
        return 130
    except Exception as exc:
        _console.report_internal(prog, exc)
        return 1


def _build_parser() -> ProgressParser:
    from tomojax.cli._progress import ProgressParser

    width = max(len(c.name) for c in PRODUCT_COMMANDS)
    commands = "\n".join(f"  {c.name:<{width}}  {c.help}" for c in PRODUCT_COMMANDS)
    parser = ProgressParser(
        prog="tomojax",
        usage="tomojax [--version] <command> [options]",
        description=(
            "Reconstruct and align tomography, laminography and lab cone-beam CT.\n\n"
            f"Commands:\n{commands}"
        ),
        epilog=(
            "Command forms:\n"
            "  inspect INPUT [--preview-dir DIR] [--json]\n"
            "  correct | align | reconstruct | export  INPUT -o OUTPUT \\[options]\n"
            "  simulate -o OUTPUT \\[options]\n  serve | mcp\n\n"
            "Commands refuse to replace an existing output without --force; "
            "`tomojax <command> --help` describes each one. Exit status: "
            "0 success, 1 failure, 2 usage error, 130 interrupted.\n\n"
            "A lab CT scan from start to finish:\n"
            "  tomojax correct scan/scan.xtekct -o scan.nxs --darks 0\n"
            "  tomojax inspect scan.nxs\n"
            "  tomojax align scan.nxs -o aligned.nxs --fit center roll\n"
            "  tomojax reconstruct aligned.nxs -o recon.nxs\n"
            "  tomojax inspect recon.nxs --preview-dir previews\n"
            "  tomojax export recon.nxs -o slices/"
        ),
    )
    _ = parser.add_argument("--version", action="store_true", help="Print the version and exit")
    return parser


@contextmanager
def _temporary_argv(argv: list[str]) -> Generator[None, None, None]:
    old_argv = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(run())
