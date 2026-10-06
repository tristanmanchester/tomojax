"""The ``tomojax`` command: dispatch to a subcommand."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import sys
from typing import TYPE_CHECKING, cast

from tomojax.cli.api import PRODUCT_COMMANDS

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

_MODULES = {c.name: c.name for c in PRODUCT_COMMANDS} | {"import": "import_"}
# Commands that run JAX on a GPU, and the allocator each wants (None: JAX's own).
_JAX_ALLOCATORS: dict[str, str | None] = {"recon": None, "align": "platform", "simulate": None}


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
        _build_parser().error(f"unknown command {command!r}")
    if command in _JAX_ALLOCATORS:
        from tomojax.cli._jax_allocator import configure_jax_allocator_defaults

        configure_jax_allocator_defaults(allocator=_JAX_ALLOCATORS[command])
    module = importlib.import_module(f"tomojax.cli.{_MODULES[command]}")
    run = cast("Callable[[], int | None]", module.main)
    with _temporary_argv([f"tomojax {command}", *tail]):
        try:
            return int(run() or 0)
        except _expected_errors() as exc:
            print(f"tomojax {command}: error: {exc}", file=sys.stderr)
            return 1


def _build_parser() -> argparse.ArgumentParser:
    width = max(len(c.name) for c in PRODUCT_COMMANDS)
    commands = "\n".join(f"  {c.name:<{width}}  {c.help}" for c in PRODUCT_COMMANDS)
    parser = argparse.ArgumentParser(
        prog="tomojax",
        usage="tomojax [--version] <command> [options]",
        description=(
            "Reconstruct and align tomography, laminography and lab cone-beam CT.\n\n"
            f"Commands:\n{commands}"
        ),
        epilog=(
            "Each command reads INPUT and writes -o OUTPUT, refusing to replace an existing "
            "output without --force; `tomojax <command> --help` describes it. Exit status: "
            "0 success, 1 failure, 2 usage error.\n\n"
            "A lab CT scan from start to finish:\n"
            "  tomojax import scan/scan.xtekct -o scan.nxs\n"
            "  tomojax inspect scan.nxs\n"
            "  tomojax align scan.nxs -o aligned.nxs --mode cor\n"
            "  tomojax recon aligned.nxs -o recon.nxs\n"
            "  tomojax inspect recon.nxs --preview previews\n"
            "  tomojax export recon.nxs -o slices/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument("--version", action="store_true", help="Print the version and exit")
    return parser


def _expected_errors() -> tuple[type[BaseException], ...]:
    from tomojax.alignment.api import CheckpointError

    return (OSError, ValueError, KeyError, CheckpointError)


@contextmanager
def _temporary_argv(argv: list[str]) -> Generator[None, None, None]:
    old_argv = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
