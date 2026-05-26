"""Top-level TomoJAX command dispatcher."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
from typing import TYPE_CHECKING

from tomojax.cli.api import product_command_names
from tomojax.cli._jax_allocator import configure_jax_allocator_defaults

configure_jax_allocator_defaults()

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911
    """Run the production-facing `tomojax` command."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _build_parser().print_help()
        return 0

    command, *tail = args
    if command == "inspect":
        from tomojax.cli import inspect

        return _run_positional_cli(inspect.main, "tomojax inspect", tail)
    if command == "validate":
        from tomojax.cli import validate

        return _run_positional_cli(validate.main, "tomojax validate", tail)
    if command == "preprocess":
        from tomojax.cli import preprocess

        return _run_positional_cli(preprocess.main, "tomojax preprocess", tail)
    if command == "ingest":
        from tomojax.cli import ingest

        return _run_positional_cli(ingest.main, "tomojax ingest", tail)
    if command == "convert":
        from tomojax.cli import convert

        return _run_sysargv_cli(convert.main, "tomojax convert", tail)
    if command == "recon":
        from tomojax.cli import recon

        return _run_sysargv_cli(recon.main, "tomojax recon", tail)
    if command == "align":
        from tomojax.cli import align

        return _run_sysargv_cli(align.main, "tomojax align", tail)
    if command == "simulate":
        from tomojax.cli import simulate

        return _run_sysargv_cli(simulate.main, "tomojax simulate", tail)
    parser = _build_parser()
    parser.error(f"unknown command {command!r}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax",
        description="TomoJAX tomography and laminography reconstruction toolbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "command",
        nargs="?",
        choices=product_command_names(),
        help="Command to run.",
    )
    parser.epilog = (
        "Examples:\n"
        "  tomojax inspect scan.nxs\n"
        "  tomojax ingest ./projections --angles angles.csv --du 0.65 --dv 0.65 --out scan.nxs\n"
        "  tomojax preprocess raw.nxs corrected.nxs\n"
        "  tomojax recon --data corrected.nxs --out recon.nxs\n"
        "  tomojax align --data corrected.nxs --out aligned.nxs --mode cor\n"
    )
    return parser


def _run_positional_cli(
    command: Callable[[Sequence[str] | None], int | None],
    prog: str,
    argv: list[str],
) -> int:
    with _temporary_argv([prog, *argv]):
        return int(command(argv) or 0)


def _run_sysargv_cli(command: Callable[[], object], prog: str, argv: list[str]) -> int:
    with _temporary_argv([prog, *argv]):
        _ = command()
    return 0


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
