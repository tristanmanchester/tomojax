"""Top-level TomoJAX command dispatcher."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
from typing import TYPE_CHECKING

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

        return _run_sysargv_cli(recon.main, "tomojax recon", _with_data_alias(tail))
    if command == "align":
        from tomojax.cli import align

        return _run_sysargv_cli(align.main, "tomojax align", _with_data_alias(tail))
    if command == "simulate":
        from tomojax.cli import simulate

        return _run_sysargv_cli(simulate.main, "tomojax simulate", tail)
    if command == "dev":
        return _run_dev_command(tail)

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
        choices=[
            "inspect",
            "validate",
            "preprocess",
            "ingest",
            "convert",
            "recon",
            "align",
            "simulate",
            "dev",
        ],
        help="Command to run.",
    )
    parser.epilog = (
        "Examples:\n"
        "  tomojax inspect scan.nxs\n"
        "  tomojax ingest ./projections --angles angles.csv --du 0.65 --dv 0.65 --out scan.nxs\n"
        "  tomojax preprocess raw.nxs corrected.nxs --log\n"
        "  tomojax recon corrected.nxs --out recon.nxs\n"
        "  tomojax align corrected.nxs --out aligned.nxs --schedule cor\n"
        "\n"
        "Developer diagnostics and benchmark probes are grouped under tomojax dev."
    )
    return parser


def _run_dev_command(argv: list[str]) -> int:  # noqa: PLR0911
    if not argv or argv[0] in {"-h", "--help"}:
        _dev_parser().print_help()
        return 0

    command, *tail = argv
    if command == "loss-bench":
        from tomojax.cli import loss_bench

        return _run_sysargv_cli(loss_bench.main, "tomojax dev loss-bench", tail)
    if command == "misalign":
        from tomojax.cli import misalign

        return _run_sysargv_cli(misalign.main, "tomojax dev misalign", tail)
    if command == "align-auto":
        from tomojax.cli import align_auto

        return _run_positional_cli(align_auto.main, "tomojax dev align-auto", tail)
    if command == "astra-parallel-bench":
        from tomojax.bench import astra_parallel

        return _run_sysargv_cli(astra_parallel.main, "tomojax dev astra-parallel-bench", tail)
    if command == "benchmark-suite":
        from tomojax.bench import benchmark_suite

        return _run_sysargv_cli(benchmark_suite.main, "tomojax dev benchmark-suite", tail)
    if command == "alignment-diagnostic-bench":
        from tomojax.bench import alignment_smoke

        return _run_sysargv_cli(
            alignment_smoke.main, "tomojax dev alignment-diagnostic-bench", tail
        )
    if command == "pallas-sanity":
        from tomojax.bench import pallas_sanity

        return _run_sysargv_cli(pallas_sanity.main, "tomojax dev pallas-sanity", tail)
    if command == "synthetic-benchmark-compare":
        from tomojax.bench import synthetic_results

        return _run_positional_cli(
            synthetic_results.main,
            "tomojax dev synthetic-benchmark-compare",
            tail,
        )
    if command == "current-baseline-normalize":
        from tomojax.bench import current_baseline

        return _run_positional_cli(
            current_baseline.main,
            "tomojax dev current-baseline-normalize",
            tail,
        )
    if command == "test-gpu":
        from tomojax.cli.runtime_checks import test_gpu_main

        return _run_sysargv_cli(test_gpu_main, "tomojax dev test-gpu", tail)
    if command == "test-cpu":
        from tomojax.cli.runtime_checks import test_cpu_main

        return _run_sysargv_cli(test_cpu_main, "tomojax dev test-cpu", tail)

    parser = _dev_parser()
    parser.error(f"unknown dev command {command!r}")
    return 2


def _dev_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tomojax dev",
        description="Developer diagnostics and benchmark probes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ = parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "loss-bench",
            "misalign",
            "align-auto",
            "astra-parallel-bench",
            "benchmark-suite",
            "alignment-diagnostic-bench",
            "pallas-sanity",
            "synthetic-benchmark-compare",
            "current-baseline-normalize",
            "test-gpu",
            "test-cpu",
        ],
    )
    return parser


def _with_data_alias(argv: list[str]) -> list[str]:
    """Allow `tomojax recon scan.nxs --out ...` as shorthand for `--data scan.nxs`."""
    if not argv:
        return argv
    first = argv[0]
    if first.startswith("-") or "--data" in argv:
        return argv
    return ["--data", first, *argv[1:]]


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
