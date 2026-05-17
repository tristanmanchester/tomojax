"""Public API for command-line orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CliCommand:
    """Command exposed through the grouped `tomojax` dispatcher."""

    name: str
    help: str


PRODUCT_COMMANDS: tuple[CliCommand, ...] = (
    CliCommand("inspect", "Inspect a projection dataset."),
    CliCommand("validate", "Validate a projection or reconstruction dataset."),
    CliCommand("preprocess", "Apply flat/dark/background preprocessing."),
    CliCommand("ingest", "Build a standard dataset from raw inputs."),
    CliCommand("convert", "Convert supported dataset formats."),
    CliCommand("recon", "Reconstruct a volume from a dataset."),
    CliCommand("align", "Run product alignment and reconstruction."),
    CliCommand("simulate", "Generate deterministic synthetic datasets."),
)

DEVELOPER_COMMANDS: tuple[CliCommand, ...] = (
    CliCommand("loss-bench", "Compare projection-domain loss functions."),
    CliCommand("misalign", "Generate controlled diagnostic misalignment datasets."),
    CliCommand("align-auto", "Run the staged synthetic alignment benchmark."),
    CliCommand("astra-parallel-bench", "Compare ASTRA and TomoJAX parallel projectors."),
    CliCommand("benchmark-suite", "Run benchmark-suite probes."),
    CliCommand("alignment-diagnostic-bench", "Run alignment diagnostic benchmarks."),
    CliCommand("pallas-sanity", "Run Pallas backend sanity checks."),
    CliCommand("synthetic-benchmark-compare", "Compare synthetic benchmark artifacts."),
    CliCommand("current-baseline-normalize", "Normalize current baseline artifacts."),
    CliCommand("test-gpu", "Check JAX GPU runtime availability."),
    CliCommand("test-cpu", "Check JAX CPU runtime availability."),
)


def product_command_names() -> tuple[str, ...]:
    """Return product-facing grouped command names."""
    return tuple(command.name for command in PRODUCT_COMMANDS)


def developer_command_names() -> tuple[str, ...]:
    """Return developer diagnostic grouped command names."""
    return tuple(command.name for command in DEVELOPER_COMMANDS)


__all__ = [
    "DEVELOPER_COMMANDS",
    "PRODUCT_COMMANDS",
    "CliCommand",
    "developer_command_names",
    "product_command_names",
]
