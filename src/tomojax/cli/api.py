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
    CliCommand("slices", "Extract labelled reconstruction slice PNGs."),
    CliCommand("align", "Run product alignment and reconstruction."),
    CliCommand("simulate", "Generate deterministic synthetic datasets."),
)


def product_command_names() -> tuple[str, ...]:
    """Return product-facing grouped command names."""
    return tuple(command.name for command in PRODUCT_COMMANDS)


__all__ = [
    "PRODUCT_COMMANDS",
    "CliCommand",
    "product_command_names",
]
