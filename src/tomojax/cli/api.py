"""Public API for command-line orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CliCommand:
    """Command exposed through the grouped `tomojax` dispatcher."""

    name: str
    help: str


PRODUCT_COMMANDS: tuple[CliCommand, ...] = (
    CliCommand("inspect", "Describe and check a dataset; preview it as PNGs."),
    CliCommand("import", "Make a dataset from a Nikon scan, TIFF stack or .npz."),
    CliCommand("preprocess", "Flat- and dark-correct raw frames."),
    CliCommand("recon", "Reconstruct a volume."),
    CliCommand("align", "Estimate the rotation axis and per-view poses."),
    CliCommand("export", "Write a reconstruction as TIFF slices or a raw file."),
    CliCommand("simulate", "Write a synthetic scan of a phantom."),
)


def product_command_names() -> tuple[str, ...]:
    """Return product-facing grouped command names."""
    return tuple(command.name for command in PRODUCT_COMMANDS)


__all__ = [
    "PRODUCT_COMMANDS",
    "CliCommand",
    "product_command_names",
]
