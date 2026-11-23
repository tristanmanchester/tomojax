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
    CliCommand("correct", "Import raw scans or apply corrections, selection, crop and binning."),
    CliCommand("reconstruct", "Reconstruct a volume."),
    CliCommand("align", "Fit a scan's geometry and per-view motion."),
    CliCommand("export", "Write a reconstruction as TIFF slices or a raw file."),
    CliCommand("simulate", "Write a synthetic scan of a phantom."),
    CliCommand("serve", "Serve a GUI session over JSON Lines."),
)


def product_command_names() -> tuple[str, ...]:
    """Return product-facing grouped command names."""
    return tuple(command.name for command in PRODUCT_COMMANDS)


__all__ = [
    "PRODUCT_COMMANDS",
    "CliCommand",
    "product_command_names",
]
