"""CLI: convert datasets between NPZ and HDF5/NXtomo.

Usage examples:

  uv run tomojax convert --in data/sim.npz --out data/sim.nxs
  uv run tomojax convert --in data/sim.nxs --out data/sim.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from tomojax.io import convert_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ConvertCommand:
    """Typed command plan for dataset conversion."""

    input_path: str
    output_path: str


def _build_parser() -> argparse.ArgumentParser:
    """Build the convert command parser."""
    p = argparse.ArgumentParser(description="Convert datasets between NPZ and HDF5/NXtomo (.nxs)")
    _ = p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input file (.npz or .nxs/.h5/.hdf5)",
    )
    _ = p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output file (.nxs/.h5/.hdf5 or .npz)",
    )
    return p


def _parse_command(argv: Sequence[str] | None) -> ConvertCommand:
    """Parse CLI arguments into a typed conversion command plan."""
    args = _build_parser().parse_args(argv)
    return ConvertCommand(
        input_path=cast("str", args.in_path),
        output_path=cast("str", args.out_path),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset conversion command."""
    command = _parse_command(argv)
    convert_dataset(command.input_path, command.output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
