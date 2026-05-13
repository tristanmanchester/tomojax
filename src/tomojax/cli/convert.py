"""CLI: convert datasets between NPZ and HDF5/NXtomo.

Usage examples:

  uv run tomojax convert --in data/sim.npz --out data/sim.nxs
  uv run tomojax convert --in data/sim.nxs --out data/sim.npz
"""

from __future__ import annotations

import argparse

from tomojax.io import convert_dataset


def main() -> None:
    """Run the dataset conversion command."""
    p = argparse.ArgumentParser(description="Convert datasets between NPZ and HDF5/NXtomo (.nxs)")
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input file (.npz or .nxs/.h5/.hdf5)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output file (.nxs/.h5/.hdf5 or .npz)",
    )
    args = p.parse_args()
    convert_dataset(args.in_path, args.out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
