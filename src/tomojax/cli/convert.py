"""CLI: convert datasets between NPZ and HDF5/NXtomo.

Usage examples:

  pixi run python -m tomojax.cli.convert --in data/sim.npz --out data/sim.nxs
  pixi run python -m tomojax.cli.convert --in data/sim.nxs --out data/sim.npz
"""

from __future__ import annotations

import argparse
from ..data.io_hdf5 import convert


def main() -> None:
    p = argparse.ArgumentParser(description="Convert datasets between NPZ and HDF5/NXtomo (.nxs)")
    p.add_argument("--in", dest="in_path", required=True, help="Input file (.npz or .nxs/.h5/.hdf5)")
    p.add_argument("--out", dest="out_path", required=True, help="Output file (.nxs/.h5/.hdf5 or .npz)")
    args = p.parse_args()
    convert(args.in_path, args.out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
