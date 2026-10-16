"""Load and align raw NeXus laminography without hand-written dataset paths.

Automatic alignment estimates missing setup, chooses a reconstruction region,
and validates motion on internal held projections. Its returned diagnostics
explain unresolved support or geometry; consistency is not physical truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tomojax as tj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="Raw HDF5/NeXus file")
    parser.add_argument("-o", "--output", type=Path, help="Save alignment and volume (.nxs)")
    args = parser.parse_args()
    scan = tj.load_frames(args.input).corrected()
    result = tj.align(scan)
    print(result.scan)
    print("Automatic outcome:", result.info.get("automatic"))
    if args.output is not None:
        tj.save(args.output, result)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
