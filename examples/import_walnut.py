"""Import one FIPS cone-beam walnut orbit as a small scan, without ASTRA or CUDA.

Run from the checkout root; the result is ready for Python or CLI reconstruction:

    python examples/import_walnut.py ~/data/walnuts/Walnut1 -o orbit.nxs
    tomojax recon orbit.nxs -o fdk.nxs --method fbp
    tomojax recon orbit.nxs -o cgls.nxs --method cgls --iterations 20

The default selects the middle orbit, every twentieth view, 12 x 12 detector
binning and a 64-cubed grid over the authors' 50.1 mm cube. Smaller binning,
more views and more voxels improve resolution but need more time and memory.
The scanner's original (uncorrected) geometry is used, as in the alignment example.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _walnut import load_scan
import tomojax as tj


def main() -> None:
    """Write a corrected, binned single-orbit dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walnut", type=Path, help="A WalnutN directory from the FIPS collection")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output scan (.nxs)")
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    parser.add_argument("--orbit", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--every", type=int, default=20, help="Use every Nth view")
    parser.add_argument("--binning", type=int, default=12, help="Average N x N detector pixels")
    parser.add_argument("--size", type=int, default=64, help="Voxels per side of the 50.1 mm cube")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"output already exists: {args.output} (pass --force to replace it)")
    logging.getLogger("tifffile").setLevel(logging.CRITICAL)  # malformed optional tag, skipped
    scan = load_scan(
        args.walnut, orbits=(args.orbit,), every=args.every, binning=args.binning, size=args.size
    )
    tj.save(args.output, scan)
    print(f"wrote {args.output}: {scan!r}")


if __name__ == "__main__":
    main()
