"""Align raw NeXus laminography from its nominal tilt and detector pitch.

Raw frames record neither the laminography tilt nor, often, the detector pitch,
and alignment refuses undeclared geometry. This example reads the frame labels
and rotation angles (recognized Diamond/GDA layouts need no dataset paths),
declares a laminography geometry from ``--tilt`` and ``--pixel``, and fits the
selected parameters jointly with the image. Consistency is not physical truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import tomojax as tj
from tomojax.io.api import locate_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="Raw HDF5/NeXus file")
    parser.add_argument(
        "--tilt", type=float, required=True, help="Nominal axis tilt from detector vertical, deg"
    )
    parser.add_argument(
        "--pixel", type=float, required=True, help="Detector pixel pitch, in the output length unit"
    )
    parser.add_argument("--data-path", help="HDF5 frames dataset, if not discovered")
    parser.add_argument("--angles-path", help="HDF5 rotation angles dataset, if not discovered")
    parser.add_argument("--image-key-path", help="HDF5 frame labels dataset, if not discovered")
    parser.add_argument("--binning", type=int, default=1, help="Detector binning before fitting")
    parser.add_argument(
        "--fit",
        nargs="+",
        default=["translations", "tilt", "roll"],
        choices=["center", "tilt", "roll", "translations", "rotations"],
        help="Parameters fitted jointly with the image (default: translations tilt roll)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Save alignment and volume (.nxs)")
    args = parser.parse_args()
    paths = {
        "data_path": args.data_path,
        "angles_path": args.angles_path,
        "image_key_path": args.image_key_path,
    }
    located = locate_frames(str(args.input), **paths)
    if located.angles is None:
        parser.error(f"{args.input} records no rotation angles: pass --angles-path")
    key = located.image_key
    angles = located.angles if key is None else located.angles[np.asarray(key) == 0]
    rows, cols, pitch = located.stack.rows, located.stack.cols, args.pixel
    # A starting region the size of the detector; alignment bounds its resolution.
    geometry = tj.LaminographyGeometry(
        tj.Grid(cols, cols, rows, pitch, pitch, pitch),
        tj.Detector(cols, rows, pitch, pitch),
        angles,
        tilt=args.tilt,
    )
    frames = tj.load_frames(args.input, geometry=geometry, **paths)
    scan = frames.corrected(binning=args.binning)
    result = tj.align(scan, fit=tuple(args.fit))
    print(result.scan)
    print("Final level:", result.info["levels"][-1]["status"])
    if args.output is not None:
        tj.save(args.output, result)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
