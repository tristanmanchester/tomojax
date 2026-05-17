"""Render 3D diagnostics for per-view TEM-grid pose corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tomojax.bench import render_tem_grid_pose_artifacts


def render(args: argparse.Namespace) -> dict[str, Any]:
    return render_tem_grid_pose_artifacts(
        params_path=Path(args.params),
        out_dir=Path(args.out),
        radius=float(args.radius),
        correction_scale=float(args.correction_scale),
        yaw_deg=float(args.yaw_deg),
        pitch_deg=float(args.pitch_deg),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="params.csv from the final alignment stage")
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, default=80.0)
    parser.add_argument("--correction-scale", type=float, default=2.0)
    parser.add_argument("--yaw-deg", type=float, default=-42.0)
    parser.add_argument("--pitch-deg", type=float, default=46.0)
    manifest = render(parser.parse_args())
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
