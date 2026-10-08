#!/usr/bin/env python3
"""Exercise real solvers on gVXR fixtures without importing its OpenGL runtime.

These fixed-budget diagnostics are not time-to-quality benchmarks. Polychromatic
volume errors use an 80 keV reference to expose model mismatch, not assert that a
single-energy linear solver can invert spectral data exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from compare_projectors import Case
from compare_reconstructions import environment, solve_astra
import numpy as np

from tomojax.geometry import Detector, Grid
from tomojax.recon import CGLSConfig, cgls


@dataclass
class FixtureGeometry:
    """Use all saved rigid poses, including their translations and tilt."""

    poses: np.ndarray

    def pose_for_view(self, index: int) -> np.ndarray:
        """Return the exact world-from-object matrix used during rendering."""
        return self.poses[index]


def check_fixture(path: Path, iterations: int) -> dict:
    """Compare two TomoJAX discretizations and ASTRA against one fixed dataset."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data["metadata"]))
    grid, detector = Grid(**metadata["grid"]), Detector.from_dict(metadata["detector"])
    truth = data["truth"].astype(np.float64)
    records = []
    for channel in ("mono_log", "poly_log", "poly_noisy_log"):
        case = Case(path.stem, grid, detector, data["poses"], data["angles"], truth, data[channel])
        for method in ("ray", "joseph", "astra_cgls"):
            if method == "astra_cgls":
                volume, info = solve_astra(case, iterations, method)
            else:
                volume, info = cgls(
                    FixtureGeometry(data["poses"]),
                    grid,
                    detector,
                    data[channel],
                    config=CGLSConfig(
                        iterations=iterations,
                        rtol=0.0,
                        views_per_batch=len(data["poses"]),
                        projector_backend="pallas",
                        projector_model=method,
                    ),
                )
            volume = np.asarray(volume, dtype=np.float64)
            finite = bool(np.isfinite(volume).all())
            if not finite or info.get("termination") == "numerical_breakdown":
                raise RuntimeError(f"{path.name}/{channel}/{method}: non-finite or broken solve")
            records.append(
                {
                    "channel": channel,
                    "method": method,
                    "iterations": iterations,
                    "volume_relative_l2_to_80kev": float(
                        np.linalg.norm(volume - truth) / np.linalg.norm(truth)
                    ),
                    "finite": finite,
                    "solver_info": info,
                    "material_mean_relative_bias": {
                        item["label"]: float(
                            np.mean(volume[np.isclose(truth, item["mu_mm_inverse"][2], rtol=1e-5)])
                            / item["mu_mm_inverse"][2]
                            - 1
                        )
                        for item in metadata["materials"]
                    },
                }
            )
    return {
        "fixture": path.name,
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata": metadata,
        "records": records,
    }


def main() -> None:
    """Write raw physical-scale errors for the fixed iteration budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("bench/results/gvxr-reconstruction.json")
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    result = {
        "environment": environment(),
        "scope": "fixed-budget correctness diagnostic, no speed claim",
        "quality_gate": (
            "finite result without numerical breakdown; errors reported without scale fitting"
        ),
        "records": [check_fixture(path, args.iterations) for path in args.fixtures],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
