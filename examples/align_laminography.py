"""Reconstruct raw HDF5 laminography using nominal tilt and train-internal checks.

This conservative example estimates setup and smooth detector motion, selects a
reconstruction region, and tries sparse pose refinement only for a resolved
region. It retains refinement only when untouched projections improve in both
ordinary and high-pass error at the final grid. These checks do not establish
physical pose or support truth. Inspect the volume and correction records.

The input is an unrolled absorption scan; angles are read explicitly in degrees.
No output is written unless requested. The iteration/sampling choices are an
example, not library defaults or a convergence guarantee.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import tomojax as tj
from tomojax.alignment import alignment_plan
from tomojax.alignment.api import AlignConfig, AlignmentSchedule, AlignmentStage
from tomojax.geometry import RotationAxisGeometry
from tomojax.recon import FistaConfig


def interpolate(table: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target, source, table[:, k]) for k in range(6)]).astype(
        np.float32
    )


def reconstruct_lamino(
    scan: tj.Scan, *, nominal_tilt_deg: float
) -> tuple[tj.Reconstruction, dict[str, object]]:
    """Return a reconstruction and its conditional selection record.

    Broad/uncertain regions retain their unvalidated projection-only seed.
    This conservative choice is not a statement that broad objects cannot align.
    """
    started = time.perf_counter()
    n = len(scan.angles)
    keep = np.linspace(0, n - 1, min(800, n)).astype(int)
    validation = keep[np.arange(len(keep)) % 5 == 2]
    fit = keep[np.arange(len(keep)) % 5 != 2]
    calibration = np.setdiff1d(np.arange(n), validation)
    scan = replace(
        scan,
        geometry=tj.LaminographyGeometry(
            scan.grid, scan.detector, scan.angles, tilt_deg=nominal_tilt_deg
        ),
    )
    schedule = AlignmentSchedule(
        "spectral_setup_and_motion",
        (
            AlignmentStage(
                "spectral_setup",
                ("det_u_px", "axis_rot_x_deg", "axis_rot_y_deg"),
                "projection_consistency",
                "spectral",
                stage_role="setup",
                differentiability="not_required",
            ),
            AlignmentStage(
                "spectral_motion",
                ("dx", "dz"),
                "projection_consistency",
                "spectral",
                gauge_policy="diagnose_only",
                stage_role="setup",
                differentiability="not_required",
            ),
        ),
    )
    cfg = AlignConfig(
        schedule=schedule,
        pose_translation_frame="detector",
        seed_translations=True,
        gn_coupling="joint",
        iterations=1,
        outer_iterations=1,
        tv_weight=0,
        projector_backend="pallas",
        ray_integrator="joseph",
    )
    seeded = tj.align(scan.selected(calibration), mode="full", levels=(1,), config=cfg)
    seed_scan, seed_poses = seeded.scan, np.asarray(seeded.poses)
    del seeded
    region = tj.estimate_region(seed_scan)
    # This schedule starts unrolled and never activates detector roll.
    base = region.scan.with_poses(None).geometry
    geometry = RotationAxisGeometry(
        region.grid, region.scan.detector, scan.angles, np.asarray(base.axis_unit_lab)
    )
    initial = interpolate(seed_poses, calibration, np.arange(n))
    work = replace(scan, geometry=geometry).with_poses(initial)
    epochs = 10 if work.grid.vx > 1.5 * max(work.detector.du, work.detector.dv) else 8
    final_config = FistaConfig(
        initial_epochs=epochs,
        initial_subsets=16,
        iterations=1,
        tv_weight=0,
        nonnegative=True,
        stream_projections=False,
    )
    region_info = dict(region.info)
    del region, seed_scan
    records = []
    accepted = False
    if region_info["status"] != "broad_uncertain":
        config = replace(
            alignment_plan("pose", work.grid).config,
            seed_translations=False,
            outer_iterations=2,
            iterations=30,
            gn_joint_iterations=20,
        )
        refined = tj.align(work.selected(fit), mode="pose", levels=(2,), config=config)
        alternative = interpolate(np.asarray(refined.poses), fit, np.arange(n))
        del refined
        target = np.asarray(work.projections)[validation]

        def high(a: np.ndarray) -> np.ndarray:
            return a - gaussian_filter(a, (0, 6, 6))

        target_hp = high(target)
        for label, poses in (("seed", initial), ("refined", alternative)):
            candidate = work.with_poses(poses)
            rec = tj.reconstruct(candidate.selected(fit), "fista", config=final_config)
            pred = np.asarray(tj.project(candidate.selected(validation).geometry, rec.volume))
            records.append(
                {
                    "label": label,
                    "l2": float(np.linalg.norm(pred - target) / np.linalg.norm(target)),
                    "hp": float(np.linalg.norm(high(pred) - target_hp) / np.linalg.norm(target_hp)),
                }
            )
            del rec, pred
        accepted = all(records[1][key] < records[0][key] for key in ("l2", "hp"))
        if accepted:
            work = work.with_poses(alternative)
    prepared = time.perf_counter() - started
    result = tj.reconstruct(work, "fista", config=final_config)
    return result, {
        "accepted": accepted,
        "internal": records,
        "region": region_info,
        "preparation_seconds": prepared,
        "warning": "Held projection consistency is not physical pose or support truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="Raw HDF5/NeXus file")
    parser.add_argument("--data-path", default="/entry/imaging/data")
    parser.add_argument("--image-key-path", required=True)
    parser.add_argument("--angles-path", required=True, help="Recorded angles in degrees")
    parser.add_argument("--tilt", type=float, required=True, help="Nominal tilt in degrees")
    parser.add_argument("-o", "--output", type=Path, help="Save the reconstruction (.nxs)")
    args = parser.parse_args()
    with h5py.File(args.input) as handle:
        angles = handle[args.angles_path][()][handle[args.image_key_path][()] == 0]
    frames = tj.load_frames(
        args.input, data_path=args.data_path, image_key_path=args.image_key_path, angles=angles
    )
    scan = frames.corrected(binning=2)
    result, info = reconstruct_lamino(scan, nominal_tilt_deg=args.tilt)
    print(scan)  # Includes the correction's nonpositive-transmission count.
    print("Region:", info["region"])
    print("Pose refinement retained:", info["accepted"])
    print("Internal held projections:", info["internal"])
    print(info["warning"])
    if args.output is not None:
        tj.save(args.output, result)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
