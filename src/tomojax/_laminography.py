"""Guarded automatic laminography, shared by alignment and reconstruction.

Explicit workflow operations are supplied by the caller: this policy composes
those operations without importing the workflow facade back into itself.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.ndimage import gaussian_filter

from tomojax._region import estimate_region
from tomojax._scan import describe, record_of
from tomojax.alignment import alignment_plan
from tomojax.alignment.api import AlignConfig, AlignmentSchedule, AlignmentStage
from tomojax.geometry import RotationAxisGeometry
from tomojax.recon import FistaConfig, project

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax._scan import Scan


def uses_laminography_workflow(scan: Scan) -> bool:
    """Oblique geometry, or an acquisition explicitly missing its axis metadata."""
    kind = describe(scan.geometry)[0]
    if kind == "lamino":
        return True
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    return kind == "parallel" and acquisition.get("geometry_status") == "not_recorded"


def _interpolate(table: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target, source, table[:, k]) for k in range(6)]).astype(
        np.float32
    )


def _final_config(scan: Scan) -> FistaConfig:
    epochs = 10 if scan.grid.vx > 1.5 * max(scan.detector.du, scan.detector.dv) else 8
    return FistaConfig(
        initial_epochs=epochs,
        initial_subsets=16,
        iterations=1,
        tv_weight=0,
        nonnegative=True,
        stream_projections=False,
    )


def _seed(scan: Scan, calibration: np.ndarray, align: Callable):
    original = scan
    if describe(scan.geometry)[0] == "parallel":
        scan = replace(
            scan,
            geometry=RotationAxisGeometry(
                scan.grid, scan.detector, scan.angles.tolist(), (0, 0, 1)
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
    seeded = align(scan.selected(calibration), mode="full", levels=(1,), config=cfg)
    seed_scan, seed_poses, setup_info = seeded.scan, np.asarray(seeded.poses), dict(seeded.info)
    del seeded
    limitation = _axis_limitation(original, seed_scan, setup_info)
    if limitation is not None:
        return original, limitation, setup_info
    axis = np.asarray(seed_scan.with_poses(None).geometry.axis_unit_lab)
    if abs(axis[1]) < 1e-3:
        return scan, {"status": "axial_or_unresolved"}, setup_info
    region = estimate_region(seed_scan)
    base = cast("RotationAxisGeometry", region.scan.with_poses(None).geometry)
    geometry = RotationAxisGeometry(
        region.grid,
        region.scan.detector,
        scan.angles.tolist(),
        np.asarray(base.axis_unit_lab).tolist(),
    )
    initial = _interpolate(seed_poses, calibration, np.arange(len(scan.angles)))
    work = replace(scan, geometry=geometry).with_poses(initial)
    if work.source is not None:
        metadata = dict(work.source.geometry_metadata)
        acquisition = dict(metadata.get("acquisition", {}))
        if acquisition.get("geometry_status") == "not_recorded":
            metadata["acquisition"] = {**acquisition, "geometry_status": "inferred"}
        if any(
            row.get("detector_center_seed_applied", False)
            for row in setup_info.get("outer_stats", [])
        ):
            # A calibrated result must not advertise its input's absent centre.
            metadata["detector_center_status"] = "inferred"
        work = replace(work, source=replace(work.source, geometry_metadata=metadata))
    return work, dict(region.info), setup_info


def _axis_limitation(
    scan: Scan, seeded: Scan, setup_info: dict[str, Any]
) -> dict[str, object] | None:
    """Decline absent-axis inference at search or finite-resolution limits."""
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    if acquisition.get("geometry_status") != "not_recorded":
        return None
    if any(
        row.get("axis_seed_on_search_boundary", False) for row in setup_info.get("outer_stats", [])
    ):
        return {
            "status": "axial_or_unresolved",
            "reason": "The unknown-axis estimate reached its search boundary; "
            "supply geometry or an explicit method.",
        }
    # Below one working pixel of parallax across a detector field, automatic
    # depth inference is unresolved. This is not a cap on physical support.
    geometry = cast("RotationAxisGeometry", seeded.with_poses(None).geometry)
    axis = np.asarray(geometry.axis_unit_lab)
    detector = seeded.detector  # Automatic setup admits only unrolled detectors.
    field = max(detector.nu * detector.du, detector.nv * detector.dv)
    parallax = float(field * np.hypot(axis[0] / detector.du, axis[2] / detector.dv))
    if parallax < 1:
        return {
            "status": "axial_or_unresolved",
            "reason": "The inferred axis provides less than one working detector pixel "
            "of depth parallax across the detector field; supply geometry or an explicit method.",
            "depth_parallax_pixels": parallax,
        }
    return None


def _select_motion(work, fit, validation, region_info, final_config, align, reconstruct):
    initial = np.asarray(work.poses)
    broad = region_info["status"] == "broad_uncertain"
    check_fit = fit
    if broad:
        check_fit = fit[np.linspace(0, len(fit) - 1, min(320, len(fit))).astype(int)]
        candidates = (("zero", np.zeros_like(initial)), ("seed", initial))
    else:
        config = replace(
            alignment_plan("pose", work.grid).config,
            seed_translations=False,
            outer_iterations=2,
            iterations=30,
            gn_joint_iterations=20,
        )
        refined = align(work.selected(fit), mode="pose", levels=(2,), config=config)
        alternative = _interpolate(np.asarray(refined.poses), fit, np.arange(len(work.angles)))
        del refined
        candidates = (("seed", initial), ("refined", alternative))
    target = np.asarray(work.projections)[validation]

    def high(a: np.ndarray) -> np.ndarray:
        return a - gaussian_filter(a, (0, 6, 6))

    target_hp = high(target)
    records = []
    for label, poses in candidates:
        candidate = work.with_poses(poses)
        rec = reconstruct(candidate.selected(check_fit), "fista", config=final_config)
        pred = np.asarray(project(candidate.selected(validation).geometry, rec.volume))
        records.append(
            {
                "label": label,
                "l2": float(np.linalg.norm(pred - target) / np.linalg.norm(target)),
                "hp": float(np.linalg.norm(high(pred) - target_hp) / np.linalg.norm(target_hp)),
            }
        )
        del rec, pred
    improved = all(records[1][key] < records[0][key] for key in ("l2", "hp"))
    return work.with_poses(candidates[int(improved)][1]), {
        "accepted": improved and not broad,
        "motion_accepted": improved if broad else None,
        "internal": records,
    }


def reconstruct_laminography(scan: Scan, *, align: Callable, reconstruct: Callable):
    """Select setup, support and motion internally, then reconstruct all input views.

    Previously posed scans retain their geometry and use only the final solver.
    Automatic unposed scans use at most 320 detector pixels across; the returned
    scan and diagnostic record make that working resolution explicit.
    """
    started = time.perf_counter()
    posed = scan.poses is not None
    automatic: dict[str, object] = {"method": "fista", "status": "ok", "reasons": []}
    if posed:
        automatic["geometry"] = "preserved_pose_corrections"
        reasons = []
        if scan.source is not None:
            for field in ("alignment_info", "reconstruction_info"):
                previous = scan.source.geometry_metadata.get(field, {}).get("automatic", {})
                if previous.get("status") == "limited":
                    automatic["status"] = "limited"
                    reasons.extend(previous.get("reasons", []))
        automatic["reasons"] = list(dict.fromkeys(reasons))
    elif float(cast("float", describe(scan.geometry)[1].get("detector_roll_deg", 0))) != 0:
        automatic.update(
            status="limited",
            reasons=["Automatic laminography setup requires an unrolled detector."],
        )
    else:
        factor = 1
        while max(scan.detector.nu, scan.detector.nv) // factor > 320:
            factor *= 2
        scan = scan.binned(factor) if factor > 1 else scan
        n = len(scan.angles)
        keep = np.linspace(0, n - 1, min(800, n)).astype(int)
        validation = keep[np.arange(len(keep)) % 5 == 2]
        fit = keep[np.arange(len(keep)) % 5 != 2]
        calibration = np.setdiff1d(np.arange(n), validation)
        work, region_info, setup_info = _seed(scan, calibration, align)
        if region_info["status"] == "axial_or_unresolved":
            result = reconstruct(work, "fbp")
            automatic.update(
                method="fbp",
                status="limited",
                setup=setup_info,
                region=region_info,
                reasons=[
                    region_info.get(
                        "reason",
                        "Projection evidence did not resolve an oblique axis; "
                        "supply geometry or an explicit method.",
                    )
                ],
            )
            return replace(result, info={**result.info, "automatic": automatic})
        scan, selected = _select_motion(
            work, fit, validation, region_info, _final_config(work), align, reconstruct
        )
        automatic.update(selected, region=region_info, setup=setup_info, detector_binning=factor)
        if region_info["status"] == "broad_uncertain":
            automatic.update(
                status="limited", reasons=["The object's axial support is unresolved."]
            )
    automatic["preparation_seconds"] = time.perf_counter() - started
    automatic["warning"] = "Held projection consistency is not physical pose or support truth."
    result = reconstruct(scan, "fista", config=_final_config(scan))
    if not posed:
        # Keep the bounded outcome on the scan too, so direct reuse need not
        # save/reload the Alignment to retain its limitations.
        source = record_of(result.scan)
        source.geometry_metadata["alignment_info"] = {
            "automatic": {key: automatic[key] for key in ("status", "reasons", "warning")}
        }
        result = replace(result, scan=replace(result.scan, source=source))
    return replace(result, info={**result.info, "automatic": automatic})
