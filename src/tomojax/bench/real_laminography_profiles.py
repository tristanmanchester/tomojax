"""Profile contracts for the real-laminography staged runner."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tomojax.bench.real_laminography_planning import (
    pose_dx_dz_bounds,
    pose_phi_bounds,
    pose_polish_bounds,
    validate_bin_factor,
)

REFERENCE_REGRESSION_CONTRACT: dict[str, Any] = {
    "projection_background": "edge_median",
    "background_edge_px": 16,
    "canonical_det_grid": False,
    "levels_setup": [8, 4, 2],
    "levels_phi": [4, 2, 1],
    "levels_dx_dz": [4, 2, 1],
    "levels_polish": [2, 1],
    "outer_iters": 8,
    "recon_iters": 40,
    "tv_prox_iters": 16,
    "lambda_tv": 0.008,
    "align_profile": "lightning",
    "regulariser": "huber_tv",
    "loss_spec": "l2_otsu",
    "loss_normalization": "align_config_default_l2_otsu_per_level",
    "mask_vol": "cyl",
    "optimizer_kind": "gn",
    "gn_damping": 1e-3,
    "quality_tier": "fast",
    "fallback_policy": "fallback",
    "fold_rigid_detector_grid": False,
    "pose_model": "per_view",
    "knot_spacing": 8,
    "pose_degree": 3,
    "pose_bounds_profile": "wide",
    "pose_gauge_policy": "mean_translation_for_dx_dz",
    "final_candidate_policy": "last_valid",
    "views_per_batch": 1,
    "gather_dtype": "bf16",
    "recon_positivity": False,
    "setup_outer_count_replay": "reference_stage_summary_counts",
    "pose_phi_bounds": "phi=-0.0872665:0.0872665",
    "pose_dx_dz_bounds": "dx=-16:16,dz=-16:16",
    "pose_polish_bounds": (
        "alpha=-0.0349066:0.0349066,beta=-0.0349066:0.0349066,"
        "phi=-0.0872665:0.0872665,dx=-16:16,dz=-16:16"
    ),
}

REAL_LAMINO_PROFILE_CHOICES = (
    "manual",
    "staged-lamino",
    "reference-regression",
    "diagnostic-fast",
)

STAGED_LAMINO_CONTRACT: dict[str, Any] = dict(REFERENCE_REGRESSION_CONTRACT)

REAL_LAMINO_STAGED_PATH: tuple[dict[str, Any], ...] = (
    {"label": "baseline", "stage": "00_baseline", "active_dofs": [], "status": "required"},
    {
        "label": "cor_detu",
        "stage": "01_setup_geometry/01_cor",
        "active_dofs": ["det_u_px"],
        "status": "required",
    },
    {
        "label": "detector_roll",
        "stage": "01_setup_geometry/02_detector_roll",
        "active_dofs": ["detector_roll_deg"],
        "status": "planned",
    },
    {
        "label": "axis_direction",
        "stage": "01_setup_geometry/03_axis_direction",
        "active_dofs": ["axis_rot_x_deg", "axis_rot_y_deg"],
        "status": "planned",
    },
    {"label": "pose_phi", "stage": "02_pose_phi", "active_dofs": ["phi"], "status": "planned"},
    {
        "label": "pose_dx_dz",
        "stage": "03_pose_dx_dz",
        "active_dofs": ["dx", "dz"],
        "status": "planned",
    },
    {
        "label": "pose_5dof_polish",
        "stage": "04_pose_polish",
        "active_dofs": ["alpha", "beta", "phi", "dx", "dz"],
        "status": "planned",
    },
    {
        "label": "final_reconstruction",
        "stage": "05_final",
        "active_dofs": ["detector_roll", "axis_direction", "pose_5dof"],
        "status": "planned",
    },
    {
        "label": "cor_only_comparator",
        "stage": "06_cor_only_fista",
        "active_dofs": ["det_u_px"],
        "status": "required",
    },
)

REFERENCE_REGRESSION_STAGE_MAP: tuple[tuple[str, str], ...] = (
    ("01_setup_geometry/01_cor", "01_setup_geometry/01_cor"),
    ("01_setup_geometry/02_detector_roll", "01_setup_geometry/02_detector_roll"),
    ("01_setup_geometry/03_axis_direction", "01_setup_geometry/03_axis_direction"),
    ("02_pose_phi", "02_pose_phi"),
    ("03_pose_dx_dz", "03_pose_dx_dz"),
    ("04_pose_polish", "04_pose_polish"),
    ("05_final", "05_final"),
    ("06_cor_only_fista", "06_cor_only_fista"),
)


def apply_real_lamino_profile_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if str(args.profile) == "reference-regression":
        if bool(args.smoke):
            parser.error("--profile reference-regression cannot be combined with diagnostic mode")
        apply_real_lamino_profile_contract_args(args, REFERENCE_REGRESSION_CONTRACT)
        args.reference_regression = True
    elif str(args.profile) == "staged-lamino":
        apply_real_lamino_profile_contract_args(args, STAGED_LAMINO_CONTRACT)
        args.reference_regression = False
    elif str(args.profile) == "diagnostic-fast":
        args.full_staged = True
        args.smoke = True
        if str(args.final_candidate_policy) == "all":
            args.final_candidate_policy = "last_valid"
        args.reference_regression = False
    else:
        args.reference_regression = False


def apply_real_lamino_profile_contract_args(
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> None:
    args.full_staged = True
    args.levels_setup = list(contract["levels_setup"])
    args.levels_phi = list(contract["levels_phi"])
    args.levels_dx_dz = list(contract["levels_dx_dz"])
    args.levels_polish = list(contract["levels_polish"])
    args.outer_iters = int(contract["outer_iters"])
    args.recon_iters = int(contract["recon_iters"])
    args.tv_prox_iters = int(contract["tv_prox_iters"])
    args.lambda_tv = float(contract["lambda_tv"])
    args.align_profile = str(contract["align_profile"])
    args.regulariser = str(contract["regulariser"])
    args.gn_damping = float(contract["gn_damping"])
    args.quality_tier = str(contract["quality_tier"])
    args.fallback_policy = str(contract["fallback_policy"])
    args.fold_rigid_detector_grid = bool(contract["fold_rigid_detector_grid"])
    args.pose_model = str(contract["pose_model"])
    args.knot_spacing = int(contract["knot_spacing"])
    args.pose_degree = int(contract["pose_degree"])
    args.pose_bounds_profile = str(contract["pose_bounds_profile"])
    args.canonical_det_grid = bool(contract["canonical_det_grid"])
    args.projection_background = str(contract["projection_background"])
    args.background_edge_px = int(contract["background_edge_px"])
    args.recon_positivity = bool(contract["recon_positivity"])
    args.views_per_batch = int(contract["views_per_batch"])
    args.gather_dtype = str(contract["gather_dtype"])
    args.final_candidate_policy = str(contract["final_candidate_policy"])


def normalize_real_lamino_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve memory-sensitive runtime defaults after CLI parsing."""
    if int(args.views_per_batch) <= 0:
        args.views_per_batch = 1
    args.bin_factor = validate_bin_factor(args.bin_factor)
    return args


def real_lamino_reference_regression_contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    actual = {
        "projection_background": str(args.projection_background),
        "background_edge_px": int(args.background_edge_px),
        "canonical_det_grid": bool(args.canonical_det_grid),
        "levels_setup": list(args.levels_setup),
        "levels_phi": list(args.levels_phi),
        "levels_dx_dz": list(args.levels_dx_dz),
        "levels_polish": list(args.levels_polish),
        "outer_iters": int(args.outer_iters),
        "recon_iters": int(args.recon_iters),
        "tv_prox_iters": int(args.tv_prox_iters),
        "lambda_tv": float(args.lambda_tv),
        "align_profile": str(args.align_profile),
        "regulariser": str(args.regulariser),
        "loss_spec": "l2_otsu",
        "loss_normalization": "align_config_default_l2_otsu_per_level",
        "mask_vol": "cyl",
        "optimizer_kind": "gn",
        "gn_damping": float(args.gn_damping),
        "quality_tier": str(args.quality_tier),
        "fallback_policy": str(args.fallback_policy),
        "fold_rigid_detector_grid": bool(getattr(args, "fold_rigid_detector_grid", True)),
        "pose_model": str(args.pose_model),
        "knot_spacing": int(args.knot_spacing),
        "pose_degree": int(args.pose_degree),
        "pose_bounds_profile": str(args.pose_bounds_profile),
        "pose_gauge_policy": "mean_translation_for_dx_dz",
        "final_candidate_policy": str(args.final_candidate_policy),
        "views_per_batch": int(args.views_per_batch),
        "gather_dtype": str(args.gather_dtype),
        "recon_positivity": bool(args.recon_positivity),
        "setup_outer_count_replay": "reference_stage_summary_counts",
        "pose_phi_bounds": pose_phi_bounds(args),
        "pose_dx_dz_bounds": pose_dx_dz_bounds(args),
        "pose_polish_bounds": pose_polish_bounds(args),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in REFERENCE_REGRESSION_CONTRACT.items()
        if actual.get(key) != expected
    }
    return {
        "schema": "tomojax.real_lamino_reference_regression_contract.v2",
        "source_script": "scripts/real_laminography/run_real_lamino_reference_regression.py",
        "expected": REFERENCE_REGRESSION_CONTRACT,
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def reference_regression_level_outer_counts(
    args: argparse.Namespace,
    *,
    stage_name: str,
) -> dict[int, int] | None:
    """Return reference-run per-level setup row counts for strict replay."""
    if str(getattr(args, "profile", "")) != "reference-regression":
        return None
    reference_report = getattr(args, "reference_report", None)
    if not reference_report:
        return None
    reference_root = Path(reference_report).resolve().parents[1]
    summary_path = reference_root / stage_name / "stage_summary.csv"
    counts: dict[int, int] = {}
    for row in _read_reference_stage_summary(summary_path):
        try:
            level = int(row.get("level_factor", ""))
        except (TypeError, ValueError):
            continue
        counts[level] = counts.get(level, 0) + 1
    return counts or None


def _read_reference_stage_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


__all__ = [
    "REAL_LAMINO_PROFILE_CHOICES",
    "REAL_LAMINO_STAGED_PATH",
    "REFERENCE_REGRESSION_CONTRACT",
    "REFERENCE_REGRESSION_STAGE_MAP",
    "STAGED_LAMINO_CONTRACT",
    "apply_real_lamino_profile_args",
    "apply_real_lamino_profile_contract_args",
    "normalize_real_lamino_runtime_args",
    "real_lamino_reference_regression_contract_payload",
    "reference_regression_level_outer_counts",
]
