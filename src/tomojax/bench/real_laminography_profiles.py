"""Profile contracts for the real-laminography staged runner."""

from __future__ import annotations

from typing import Any

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
