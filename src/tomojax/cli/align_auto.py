"""CLI for the Phase 7 deterministic auto-alignment smoke pipeline."""
# pyright: reportAny=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from tomojax.align.api import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    ContinuationScheduleName,
    GeometryUpdateVolumeSource,
    PreviewInitialization,
    PreviewResidualFilterMode,
    PreviewVolumeSupport,
    StoppedPreviewPolicy,
    reference_continuation_schedule,
)
from tomojax.datasets import (
    SyntheticDatasetSidecars,
    generate_synthetic_dataset,
    load_synthetic_dataset_sidecars,
    synthetic128_spec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROFILE_CHOICES = ("smoke32", "lightning", "balanced", "reference")
_SYNTHETIC_SIZE_CHOICES = (32, 64, 128)
_GEOMETRY_UPDATE_VOLUME_SOURCE_CHOICES = ("stopped_reconstruction", "fixed_synthetic_truth")
_PREVIEW_VOLUME_SUPPORT_CHOICES = ("none", "cylindrical", "spherical")
_PREVIEW_INITIALIZATION_CHOICES = ("backprojection", "zero", "constant", "average_projection")
_PREVIEW_RESIDUAL_FILTER_MODE_CHOICES = ("continuation", "raw")
_STOPPED_PREVIEW_POLICY_CHOICES = ("standard", "constant_cylindrical_first_level")
SyntheticSize = Literal[32, 64, 128]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic Phase 7 align=auto smoke pipeline and write "
            "final volume, geometry, and verification artifacts."
        )
    )
    _ = parser.add_argument(
        "--out-dir",
        required=True,
        help="Run directory for smoke artifacts.",
    )
    _ = parser.add_argument(
        "--profile",
        choices=_PROFILE_CHOICES,
        default="smoke32",
        help="Continuation profile for the deterministic smoke run.",
    )
    _ = parser.add_argument("--seed", type=int, default=17, help="Synthetic phantom seed.")
    _ = parser.add_argument(
        "--size",
        type=int,
        choices=_SYNTHETIC_SIZE_CHOICES,
        default=32,
        help="Synthetic cubic volume size.",
    )
    _ = parser.add_argument("--views", type=int, default=4, help="Number of synthetic views.")
    _ = parser.add_argument(
        "--geometry-update-volume-source",
        choices=_GEOMETRY_UPDATE_VOLUME_SOURCE_CHOICES,
        default="stopped_reconstruction",
        help=(
            "Volume source for Schur geometry updates. "
            "Use fixed_synthetic_truth only for synthetic oracle diagnostics."
        ),
    )
    _ = parser.add_argument(
        "--geometry-update-setup-prior-strength",
        type=float,
        help="Optional setup-parameter prior strength for Schur geometry updates.",
    )
    _ = parser.add_argument(
        "--geometry-update-pose-prior-strength",
        type=float,
        help="Optional per-view pose prior strength for Schur geometry updates.",
    )
    _ = parser.add_argument(
        "--synthetic-dataset",
        help="Optional synthetic128 benchmark spec name to generate and record for this run.",
    )
    _ = parser.add_argument(
        "--dataset-out-dir",
        help="Directory for generated synthetic benchmark artifacts. Defaults under --out-dir.",
    )
    _ = parser.add_argument(
        "--synthetic-dataset-dir",
        help=(
            "Existing generated synthetic benchmark artifact directory to ingest. "
            "When supplied, --synthetic-dataset is optional metadata and no dataset is generated."
        ),
    )
    _ = parser.add_argument(
        "--current-default-baseline-json",
        help=(
            "Optional current/default TomoJAX baseline JSON artifact with a volume_nmse "
            "field for benchmark comparison criteria."
        ),
    )
    _ = parser.add_argument(
        "--apply-synthetic-nuisance",
        action="store_true",
        help="Apply nuisance terms from the named synthetic benchmark to generated projections.",
    )
    _ = parser.add_argument(
        "--fit-gain-offset-nuisance",
        action="store_true",
        help="Fit per-view gain/offset nuisance during Schur geometry updates.",
    )
    _ = parser.add_argument(
        "--fit-background-nuisance",
        action="store_true",
        help="Fit low-frequency background nuisance during Schur geometry updates.",
    )
    _ = parser.add_argument(
        "--supported-only-setup-global",
        action="store_true",
        help=(
            "Generate a clean setup-global sidecar variant limited to supported "
            "theta/detector-shift DOFs."
        ),
    )
    _ = parser.add_argument(
        "--geometry-update-pose-frozen",
        action="store_true",
        help="Freeze per-view pose DOFs during Schur geometry updates.",
    )
    _ = parser.add_argument(
        "--geometry-update-active-pose-dofs",
        default="phi_residual_rad,dx_px,dz_px",
        help=(
            "Comma-separated active pose DOFs for Schur updates. "
            "Supported names: alpha_rad, beta_rad, phi_residual_rad, dx_px, dz_px."
        ),
    )
    _ = parser.add_argument(
        "--geometry-update-active-setup-parameters",
        default="theta_offset_rad,det_u_px",
        help=(
            "Comma-separated active setup parameters for Schur updates. "
            "Supported names: theta_offset_rad, det_u_px, det_v_px, "
            "detector_roll_rad, axis_rot_x_rad, axis_rot_y_rad, theta_scale."
        ),
    )
    _ = parser.add_argument(
        "--geometry-update-pose-activate-at-level-factor",
        type=int,
        help=(
            "Keep pose frozen for coarser levels and activate configured pose DOFs "
            "at this continuation level factor or finer."
        ),
    )
    _ = parser.add_argument(
        "--preview-volume-support",
        choices=_PREVIEW_VOLUME_SUPPORT_CHOICES,
        default="none",
        help="Optional centered support mask for preview reconstruction.",
    )
    _ = parser.add_argument(
        "--preview-initialization",
        choices=_PREVIEW_INITIALIZATION_CHOICES,
        default="backprojection",
        help="Initial volume source for preview reconstruction.",
    )
    _ = parser.add_argument(
        "--preview-tv-scale",
        type=float,
        default=1.0,
        help="Scale factor for continuation preview TV weights.",
    )
    _ = parser.add_argument(
        "--preview-residual-filter-mode",
        choices=_PREVIEW_RESIDUAL_FILTER_MODE_CHOICES,
        default="continuation",
        help="Residual filters used inside preview reconstruction.",
    )
    _ = parser.add_argument(
        "--preview-center-l2-weight",
        type=float,
        default=0.0,
        help="Opt-in lateral center-of-mass gauge penalty for preview FISTA.",
    )
    _ = parser.add_argument(
        "--stopped-preview-policy",
        choices=_STOPPED_PREVIEW_POLICY_CHOICES,
        default="standard",
        help=(
            "Optional first stopped-reconstruction preview constraint. "
            "constant_cylindrical_first_level uses a constant initial volume, "
            "cylindrical support, and raw residuals for the coarsest preview only."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic Phase 7 auto-alignment smoke command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    profile = cast("ContinuationScheduleName", args.profile)
    geometry_update_volume_source = cast(
        "GeometryUpdateVolumeSource",
        args.geometry_update_volume_source,
    )
    preview_volume_support = cast("PreviewVolumeSupport", args.preview_volume_support)
    preview_initialization = cast("PreviewInitialization", args.preview_initialization)
    preview_residual_filter_mode = cast(
        "PreviewResidualFilterMode",
        args.preview_residual_filter_mode,
    )
    stopped_preview_policy = cast("StoppedPreviewPolicy", args.stopped_preview_policy)
    size = cast("SyntheticSize", int(args.size))
    views = int(args.views)
    out_dir = Path(args.out_dir)
    dataset_name = None if args.synthetic_dataset is None else str(args.synthetic_dataset)
    dataset_dir: Path | None = None
    sidecar_readback: dict[str, object] | None = None
    synthetic_nuisance_applied = bool(args.apply_synthetic_nuisance)
    if args.synthetic_dataset_dir is not None:
        dataset_dir = Path(args.synthetic_dataset_dir)
        sidecars = load_synthetic_dataset_sidecars(dataset_dir)
        manifest_name = _sidecar_manifest_name(sidecars)
        if dataset_name is not None and dataset_name != manifest_name:
            parser.error(
                "--synthetic-dataset must match the existing sidecar manifest name "
                f"{manifest_name!r}"
            )
        dataset_name = manifest_name
        size, views = _sidecar_size_and_views(sidecars)
        synthetic_nuisance_applied = _sidecar_nuisance_applied(sidecars)
        sidecar_readback = _sidecar_readback_payload(sidecars)
    elif dataset_name is not None:
        _ = synthetic128_spec(dataset_name)
        dataset_root = Path(args.dataset_out_dir) if args.dataset_out_dir else out_dir / "datasets"
        dataset_paths = generate_synthetic_dataset(
            dataset_name,
            dataset_root,
            size=size,
            clean=not synthetic_nuisance_applied,
            views=views,
            supported_only=bool(args.supported_only_setup_global),
        )
        dataset_dir = dataset_paths.dataset_dir
        sidecar_readback = _sidecar_readback_payload(load_synthetic_dataset_sidecars(dataset_dir))
    if sidecar_readback is not None and args.current_default_baseline_json is not None:
        sidecar_readback["current_default_baseline"] = _current_default_baseline_payload(
            Path(args.current_default_baseline_json)
        )
    solver = AlternatingAlignmentSolver(
        AlternatingSmokeConfig(
            seed=int(args.seed),
            size=size,
            n_views=views,
            schedule=reference_continuation_schedule(profile),
            geometry_update_volume_source=geometry_update_volume_source,
            geometry_update_setup_prior_strength=args.geometry_update_setup_prior_strength,
            geometry_update_pose_prior_strength=args.geometry_update_pose_prior_strength,
            geometry_update_pose_frozen=bool(args.geometry_update_pose_frozen),
            geometry_update_pose_activate_at_level_factor=(
                args.geometry_update_pose_activate_at_level_factor
            ),
            geometry_update_active_setup_parameters=_parse_active_setup_parameters(
                str(args.geometry_update_active_setup_parameters)
            ),
            geometry_update_active_pose_dofs=_parse_active_pose_dofs(
                str(args.geometry_update_active_pose_dofs)
            ),
            preview_volume_support=preview_volume_support,
            preview_initialization=preview_initialization,
            preview_tv_scale=float(args.preview_tv_scale),
            preview_residual_filter_mode=preview_residual_filter_mode,
            preview_center_l2_weight=float(args.preview_center_l2_weight),
            stopped_preview_policy=stopped_preview_policy,
            fit_gain_offset_nuisance=bool(args.fit_gain_offset_nuisance),
            fit_background_nuisance=bool(args.fit_background_nuisance),
            synthetic_dataset_name=dataset_name,
            synthetic_dataset_artifact_dir=dataset_dir,
            synthetic_dataset_nuisance_applied=synthetic_nuisance_applied,
            synthetic_dataset_sidecar_readback=sidecar_readback,
        )
    )
    result = solver.run_smoke(out_dir)
    if dataset_dir is not None:
        print(f"synthetic_dataset: {dataset_dir}")
    print(f"verification: {result.artifacts['verification_json']}")
    print(f"geometry: {result.artifacts['geometry_final_json']}")
    print(f"volume: {result.artifacts['final_volume_npy']}")
    return 0


def _sidecar_readback_payload(sidecars: SyntheticDatasetSidecars) -> dict[str, object]:
    """Return a compact verification payload for generated synthetic sidecars."""
    payload: dict[str, object] = {
        "validated": True,
        "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
        "n_views": sidecars.true_geometry.pose.n_views,
        "true_det_u_px": sidecars.true_geometry.setup.det_u_px.value,
        "nominal_det_u_px": sidecars.nominal_geometry.setup.det_u_px.value,
        "corrupted_det_u_px": sidecars.corrupted_geometry.setup.det_u_px.value,
        "volume": sidecars.volume.to_dict(),
        "projections": sidecars.projections.to_dict(),
        "mask": sidecars.mask.to_dict(),
        "consistency": sidecars.consistency.to_dict(),
        "recovery_tolerances": _sidecar_recovery_tolerances(sidecars),
        "unsupported_dofs_not_evaluated": sidecars.manifest.get(
            "unsupported_dofs_not_evaluated",
            [],
        ),
        "unsupported_dof_status": sidecars.manifest.get("unsupported_dof_status"),
        "true_object_motion": _sidecar_object_motion_payload(sidecars),
    }
    detector_grid = sidecars.manifest.get("detector_grid")
    if isinstance(detector_grid, str):
        payload["detector_grid"] = detector_grid
    return payload


def _sidecar_object_motion_payload(sidecars: SyntheticDatasetSidecars) -> dict[str, object]:
    trace = sidecars.true_motion
    zero = type(trace).zeros(trace.n_views)
    return {
        "schema": "tomojax.object_motion_truth.v1",
        "n_views": trace.n_views,
        "tx_span_px": float(np.max(trace.tx_obj_px) - np.min(trace.tx_obj_px))
        if trace.n_views
        else 0.0,
        "tx_zero_model_rmse_px": trace.tx_rmse_px(zero) if trace.n_views else 0.0,
        "has_nonzero_motion": bool(
            trace.n_views
            and (
                np.any(np.abs(trace.tx_obj_px) > 0.0)
                or np.any(np.abs(trace.ty_obj_px) > 0.0)
                or np.any(np.abs(trace.tz_obj_px) > 0.0)
                or np.any(np.abs(trace.rot_obj_z_deg) > 0.0)
            )
        ),
    }


def _current_default_baseline_payload(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("current-default baseline JSON must contain an object")
    payload = cast("dict[object, object]", raw)
    volume_nmse = _baseline_volume_nmse(payload)
    if volume_nmse is None:
        raise ValueError("current-default baseline JSON must contain numeric volume_nmse")
    return {
        "schema": "tomojax.current_default_baseline.v1",
        "source_path": str(path),
        "volume_nmse": volume_nmse,
        "raw": {str(key): value for key, value in payload.items()},
    }


def _baseline_volume_nmse(payload: dict[object, object]) -> float | None:
    direct = payload.get("volume_nmse")
    if isinstance(direct, int | float):
        return float(direct)
    reconstruction = payload.get("reconstruction")
    if isinstance(reconstruction, dict):
        value = cast("dict[object, object]", reconstruction).get("volume_nmse")
        if isinstance(value, int | float):
            return float(value)
    return None


def _parse_active_pose_dofs(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() in {"", "none"}:
        return ()
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"alpha_rad", "beta_rad", "phi_residual_rad", "dx_px", "dz_px"}
    if any(value not in allowed for value in values):
        raise ValueError(f"unsupported --geometry-update-active-pose-dofs value {raw!r}")
    return values


def _parse_active_setup_parameters(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() in {"", "none"}:
        return ()
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {
        "axis_rot_x_rad",
        "axis_rot_y_rad",
        "theta_offset_rad",
        "det_u_px",
        "det_v_px",
        "detector_roll_rad",
        "theta_scale",
    }
    if any(value not in allowed for value in values):
        raise ValueError(f"unsupported --geometry-update-active-setup-parameters value {raw!r}")
    return values


def _sidecar_manifest_name(sidecars: SyntheticDatasetSidecars) -> str:
    name = sidecars.manifest.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("synthetic dataset sidecar manifest must contain a string name")
    return name


def _sidecar_size_and_views(sidecars: SyntheticDatasetSidecars) -> tuple[SyntheticSize, int]:
    raw_volume_shape = cast("object", sidecars.manifest.get("volume_shape"))
    if not isinstance(raw_volume_shape, list):
        raise ValueError("synthetic dataset sidecar manifest must contain volume_shape")
    volume_shape = cast("list[object]", raw_volume_shape)
    if len(volume_shape) != 3:
        raise ValueError("synthetic dataset sidecar manifest must contain volume_shape")
    if not all(isinstance(dim, int) for dim in volume_shape):
        raise ValueError("synthetic dataset sidecar volume_shape must contain integers")
    dims = [dim for dim in volume_shape if isinstance(dim, int)]
    size = dims[0]
    if dims != [size, size, size]:
        raise ValueError("synthetic dataset sidecar volume_shape must be cubic")
    views = sidecars.manifest.get("views")
    if not isinstance(views, int):
        raise ValueError("synthetic dataset sidecar manifest must contain integer views")
    if size == 32:
        return 32, views
    if size == 64:
        return 64, views
    if size == 128:
        return 128, views
    raise ValueError("synthetic dataset sidecar size must be 32, 64, or 128")


def _sidecar_nuisance_applied(sidecars: SyntheticDatasetSidecars) -> bool:
    path = sidecars.artifacts.get("nuisance_truth_json")
    if path is None:
        return False
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    return bool(payload.get("applied_to_projections"))


def _sidecar_recovery_tolerances(sidecars: SyntheticDatasetSidecars) -> dict[str, object]:
    tolerances = sidecars.manifest.get("recovery_tolerances")
    if isinstance(tolerances, dict):
        tolerance_items = cast("dict[object, object]", tolerances)
        return {str(key): value for key, value in tolerance_items.items()}
    return {}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
