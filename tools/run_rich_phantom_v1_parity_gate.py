"""Run the narrow rich PHANTOM94 v1-parity setup/global gate."""
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, cast

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from tomojax.align.api import (
    AlternatingAlignmentSolver,
    AlternatingSmokeConfig,
    reference_continuation_schedule,
)
from tomojax.core.multires import bin_volume, upsample_volume
from tomojax.datasets import generate_synthetic_dataset, load_synthetic_dataset_sidecars
from tomojax.forward import project_parallel_reference
from tomojax.geometry import (
    GeometryState,
    PoseParameters,
    read_geometry_json,
    read_pose_params_csv,
    write_geometry_json,
    write_pose_params_csv,
)

DATASET_NAME = "rich_phantom94_det_u_only_v1_parity"
FACTORS = (4, 2, 1)


def main() -> int:
    """Run the requested fixed-truth or stopped multires gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, default=128, choices=(128, 256))
    parser.add_argument("--profile", default="reference")
    parser.add_argument(
        "--mode",
        choices=("fixed_truth", "stopped_multires"),
        default="stopped_multires",
    )
    parser.add_argument(
        "--preview-volume-support",
        choices=("cylindrical", "none", "scout_soft", "spherical"),
        default="cylindrical",
    )
    parser.add_argument("--preview-support-outside-weight", type=float, default=0.0)
    parser.add_argument("--preview-low-frequency-anchor-weight", type=float, default=0.0)
    parser.add_argument("--preview-det-u-gauge-mode-weight", type=float, default=0.0)
    args = parser.parse_args()

    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    full = generate_synthetic_dataset(
        DATASET_NAME,
        root / "datasets_source",
        size=128,
        clean=True,
        views=args.views,
        supported_only=True,
    )
    if args.mode == "fixed_truth":
        rows = [
            _run_solver_inprocess(
                root=root,
                run_name=f"fixed_truth_otsu_l2_{args.profile}_{args.views}v",
                dataset_dir=full.dataset_dir,
                size=128,
                views=args.views,
                profile=args.profile,
                volume_source="fixed_synthetic_truth",
                preview_volume_support=args.preview_volume_support,
                preview_support_outside_weight=args.preview_support_outside_weight,
                preview_low_frequency_anchor_weight=args.preview_low_frequency_anchor_weight,
                preview_det_u_gauge_mode_weight=args.preview_det_u_gauge_mode_weight,
            )
        ]
    else:
        rows = _run_stopped_multires(
            root=root,
            full_dataset_dir=full.dataset_dir,
            views=args.views,
            profile=args.profile,
            preview_volume_support=args.preview_volume_support,
            preview_support_outside_weight=args.preview_support_outside_weight,
            preview_low_frequency_anchor_weight=args.preview_low_frequency_anchor_weight,
            preview_det_u_gauge_mode_weight=args.preview_det_u_gauge_mode_weight,
        )
    _write_summary(root, rows)
    return 0


def _run_stopped_multires(
    *,
    root: Path,
    full_dataset_dir: Path,
    views: int,
    profile: str,
    preview_volume_support: str,
    preview_support_outside_weight: float,
    preview_low_frequency_anchor_weight: float,
    preview_det_u_gauge_mode_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    carried_geometry: GeometryState | None = None
    carried_factor: int | None = None
    carried_volume_path: Path | None = None
    carried_volume_factor: int | None = None
    for factor in FACTORS:
        level_dir = root / "datasets" / f"{DATASET_NAME}_f{factor}_{views}v"
        level_geometry = (
            None
            if carried_geometry is None or carried_factor is None
            else _rescale_geometry(carried_geometry, from_factor=carried_factor, to_factor=factor)
        )
        _write_downsampled_sidecar(
            source_dir=full_dataset_dir,
            output_dir=level_dir,
            factor=factor,
            carried_geometry=level_geometry,
        )
        size = 128 // factor
        initial_volume_path = None
        if carried_volume_path is not None and carried_volume_factor is not None:
            initial_volume_path = root / f"carried_initial_f{factor}_{size}_{views}v.npy"
            previous = np.load(carried_volume_path)
            upsampled = upsample_volume(
                jnp.asarray(previous, dtype=jnp.float32),
                factor=carried_volume_factor // factor,
                target_shape=(size, size, size),
            )
            np.save(initial_volume_path, np.asarray(upsampled, dtype=np.float32))
        row = _run_solver_inprocess(
            root=root,
            run_name=f"stopped_otsu_l2_multires_f{factor}_{size}_{views}v",
            dataset_dir=level_dir,
            size=size,
            views=views,
            profile=profile,
            volume_source="stopped_reconstruction",
            initial_volume_path=initial_volume_path,
            preview_volume_support=preview_volume_support,
            preview_support_outside_weight=preview_support_outside_weight,
            preview_low_frequency_anchor_weight=preview_low_frequency_anchor_weight,
            preview_det_u_gauge_mode_weight=preview_det_u_gauge_mode_weight,
        )
        rows.append(row)
        final_pose = read_pose_params_csv(root / row["run_name"] / "pose_params.csv")
        carried_geometry = read_geometry_json(
            root / row["run_name"] / "geometry_final.json",
            final_pose,
        )
        carried_factor = factor
        carried_volume_path = root / row["run_name"] / "final_volume.npy"
        carried_volume_factor = factor
    return rows


def _run_solver_inprocess(
    *,
    root: Path,
    run_name: str,
    dataset_dir: Path,
    size: int,
    views: int,
    profile: str,
    volume_source: str,
    initial_volume_path: Path | None = None,
    preview_volume_support: str = "cylindrical",
    preview_support_outside_weight: float = 0.0,
    preview_low_frequency_anchor_weight: float = 0.0,
    preview_det_u_gauge_mode_weight: float = 0.0,
) -> dict[str, Any]:
    run_dir = root / run_name
    start = time.perf_counter()
    schedule = reference_continuation_schedule(cast("Any", profile))
    sidecars = load_synthetic_dataset_sidecars(dataset_dir)
    recovery_tolerances = sidecars.manifest.get("recovery_tolerances", {})
    unsupported_dofs = sidecars.manifest.get("unsupported_dofs_not_evaluated", [])
    config = AlternatingSmokeConfig(
        seed=17,
        size=size,
        n_views=views,
        schedule=schedule,
        projection_loss_mode="otsu_l2",
        geometry_update_volume_source=cast("Any", volume_source),
        geometry_update_solver="joint_schur",
        geometry_update_pose_frozen=True,
        geometry_update_active_setup_parameters=("det_u_px",),
        geometry_update_active_pose_dofs=(),
        preview_volume_support=cast("Any", preview_volume_support),
        preview_initialization="backprojection",
        preview_initial_volume_path=initial_volume_path,
        preview_tv_scale=1.0,
        preview_residual_filter_mode="continuation",
        preview_center_l2_weight=0.02,
        preview_support_outside_weight=max(float(preview_support_outside_weight), 0.0),
        preview_low_frequency_anchor_weight=max(float(preview_low_frequency_anchor_weight), 0.0),
        preview_det_u_gauge_mode_weight=max(float(preview_det_u_gauge_mode_weight), 0.0),
        preview_views_per_batch=0,
        synthetic_dataset_name=DATASET_NAME,
        synthetic_dataset_artifact_dir=dataset_dir,
        synthetic_dataset_nuisance_applied=False,
        synthetic_dataset_sidecar_readback={
            "validated": True,
            "source": "tomojax.datasets.load_synthetic_dataset_sidecars",
            "n_views": sidecars.true_geometry.pose.n_views,
            "consistency": sidecars.consistency.to_dict(),
            "recovery_tolerances": recovery_tolerances,
            "unsupported_dofs_not_evaluated": unsupported_dofs,
        },
    )
    _ = AlternatingAlignmentSolver(config).run_smoke(run_dir)
    elapsed = time.perf_counter() - start
    return _summary_row(run_name, run_dir, elapsed)


def _write_downsampled_sidecar(
    *,
    source_dir: Path,
    output_dir: Path,
    factor: int,
    carried_geometry: GeometryState | None,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    volume = np.load(output_dir / "ground_truth_volume.npy")
    if factor != 1:
        volume = np.asarray(bin_volume(jnp.asarray(volume), factor), dtype=np.float32)
    np.save(output_dir / "ground_truth_volume.npy", volume.astype(np.float32))

    _scale_geometry_artifacts(
        output_dir=output_dir,
        factor=factor,
        carried_geometry=carried_geometry,
    )
    true_pose = read_pose_params_csv(output_dir / "v2_true_pose_params.csv")
    true_geometry = read_geometry_json(output_dir / "v2_true_geometry.json", true_pose)
    projections = _project_level_projections(volume, true_geometry)
    mask = np.ones(projections.shape, dtype=bool)
    np.save(output_dir / "projections.npy", projections.astype(np.float32))
    np.save(output_dir / "mask.npy", mask)
    manifest["volume_shape"] = [int(dim) for dim in volume.shape]
    manifest["detector_shape"] = [int(projections.shape[1]), int(projections.shape[2])]
    manifest["views"] = int(projections.shape[0])
    manifest["multires_factor"] = int(factor)
    manifest["multires_source_dataset"] = str(source_dir)
    manifest["artifact_contract"] = "tomojax-v2.synthetic-dataset.multires-forward-consistent.v1"
    manifest["coarse_projection_policy"] = (
        "project_binned_volume_with_level_true_geometry; do not use binned full-resolution "
        "projections as oracle evidence"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scale_geometry_artifacts(
    *,
    output_dir: Path,
    factor: int,
    carried_geometry: GeometryState | None,
) -> None:
    for stem in ("nominal", "true"):
        pose = read_pose_params_csv(output_dir / f"v2_{stem}_pose_params.csv")
        state = read_geometry_json(output_dir / f"v2_{stem}_geometry.json", pose)
        scaled = _rescale_geometry(state, from_factor=1, to_factor=factor)
        write_geometry_json(output_dir / f"v2_{stem}_geometry.json", scaled)
        write_pose_params_csv(output_dir / f"v2_{stem}_pose_params.csv", scaled.pose)
    if carried_geometry is None:
        pose = read_pose_params_csv(output_dir / "v2_corrupted_pose_params.csv")
        state = read_geometry_json(output_dir / "v2_corrupted_geometry.json", pose)
        scaled = _rescale_geometry(state, from_factor=1, to_factor=factor)
    else:
        scaled = carried_geometry
    write_geometry_json(output_dir / "v2_corrupted_geometry.json", scaled)
    write_pose_params_csv(output_dir / "v2_corrupted_pose_params.csv", scaled.pose)


def _rescale_geometry(
    state: GeometryState,
    *,
    from_factor: int,
    to_factor: int,
) -> GeometryState:
    ratio = float(from_factor) / float(to_factor)
    setup = state.setup
    setup = setup.replace_parameter(
        "det_u_px",
        setup.det_u_px.with_value(setup.det_u_px.value * ratio),
    )
    setup = setup.replace_parameter(
        "det_v_px",
        setup.det_v_px.with_value(setup.det_v_px.value * ratio),
    )
    pose = state.pose
    scaled_pose = PoseParameters(
        alpha_rad=pose.alpha_rad,
        beta_rad=pose.beta_rad,
        theta_nominal_rad=pose.theta_nominal_rad,
        phi_residual_rad=pose.phi_residual_rad,
        dx_px=pose.dx_px * ratio,
        dz_px=pose.dz_px * ratio,
    )
    return GeometryState(setup=setup, pose=scaled_pose, acquisition=state.acquisition)


def _project_level_projections(
    volume: np.ndarray,
    geometry: GeometryState,
    *,
    views_per_chunk: int = 16,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, geometry.pose.n_views, int(views_per_chunk)):
        stop = min(geometry.pose.n_views, start + int(views_per_chunk))
        pose = PoseParameters(
            alpha_rad=geometry.pose.alpha_rad[start:stop],
            beta_rad=geometry.pose.beta_rad[start:stop],
            theta_nominal_rad=geometry.pose.theta_nominal_rad[start:stop],
            phi_residual_rad=geometry.pose.phi_residual_rad[start:stop],
            dx_px=geometry.pose.dx_px[start:stop],
            dz_px=geometry.pose.dz_px[start:stop],
        )
        state = GeometryState(setup=geometry.setup, pose=pose, acquisition=geometry.acquisition)
        chunks.append(
            np.asarray(
                project_parallel_reference(jnp.asarray(volume, dtype=jnp.float32), state),
                dtype=np.float32,
            )
        )
    return np.concatenate(chunks, axis=0)


def _summary_row(run_name: str, run_dir: Path, elapsed: float) -> dict[str, Any]:
    result = json.loads((run_dir / "benchmark_result.json").read_text(encoding="utf-8"))
    geometry = result["geometry_recovery"]
    reconstruction = result["reconstruction"]
    runtime = result["runtime"]
    sidecar = result.get("dataset", {})
    schur = json.loads((run_dir / "schur_diagnostics.json").read_text(encoding="utf-8"))
    diagnostics = schur.get("diagnostics", {})
    return {
        "run_name": run_name,
        "status": result.get("status"),
        "projection_loss_mode": result.get("projection_loss_mode"),
        "geometry_update_volume_source": result.get("geometry_update_volume_source"),
        "views": sidecar.get("projection_views"),
        "volume_shape": "x".join(str(dim) for dim in sidecar.get("volume_shape", [])),
        "artifact_dir": str(run_dir),
        "det_u_realized_rmse_px": geometry.get("det_u_realized_rmse_px"),
        "det_v_realized_rmse_px": geometry.get("det_v_realized_rmse_px"),
        "theta_realized_rmse_rad": geometry.get("theta_realized_rmse_rad"),
        "final_volume_final_geometry_loss": reconstruction.get(
            "final_volume_final_geometry_loss_all_views"
        ),
        "final_volume_true_geometry_loss": reconstruction.get(
            "final_volume_true_geometry_loss_all_views"
        ),
        "true_volume_final_geometry_loss": reconstruction.get(
            "true_volume_final_geometry_loss_all_views"
        ),
        "true_volume_true_geometry_loss": reconstruction.get(
            "true_volume_true_geometry_loss_all_views"
        ),
        "volume_nmse": reconstruction.get("volume_nmse"),
        "schur_accepted": diagnostics.get("accepted"),
        "classification": reconstruction.get("projection_loss_classification"),
        "runtime_seconds": runtime.get("total_wall_seconds", elapsed),
        "peak_memory": None,
    }


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    _write_multires_carried_detu_landscape(root, rows)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Rich PHANTOM94 v1-Parity Gate",
        "",
        "| Run | Source | Shape | det_u RMSE px | Volume NMSE | Schur accepted | Classification |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    lines.extend(
        (
            "| {run_name} | {geometry_update_volume_source} | {volume_shape} | "
            "{det_u_realized_rmse_px} | {volume_nmse} | {schur_accepted} | "
            "{classification} |".format(**row)
        )
        for row in rows
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_multires_carried_detu_landscape(root: Path, rows: list[dict[str, Any]]) -> None:
    curve_rows = _multires_carried_detu_rows(root, rows)
    if not curve_rows:
        return
    csv_path = root / "multires_carried_detu_loss_curves.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    summary = _multires_carried_detu_summary(curve_rows)
    (root / "multires_carried_detu_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_multires_carried_detu_markdown(root / "multires_carried_detu_summary.md", summary)


def _multires_carried_detu_rows(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    carried_rows: list[dict[str, Any]] = []
    for row in rows:
        run_name = str(row["run_name"])
        curve_path = root / run_name / "detu_loss_curves.csv"
        if not curve_path.exists():
            continue
        factor = _factor_from_run_name(run_name)
        with curve_path.open("r", newline="", encoding="utf-8") as handle:
            for curve_row in csv.DictReader(handle):
                if curve_row.get("volume_source") != "final_stopped_volume":
                    continue
                carried = dict(curve_row)
                carried["volume_source"] = f"multires_carried_f{factor}_final_volume"
                carried["source_run_name"] = run_name
                carried["multires_factor"] = factor
                carried["volume_shape"] = row.get("volume_shape")
                carried["artifact_dir"] = row.get("artifact_dir")
                carried_rows.append(carried)
    return carried_rows


def _multires_carried_detu_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["volume_source"]), []).append(row)
    curves: list[dict[str, Any]] = []
    for source, source_rows in sorted(by_source.items()):
        argmin = min(source_rows, key=lambda item: float(item["loss"]))
        curves.append(
            {
                "volume_source": source,
                "source_run_name": argmin["source_run_name"],
                "multires_factor": int(argmin["multires_factor"]),
                "volume_shape": argmin["volume_shape"],
                "argmin_det_u_px": float(argmin["det_u_px"]),
                "argmin_loss": float(argmin["loss"]),
                "mask_role": argmin.get("mask_role"),
                "loss_mode": argmin.get("loss_mode"),
            }
        )
    return {
        "schema": "tomojax.multires_carried_detu_landscape.v1",
        "status": "recorded",
        "purpose": (
            "diagnostic_multires_carried_fixed_volume_landscape_not_production_center_search"
        ),
        "curve_count": len(curves),
        "curves": curves,
    }


def _write_multires_carried_detu_markdown(path: Path, summary: dict[str, Any]) -> None:
    curves = cast("list[dict[str, Any]]", summary["curves"])
    lines = [
        "# Multires-Carried det_u Landscapes",
        "",
        "| Source | Run | Shape | Argmin det_u px | Argmin loss |",
        "|---|---|---:|---:|---:|",
    ]
    lines.extend(
        (
            "| {volume_source} | {source_run_name} | {volume_shape} | "
            "{argmin_det_u_px} | {argmin_loss} |".format(**curve)
        )
        for curve in curves
    )
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _factor_from_run_name(run_name: str) -> int:
    for part in run_name.split("_"):
        if part.startswith("f") and part[1:].isdigit():
            return int(part[1:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
