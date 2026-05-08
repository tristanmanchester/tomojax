"""Run the narrow rich PHANTOM94 v1-parity setup/global gate."""
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from tomojax.core.multires import bin_projections, bin_volume
from tomojax.datasets import generate_synthetic_dataset, load_synthetic_dataset_sidecars
from tomojax.geometry import (
    GeometryState,
    PoseParameters,
    read_geometry_json,
    read_pose_params_csv,
    write_geometry_json,
    write_pose_params_csv,
)

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
    args = parser.parse_args()

    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    full = generate_synthetic_dataset(
        "rich_phantom94_setup_global_tomo",
        root / "datasets_source",
        size=128,
        clean=True,
        views=args.views,
        supported_only=True,
    )
    if args.mode == "fixed_truth":
        rows = [
            _run_align_auto(
                root=root,
                run_name=f"fixed_truth_otsu_l2_{args.profile}_{args.views}v",
                dataset_dir=full.dataset_dir,
                size=128,
                views=args.views,
                profile=args.profile,
                volume_source="fixed_synthetic_truth",
            )
        ]
    else:
        rows = _run_stopped_multires(
            root=root,
            full_dataset_dir=full.dataset_dir,
            views=args.views,
            profile=args.profile,
        )
    _write_summary(root, rows)
    return 0


def _run_stopped_multires(
    *,
    root: Path,
    full_dataset_dir: Path,
    views: int,
    profile: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    carried_geometry: GeometryState | None = None
    carried_factor: int | None = None
    for factor in FACTORS:
        level_dir = root / "datasets" / f"rich_phantom94_setup_global_tomo_f{factor}_{views}v"
        _write_downsampled_sidecar(
            source_dir=full_dataset_dir,
            output_dir=level_dir,
            factor=factor,
            carried_geometry=None
            if carried_geometry is None or carried_factor is None
            else _rescale_geometry(carried_geometry, from_factor=carried_factor, to_factor=factor),
        )
        size = 128 // factor
        row = _run_align_auto(
            root=root,
            run_name=f"stopped_otsu_l2_multires_f{factor}_{size}_{views}v",
            dataset_dir=level_dir,
            size=size,
            views=views,
            profile=profile,
            volume_source="stopped_reconstruction",
        )
        rows.append(row)
        sidecars = load_synthetic_dataset_sidecars(level_dir)
        final_pose = read_pose_params_csv(root / row["run_name"] / "pose_params.csv")
        carried_geometry = read_geometry_json(
            root / row["run_name"] / "geometry_final.json",
            final_pose,
        )
        carried_factor = factor
        _ = sidecars
    return rows


def _run_align_auto(
    *,
    root: Path,
    run_name: str,
    dataset_dir: Path,
    size: int,
    views: int,
    profile: str,
    volume_source: str,
) -> dict[str, Any]:
    run_dir = root / run_name
    start = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tomojax.cli.align_auto",
            "--out-dir",
            str(run_dir),
            "--profile",
            profile,
            "--size",
            str(size),
            "--views",
            str(views),
            "--synthetic-dataset",
            "rich_phantom94_setup_global_tomo",
            "--synthetic-dataset-dir",
            str(dataset_dir),
            "--projection-loss-mode",
            "otsu_l2",
            "--geometry-update-volume-source",
            volume_source,
            "--geometry-update-pose-frozen",
            "--geometry-update-active-setup-parameters",
            "det_u_px",
            "--preview-volume-support",
            "cylindrical",
            "--preview-initialization",
            "backprojection",
            "--preview-tv-scale",
            "1.0",
            "--preview-residual-filter-mode",
            "continuation",
            "--preview-center-l2-weight",
            "0.02",
        ],
        check=False,
        env=_jax_subprocess_env(),
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"align-auto failed for {run_name}")
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
    projections = np.load(output_dir / "projections.npy")
    mask = np.load(output_dir / "mask.npy")
    if factor != 1:
        volume = np.asarray(bin_volume(jnp.asarray(volume), factor), dtype=np.float32)
        projections = np.asarray(
            bin_projections(jnp.asarray(projections), factor),
            dtype=np.float32,
        )
        mask = np.asarray(bin_projections(jnp.asarray(mask.astype(np.float32)), factor) > 0.5)
    np.save(output_dir / "ground_truth_volume.npy", volume.astype(np.float32))
    np.save(output_dir / "projections.npy", projections.astype(np.float32))
    np.save(output_dir / "mask.npy", mask.astype(bool))

    _scale_geometry_artifacts(
        output_dir=output_dir,
        factor=factor,
        carried_geometry=carried_geometry,
    )
    manifest["volume_shape"] = [int(dim) for dim in volume.shape]
    manifest["detector_shape"] = [int(projections.shape[1]), int(projections.shape[2])]
    manifest["views"] = int(projections.shape[0])
    manifest["multires_factor"] = int(factor)
    manifest["multires_source_dataset"] = str(source_dir)
    manifest["artifact_contract"] = "tomojax-v2.synthetic-dataset.multires-sidecar.v1"
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


def _jax_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
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


if __name__ == "__main__":
    raise SystemExit(main())
