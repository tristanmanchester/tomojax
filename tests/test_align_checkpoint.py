from __future__ import annotations

import json
import numpy as np
import pytest

import jax.numpy as jnp

from tomojax.align.checkpoint import (
    CheckpointError,
    build_alignment_checkpoint_metadata,
    load_alignment_checkpoint,
    save_alignment_checkpoint,
    validate_alignment_checkpoint,
)
from tomojax.align.pipeline import (
    AlignConfig,
    AlignMultiresResumeState,
    AlignResumeState,
    align,
    align_multires,
)
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon.multires import scale_detector, scale_grid

from test_align_quick import make_misaligned_case


def _metadata(
    *,
    cfg: AlignConfig,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    levels: list[int] | None = None,
    state_grid: Grid | None = None,
    state_detector: Detector | None = None,
    level_index: int = 0,
    level_factor: int = 1,
    completed_outer_iters_in_level: int = 0,
    global_outer_iters_completed: int = 0,
    prev_factor: int | None = None,
    L_prev: float | None = None,
    small_impr_streak: int = 0,
    elapsed_offset: float = 0.0,
    level_complete: bool = False,
    run_complete: bool = False,
) -> dict[str, object]:
    return build_alignment_checkpoint_metadata(
        projections_shape=tuple(int(v) for v in projections.shape),
        projections_dtype=str(projections.dtype),
        geometry_type="parallel",
        geometry_meta={},
        reconstruction_grid=grid.to_dict(),
        detector=detector.to_dict(),
        state_grid=(state_grid or grid).to_dict(),
        state_detector=(state_detector or detector).to_dict(),
        levels=levels,
        level_index=level_index,
        level_factor=level_factor,
        completed_outer_iters_in_level=completed_outer_iters_in_level,
        global_outer_iters_completed=global_outer_iters_completed,
        config=cfg,
        cli_options={
            "roi": "off",
            "grid": None,
            "requested_gather_dtype": "fp32",
            "gather_dtype": "fp32",
            "views_per_batch": 1,
            "projector_unroll": 1,
            "checkpoint_projector": True,
            "mask_vol": "off",
        },
        prev_factor=prev_factor,
        L_prev=L_prev,
        small_impr_streak=small_impr_streak,
        elapsed_offset=elapsed_offset,
        random_state={"alignment": None},
        level_complete=level_complete,
        run_complete=run_complete,
    )


def _resume_single_from_checkpoint(path, expected_metadata) -> AlignResumeState:
    checkpoint = load_alignment_checkpoint(path)
    validate_alignment_checkpoint(checkpoint, expected_metadata)
    meta = checkpoint.metadata
    return AlignResumeState(
        x=jnp.asarray(checkpoint.x, dtype=jnp.float32),
        params5=jnp.asarray(checkpoint.params5, dtype=jnp.float32),
        motion_coeffs=(
            None
            if checkpoint.motion_coeffs is None
            else jnp.asarray(checkpoint.motion_coeffs, dtype=jnp.float32)
        ),
        start_outer_iter=int(meta["completed_outer_iters_in_level"]),
        loss=list(checkpoint.loss_history),
        outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
        L=meta.get("L_prev"),
        small_impr_streak=int(meta.get("small_impr_streak", 0)),
        elapsed_offset=float(meta.get("elapsed_offset", 0.0)),
    )


def _resume_multires_from_checkpoint(path, expected_metadata) -> AlignMultiresResumeState:
    checkpoint = load_alignment_checkpoint(path)
    validate_alignment_checkpoint(checkpoint, expected_metadata)
    meta = checkpoint.metadata
    return AlignMultiresResumeState(
        x=jnp.asarray(checkpoint.x, dtype=jnp.float32),
        params5=jnp.asarray(checkpoint.params5, dtype=jnp.float32),
        motion_coeffs=(
            None
            if checkpoint.motion_coeffs is None
            else jnp.asarray(checkpoint.motion_coeffs, dtype=jnp.float32)
        ),
        level_index=int(meta["level_index"]),
        level_factor=int(meta["level_factor"]),
        completed_outer_iters_in_level=int(meta["completed_outer_iters_in_level"]),
        global_outer_iters_completed=int(meta["global_outer_iters_completed"]),
        prev_factor=None if meta.get("prev_factor") is None else int(meta["prev_factor"]),
        loss=list(checkpoint.loss_history),
        outer_stats=[dict(stat) for stat in checkpoint.outer_stats],
        L=meta.get("L_prev"),
        small_impr_streak=int(meta.get("small_impr_streak", 0)),
        elapsed_offset=float(meta.get("elapsed_offset", 0.0)),
        level_complete=bool(meta.get("level_complete", False)),
        run_complete=bool(meta.get("run_complete", False)),
    )


def test_alignment_checkpoint_round_trips_arrays_and_metadata(tmp_path):
    grid = Grid(nx=3, ny=3, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=2, du=1.0, dv=1.0)
    projections = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    cfg = AlignConfig(outer_iters=2, recon_iters=1, early_stop=False)
    metadata = _metadata(
        cfg=cfg,
        grid=grid,
        detector=detector,
        projections=projections,
        completed_outer_iters_in_level=1,
        global_outer_iters_completed=1,
        L_prev=2.5,
        small_impr_streak=1,
    )

    path = tmp_path / "align_checkpoint.npz"
    save_alignment_checkpoint(
        path,
        x=np.ones((3, 3, 2), dtype=np.float32),
        params5=np.zeros((4, 5), dtype=np.float32),
        motion_coeffs=None,
        loss_history=[3.0, 2.0],
        outer_stats=[{"outer_idx": 1, "loss_after": 2.0}],
        metadata=metadata,
    )

    checkpoint = load_alignment_checkpoint(path)
    validate_alignment_checkpoint(checkpoint, metadata)

    assert checkpoint.x.shape == (3, 3, 2)
    assert checkpoint.params5.shape == (4, 5)
    assert checkpoint.motion_coeffs is None
    assert checkpoint.loss_history == [3.0, 2.0]
    assert checkpoint.outer_stats[0]["outer_idx"] == 1
    assert checkpoint.metadata["completed_outer_iters_in_level"] == 1


def test_alignment_checkpoint_accepts_missing_recon_solver_defaults(tmp_path):
    grid = Grid(nx=3, ny=3, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=2, du=1.0, dv=1.0)
    projections = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    cfg = AlignConfig(outer_iters=2, recon_iters=1, early_stop=False)
    metadata = _metadata(cfg=cfg, grid=grid, detector=detector, projections=projections)
    legacy_metadata = dict(metadata)
    legacy_metadata["config"] = dict(legacy_metadata["config"])
    legacy_metadata["cli_options"] = dict(legacy_metadata["cli_options"])
    for key in (
        "recon_algo",
        "recon_positivity",
        "spdhg_seed",
        "lbfgs_maxiter",
        "lbfgs_ftol",
        "lbfgs_gtol",
        "lbfgs_maxls",
        "lbfgs_memory_size",
    ):
        legacy_metadata["config"].pop(key, None)
        legacy_metadata["cli_options"].pop(key, None)

    path = tmp_path / "legacy_align_checkpoint.npz"
    save_alignment_checkpoint(
        path,
        x=np.ones((3, 3, 2), dtype=np.float32),
        params5=np.zeros((4, 5), dtype=np.float32),
        loss_history=[],
        outer_stats=[],
        metadata=legacy_metadata,
    )

    checkpoint = load_alignment_checkpoint(path)
    validate_alignment_checkpoint(checkpoint, metadata)


def test_alignment_checkpoint_accepts_legacy_missing_motion_coeffs(tmp_path):
    grid = Grid(nx=3, ny=3, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=2, du=1.0, dv=1.0)
    projections = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    cfg = AlignConfig(outer_iters=2, recon_iters=1, early_stop=False)
    metadata = _metadata(cfg=cfg, grid=grid, detector=detector, projections=projections)
    legacy_metadata = dict(metadata)
    legacy_metadata.pop("has_motion_coeffs", None)

    path = tmp_path / "legacy_no_motion_coeffs.npz"
    np.savez_compressed(
        path,
        x=np.ones((3, 3, 2), dtype=np.float32),
        params5=np.zeros((4, 5), dtype=np.float32),
        loss_history=np.asarray([1.0], dtype=np.float64),
        metadata_json=np.asarray(json.dumps(legacy_metadata, allow_nan=False)),
        outer_stats_json=np.asarray("[]"),
    )

    checkpoint = load_alignment_checkpoint(path)
    validate_alignment_checkpoint(checkpoint, metadata)
    assert checkpoint.motion_coeffs is None


def test_alignment_checkpoint_reports_corrupt_file(tmp_path):
    path = tmp_path / "bad_checkpoint.npz"
    path.write_text("not an npz", encoding="utf-8")

    with pytest.raises(CheckpointError, match="could not read checkpoint"):
        load_alignment_checkpoint(path)


def test_alignment_checkpoint_rejects_incompatible_projection_shape(tmp_path):
    grid = Grid(nx=3, ny=3, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=2, du=1.0, dv=1.0)
    projections = jnp.zeros((4, 2, 3), dtype=jnp.float32)
    cfg = AlignConfig(outer_iters=2, recon_iters=1, early_stop=False)
    metadata = _metadata(cfg=cfg, grid=grid, detector=detector, projections=projections)
    path = tmp_path / "align_checkpoint.npz"
    save_alignment_checkpoint(
        path,
        x=np.zeros((3, 3, 2), dtype=np.float32),
        params5=np.zeros((4, 5), dtype=np.float32),
        loss_history=[],
        outer_stats=[],
        metadata=metadata,
    )

    checkpoint = load_alignment_checkpoint(path)
    expected = dict(metadata)
    expected["projection_shape"] = [5, 2, 3]

    with pytest.raises(CheckpointError, match="projection shape"):
        validate_alignment_checkpoint(checkpoint, expected)


def test_align_single_level_resume_matches_uninterrupted_run(tmp_path):
    grid, detector, geom, _, projections, _ = make_misaligned_case(5, 5, 5, 5, 11)
    cfg = AlignConfig(
        outer_iters=2,
        recon_iters=1,
        lambda_tv=0.0,
        opt_method="gd",
        lr_rot=5e-3,
        lr_trans=1e-1,
        early_stop=False,
    )
    x_full, params_full, info_full = align(geom, grid, detector, projections, cfg=cfg)

    checkpoint_path = tmp_path / "single_resume.npz"
    expected_metadata = _metadata(cfg=cfg, grid=grid, detector=detector, projections=projections)

    def checkpoint_callback(state: AlignResumeState) -> None:
        if state.start_outer_iter != 1:
            return
        metadata = _metadata(
            cfg=cfg,
            grid=grid,
            detector=detector,
            projections=projections,
            completed_outer_iters_in_level=state.start_outer_iter,
            global_outer_iters_completed=state.start_outer_iter,
            L_prev=state.L,
            small_impr_streak=state.small_impr_streak,
            elapsed_offset=state.elapsed_offset,
        )
        save_alignment_checkpoint(
            checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )

    def stop_after_first_outer(_x, _params, stat):
        return "stop_run" if int(stat["outer_idx"]) == 1 else "continue"

    align(
        geom,
        grid,
        detector,
        projections,
        cfg=cfg,
        observer=stop_after_first_outer,
        checkpoint_callback=checkpoint_callback,
    )
    resume_state = _resume_single_from_checkpoint(checkpoint_path, expected_metadata)
    x_resume, params_resume, info_resume = align(
        geom,
        grid,
        detector,
        projections,
        cfg=cfg,
        resume_state=resume_state,
    )

    np.testing.assert_allclose(np.asarray(x_resume), np.asarray(x_full), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(params_resume),
        np.asarray(params_full),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(info_resume["loss"], info_full["loss"], rtol=1e-5, atol=1e-5)


def test_align_multires_resume_after_level_checkpoint_matches_uninterrupted_run(tmp_path):
    grid, detector, geom, _, projections, _ = make_misaligned_case(6, 6, 6, 6, 12)
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        lambda_tv=0.0,
        opt_method="gd",
        lr_rot=5e-3,
        lr_trans=1e-1,
        early_stop=False,
    )
    factors = [2, 1]
    x_full, params_full, info_full = align_multires(
        geom,
        grid,
        detector,
        projections,
        factors=factors,
        cfg=cfg,
    )

    checkpoint_path = tmp_path / "multires_resume.npz"
    expected_metadata = _metadata(
        cfg=cfg,
        grid=grid,
        detector=detector,
        projections=projections,
        levels=factors,
    )

    def checkpoint_callback(state: AlignMultiresResumeState) -> None:
        if not state.level_complete or state.level_index != 0:
            return
        state_grid = scale_grid(grid, state.level_factor)
        state_detector = scale_detector(detector, state.level_factor)
        metadata = _metadata(
            cfg=cfg,
            grid=grid,
            detector=detector,
            projections=projections,
            levels=factors,
            state_grid=state_grid,
            state_detector=state_detector,
            level_index=state.level_index,
            level_factor=state.level_factor,
            completed_outer_iters_in_level=state.completed_outer_iters_in_level,
            global_outer_iters_completed=state.global_outer_iters_completed,
            prev_factor=state.prev_factor,
            L_prev=state.L,
            small_impr_streak=state.small_impr_streak,
            elapsed_offset=state.elapsed_offset,
            level_complete=state.level_complete,
        )
        save_alignment_checkpoint(
            checkpoint_path,
            x=state.x,
            params5=state.params5,
            motion_coeffs=state.motion_coeffs,
            loss_history=state.loss,
            outer_stats=state.outer_stats,
            metadata=metadata,
        )

    def stop_after_first_global_outer(_x, _params, stat):
        return "stop_run" if int(stat["global_outer_idx"]) == 1 else "continue"

    align_multires(
        geom,
        grid,
        detector,
        projections,
        factors=factors,
        cfg=cfg,
        observer=stop_after_first_global_outer,
        checkpoint_callback=checkpoint_callback,
    )

    resume_state = _resume_multires_from_checkpoint(checkpoint_path, expected_metadata)
    x_resume, params_resume, info_resume = align_multires(
        geom,
        grid,
        detector,
        projections,
        factors=factors,
        cfg=cfg,
        resume_state=resume_state,
    )

    np.testing.assert_allclose(np.asarray(x_resume), np.asarray(x_full), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(params_resume),
        np.asarray(params_full),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(info_resume["loss"], info_full["loss"], rtol=1e-5, atol=1e-5)
