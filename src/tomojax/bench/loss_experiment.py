from __future__ import annotations

"""Shared loss-benchmark helpers.

This module is the reusable boundary between user-facing loss-benchmark orchestration
and controller-specific benchmark harness code. Keep stable dataset, misalignment,
and metric helpers here instead of duplicating them across ``bench/`` and ``scripts/``.
"""

import os
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from ..align.parametrizations import se3_from_5d
from ..core.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from ..core.geometry.views import stack_view_poses
from ..core.projector import forward_project_view_T, get_detector_grid_device
from ..data.geometry_meta import build_geometry_from_meta
from ..data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo
from ..data.simulate import SimConfig, simulate_to_file


def make_gt_dataset(
    expdir: str,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    n_views: int,
    geometry: str,
    seed: int,
) -> str:
    """Create or reuse the benchmark ground-truth dataset for an experiment."""
    gt_path = os.path.join(expdir, "gt.nxs")
    if os.path.exists(gt_path):
        return gt_path
    cfg = SimConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        nv=nv,
        n_views=n_views,
        geometry=geometry,
        phantom="shepp",
        rotation_deg=None,
        seed=seed,
    )
    simulate_to_file(cfg, gt_path)
    return gt_path


def make_misaligned_dataset(
    expdir: str, gt_path: str, *, rot_deg: float, trans_px: float, seed: int
) -> str:
    """Create or reuse a misaligned benchmark dataset derived from the GT volume."""
    mis_path = os.path.join(expdir, "misaligned.nxs")
    if os.path.exists(mis_path):
        return mis_path

    meta = load_nxtomo(gt_path)
    volume = np.asarray(meta.volume, dtype=np.float32)
    grid, det, geom = build_geometry_from_meta(
        meta.geometry_inputs(),
        apply_saved_alignment=False,
        volume_shape=volume.shape,
    )

    thetas = np.asarray(meta.thetas_deg, dtype=np.float32)
    n_views = len(thetas)
    rng = np.random.default_rng(seed)
    rot_scale = np.deg2rad(float(rot_deg))

    params5 = np.zeros((n_views, 5), np.float32)
    params5[:, 0] = rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
    params5[:, 1] = rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
    params5[:, 2] = rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
    params5[:, 3] = rng.uniform(-float(trans_px), float(trans_px), n_views).astype(
        np.float32
    ) * float(det.du)
    params5[:, 4] = rng.uniform(-float(trans_px), float(trans_px), n_views).astype(
        np.float32
    ) * float(det.dv)

    T_nom = stack_view_poses(geom, n_views)
    T_aug = T_nom @ jax.vmap(se3_from_5d)(jnp.asarray(params5, jnp.float32))
    det_grid = get_detector_grid_device(det)
    vm_project = jax.vmap(
        lambda T: forward_project_view_T(
            T,
            grid,
            det,
            jnp.asarray(volume, jnp.float32),
            use_checkpoint=True,
            det_grid=det_grid,
        ),
        in_axes=0,
    )
    projections = vm_project(T_aug).astype(jnp.float32)

    save_meta = meta.copy_metadata()
    save_meta.thetas_deg = thetas
    save_meta.grid = grid.to_dict()
    save_meta.detector = det.to_dict()
    save_meta.geometry_type = "parallel" if isinstance(geom, ParallelGeometry) else "lamino"
    save_meta.volume = volume
    save_meta.align_params = np.asarray(params5)
    save_meta.frame = str(meta.frame or "sample")
    save_nxtomo(
        mis_path,
        projections=np.asarray(projections),
        metadata=save_meta,
    )
    return mis_path


def metrics_abs(
    params_true: np.ndarray, params_est: np.ndarray, du: float, dv: float
) -> Dict[str, float]:
    """Absolute parameter RMSE/MAE in degrees and pixels."""
    assert params_true.shape == params_est.shape
    d = params_est - params_true
    d_deg = np.rad2deg(d[:, :3])
    dx_px = d[:, 3] / max(1e-12, float(du))
    dz_px = d[:, 4] / max(1e-12, float(dv))
    trans_err = np.stack([dx_px, dz_px], axis=1)
    rot_rmse = float(np.sqrt(np.mean(d_deg**2)))
    trans_rmse = float(np.sqrt(np.mean(trans_err**2)))
    rot_mse = float(np.mean(d_deg**2))
    trans_mse = float(np.mean(trans_err**2))
    rot_mae = float(np.mean(np.abs(d_deg)))
    trans_mae = float(np.mean(np.abs(trans_err)))
    return {
        "rot_rmse_deg": rot_rmse,
        "trans_rmse_px": trans_rmse,
        "rot_mse_deg": rot_mse,
        "trans_mse_px": trans_mse,
        "rot_mae_deg": rot_mae,
        "trans_mae_px": trans_mae,
    }


def _so3_geodesic_deg(R: np.ndarray) -> float:
    tr = np.clip((np.trace(R[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti


def _params_to_T(params: np.ndarray) -> np.ndarray:
    p_j = jnp.asarray(params, jnp.float32)
    return np.asarray(jax.vmap(se3_from_5d)(p_j))


def metrics_relative(
    params_true: np.ndarray,
    params_est: np.ndarray,
    du: float,
    dv: float,
    *,
    k_step: int = 1,
) -> Dict[str, float]:
    """Gauge-invariant relative-motion error over k-step pose differences."""
    assert params_true.shape == params_est.shape
    if params_true.shape[0] <= k_step:
        return {"rot_rel_rmse_deg": float("nan"), "trans_rel_rmse_px": float("nan")}

    Tt = _params_to_T(params_true)
    Te = _params_to_T(params_est)
    rot_errs: List[float] = []
    trans_errs_sq: List[float] = []
    for i in range(params_true.shape[0] - k_step):
        Dt = Tt[i + k_step] @ _inv_T(Tt[i])
        De = Te[i + k_step] @ _inv_T(Te[i])
        E = _inv_T(Dt) @ De
        rot_errs.append(_so3_geodesic_deg(E))
        ex = float(E[0, 3]) / max(1e-12, float(du))
        ez = float(E[2, 3]) / max(1e-12, float(dv))
        trans_errs_sq.append(ex * ex + ez * ez)

    rot_rel_rmse = float(np.sqrt(np.mean(np.square(rot_errs)))) if rot_errs else float("nan")
    trans_rel_rmse = float(np.sqrt(np.mean(trans_errs_sq))) if trans_errs_sq else float("nan")
    return {
        "rot_rel_rmse_deg": rot_rel_rmse,
        "trans_rel_rmse_px": trans_rel_rmse,
    }


def metrics_gauge_fixed(
    params_true: np.ndarray, params_est: np.ndarray, du: float, dv: float
) -> Dict[str, float]:
    """Align the estimate with one global gauge transform before scoring."""
    Tt = _params_to_T(params_true)
    Te = _params_to_T(params_est)
    Rsum = np.zeros((3, 3), dtype=np.float64)
    for i in range(Tt.shape[0]):
        Rsum += Tt[i, :3, :3] @ Te[i, :3, :3].T
    U, _, Vt = np.linalg.svd(Rsum)
    Rg = U @ Vt
    if np.linalg.det(Rg) < 0:
        U[:, -1] *= -1
        Rg = U @ Vt

    t_res = []
    for i in range(Tt.shape[0]):
        t_res.append(Tt[i, :3, 3] - Rg @ Te[i, :3, 3])
    tg = np.mean(np.asarray(t_res), axis=0)

    G = np.eye(4, dtype=np.float32)
    G[:3, :3] = Rg.astype(np.float32)
    G[:3, 3] = tg.astype(np.float32)

    rot_errs: List[float] = []
    trans_errs_sq: List[float] = []
    for i in range(Tt.shape[0]):
        E = _inv_T(Tt[i]) @ (G @ Te[i])
        rot_errs.append(_so3_geodesic_deg(E))
        ex = float(E[0, 3]) / max(1e-12, float(du))
        ez = float(E[2, 3]) / max(1e-12, float(dv))
        trans_errs_sq.append(ex * ex + ez * ez)

    rot_rmse = float(np.sqrt(np.mean(np.square(rot_errs)))) if rot_errs else float("nan")
    trans_rmse = float(np.sqrt(np.mean(trans_errs_sq))) if trans_errs_sq else float("nan")
    return {
        "rot_gf_rmse_deg": rot_rmse,
        "trans_gf_rmse_px": trans_rmse,
    }


def project_gt_with_estimated_poses(
    x_gt: jnp.ndarray,
    grid: Grid,
    det: Detector,
    geom: ParallelGeometry | LaminographyGeometry,
    params_est: np.ndarray,
) -> jnp.ndarray:
    """Forward-project a GT volume after composing estimated per-view poses."""
    T_nom = stack_view_poses(geom, len(np.asarray(geom.thetas_deg, dtype=np.float32)))
    S_est = jax.vmap(se3_from_5d)(jnp.asarray(params_est, jnp.float32))
    T_est_full = T_nom @ S_est
    det_grid = get_detector_grid_device(det)
    vm_project = jax.vmap(
        lambda T: forward_project_view_T(
            T,
            grid,
            det,
            x_gt,
            use_checkpoint=True,
            det_grid=det_grid,
        ),
        in_axes=0,
    )
    return vm_project(T_est_full).astype(jnp.float32)


__all__ = [
    "make_gt_dataset",
    "make_misaligned_dataset",
    "metrics_abs",
    "metrics_relative",
    "metrics_gauge_fixed",
    "project_gt_with_estimated_poses",
]
