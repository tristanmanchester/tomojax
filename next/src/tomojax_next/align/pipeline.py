from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List

import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view, forward_project_view_T
from ..recon.fista_tv import fista_tv
from ..utils.logging import progress_iter
from .parametrizations import se3_from_5d


@dataclass
class AlignConfig:
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units
    # Memory/throughput knobs
    views_per_batch: int = 0  # 0 -> all views at once
    projector_unroll: int = 1


class AugmentedGeometry:
    """Wraps an existing geometry to apply per-view alignment transforms.

    pose_aug(i) = pose_nominal(i) @ T_align(params[i])
    """

    def __init__(self, base: Geometry, params5: jnp.ndarray):
        self.base = base
        self.params5 = params5  # shape (n_views, 5)

    def pose_for_view(self, i: int):
        T_nom = jnp.asarray(self.base.pose_for_view(i), dtype=jnp.float32)
        T_al = se3_from_5d(self.params5[i])
        T = T_nom @ T_al
        # Return as nested tuples to keep compatibility with downstream code
        return tuple(map(tuple, T))

    def rays_for_view(self, i: int):
        return self.base.rays_for_view(i)


def align(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,  # (n_views, nv, nu)
    *,
    cfg: AlignConfig | None = None,
    init_x: jnp.ndarray | None = None,
    init_params5: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Alternating reconstruction + per-view alignment (5-DOF) on small cases.

    Returns (x, params5, info) with loss history and optional metrics.
    """
    if cfg is None:
        cfg = AlignConfig()
    n_views = int(projections.shape[0])
    # Initialize volume and params
    x = (
        jnp.asarray(init_x, dtype=jnp.float32)
        if init_x is not None
        else jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    )
    params5 = (
        jnp.asarray(init_params5, dtype=jnp.float32)
        if init_params5 is not None
        else jnp.zeros((n_views, 5), dtype=jnp.float32)
    )

    loss_hist = []

    # Precompute nominal poses once
    n_views = int(projections.shape[0])
    T_nom_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )

    # Vmapped projector across views (pose-aware)
    vm_project = jax.vmap(
        lambda T, vol: forward_project_view_T(
            T, grid, detector, vol, use_checkpoint=True, unroll=int(cfg.projector_unroll)
        ),
        in_axes=(0, None),
    )

    def align_loss(params5, vol):
        # Compose augmented poses
        T_aug = T_nom_all @ jax.vmap(se3_from_5d)(params5)  # (n_views, 4, 4)
        n = params5.shape[0]
        b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n
        loss = jnp.float32(0.0)
        # Chunk over views to control peak memory; Python loop is unrolled at trace-time using static shape
        for s in range(0, n, b):
            T_chunk = T_aug[s : s + b]
            y_chunk = projections[s : s + b]
            pred = vm_project(T_chunk, vol)
            resid = (pred - y_chunk).astype(jnp.float32)
            loss = loss + 0.5 * jnp.vdot(resid, resid).real
        return loss

    grad_all = jax.jit(jax.grad(align_loss, argnums=0))
    align_loss_jit = jax.jit(align_loss)

    for it in progress_iter(range(cfg.outer_iters), total=cfg.outer_iters, desc="Align: outer iters"):
        # Reconstruction step
        class _GAll:
            def pose_for_view(self, i):
                T_nom = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(params5[i])
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, i):
                return geometry.rays_for_view(i)

        x, info_rec = fista_tv(
            _GAll(),
            grid,
            detector,
            projections,
            iters=cfg.recon_iters,
            lambda_tv=cfg.lambda_tv,
            init_x=x,
            views_per_batch=(cfg.views_per_batch if cfg.views_per_batch > 0 else None),
            projector_unroll=int(cfg.projector_unroll),
        )

        # Alignment step: single vmapped gradient over all views
        g_params = grad_all(params5, x)
        scales = jnp.array(
            [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans], dtype=jnp.float32
        )
        params5 = params5 - g_params * scales

        # Track overall data loss
        total_loss = float(align_loss_jit(params5, x))
        loss_hist.append(total_loss)

    info = {"loss": loss_hist}
    return x, params5, info


def align_multires(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    factors: Iterable[int] = (2, 1),
    cfg: AlignConfig | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Coarse-to-fine alignment using simple binning for speed and robustness.

    Carries alignment parameters across levels and downsamples/upsamples volume.
    """
    from ..recon.multires import scale_grid, scale_detector, bin_projections, bin_volume, upsample_volume

    if cfg is None:
        cfg = AlignConfig()

    levels: List[dict] = []
    for f in factors:
        levels.append(
            {
                "factor": int(f),
                "grid": scale_grid(grid, int(f)),
                "detector": scale_detector(detector, int(f)),
                "projections": bin_projections(projections, int(f)),
            }
        )

    x_init = None
    params5 = None
    loss_hist: List[float] = []

    for lvl in levels:
        g = lvl["grid"]; d = lvl["detector"]; y = lvl["projections"]
        if x_init is not None:
            # Bin previous x to current level as init
            f_prev = prev_factor // lvl["factor"]
            x0 = bin_volume(x_init, f_prev)
        else:
            x0 = None

        # Run a few outer iterations at this level
        # Reuse align() but allow initialization by starting params at previous values
        # (params are resolution-agnostic: rotations [rad], translations [units])
        x_lvl, params5, info = align(
            geometry, g, d, y, cfg=cfg, init_x=x0, init_params5=params5
        )
        loss_hist.extend(info.get("loss", []))
        x_init = x_lvl
        prev_factor = lvl["factor"]

    # Upsample to finest grid if last level not 1
    if levels and levels[-1]["factor"] != 1:
        x_final = upsample_volume(x_init, levels[-1]["factor"], (grid.nx, grid.ny, grid.nz))
    else:
        x_final = x_init

    return x_final, params5 if params5 is not None else jnp.zeros((projections.shape[0], 5), jnp.float32), {"loss": loss_hist, "factors": list(factors)}
