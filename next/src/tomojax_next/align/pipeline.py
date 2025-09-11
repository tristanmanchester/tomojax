from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view
from ..recon.fista_tv import fista_tv
from .parametrizations import se3_from_5d


@dataclass
class AlignConfig:
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units


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
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Alternating reconstruction + per-view alignment (5-DOF) on small cases.

    Returns (x, params5, info) with loss history and optional metrics.
    """
    if cfg is None:
        cfg = AlignConfig()
    n_views = int(projections.shape[0])
    # Initialize volume with zeros and params with zeros
    x = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    params5 = jnp.zeros((n_views, 5), dtype=jnp.float32)

    loss_hist = []

    def single_view_loss(p5, i, vol):
        class _G:
            def pose_for_view(self, _):
                T_nom = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(p5)
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, _):
                return geometry.rays_for_view(i)

        pred = forward_project_view(_G(), grid, detector, vol, view_index=0)
        resid = (pred - projections[i]).astype(jnp.float32)
        return 0.5 * jnp.vdot(resid, resid).real

    for it in range(cfg.outer_iters):
        # Reconstruction step
        class _GAll:
            def pose_for_view(self, i):
                T_nom = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(params5[i])
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, i):
                return geometry.rays_for_view(i)

        x, info_rec = fista_tv(_GAll(), grid, detector, projections, iters=cfg.recon_iters, lambda_tv=cfg.lambda_tv)

        # Alignment step: simple gradient descent per view
        for i in range(n_views):
            grad_params_i = jax.grad(lambda p: single_view_loss(p, i, x))
            g = grad_params_i(params5[i])
            # Separate scales for rot (first 3) and trans (last 2)
            upd = jnp.array([cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans], dtype=jnp.float32) * g
            params5 = params5.at[i].add(-upd)

        # Track overall data loss
        total_loss = 0.0
        for i in range(n_views):
            total_loss = total_loss + float(single_view_loss(params5[i], i, x))
        loss_hist.append(total_loss)

    info = {"loss": loss_hist}
    return x, params5, info
