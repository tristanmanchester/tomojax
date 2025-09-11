from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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

    # Precompute nominal poses once
    n_views = int(projections.shape[0])
    T_nom_all = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )

    # Vmapped projector across views (pose-aware)
    vm_project = jax.vmap(
        lambda T, vol: forward_project_view_T(T, grid, detector, vol, use_checkpoint=True),
        in_axes=(0, None),
    )

    def align_loss(params5, vol):
        # Compose augmented poses and project all views at once
        T_aug = T_nom_all @ jax.vmap(se3_from_5d)(params5)  # (n_views, 4, 4)
        pred = vm_project(T_aug, vol)  # (n_views, nv, nu)
        resid = (pred - projections).astype(jnp.float32)
        return 0.5 * jnp.vdot(resid, resid).real

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

        x, info_rec = fista_tv(_GAll(), grid, detector, projections, iters=cfg.recon_iters, lambda_tv=cfg.lambda_tv)

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
