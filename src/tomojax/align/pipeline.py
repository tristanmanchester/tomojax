from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Tuple, Iterable, List

import jax
import jax.numpy as jnp

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view_T
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
    projector_unroll: int = 4
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    # Solver & regularization
    opt_method: str = "gd"  # default to GD for test compatibility
    gn_damping: float = 1e-6
    w_rot: float = 0.0
    w_trans: float = 0.0
    seed_translations: bool = False
    # Logging
    log_summary: bool = False
    # Reconstruction Lipschitz (optional override to skip power-method)
    recon_L: float | None = None


 # (Removed: AugmentedGeometry legacy wrapper; new alignment path uses pose-aware projector directly)


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

    # Vmapped projector across views (pose-aware). Closure captures unroll as a static constant.
    def _project_batch(T_batch, vol):
        f = lambda T: forward_project_view_T(
            T,
            grid,
            detector,
            vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
        )
        return jax.vmap(f, in_axes=0)(T_batch)

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
            pred = _project_batch(T_chunk, vol)
            resid = (pred - y_chunk).astype(jnp.float32)
            loss = loss + 0.5 * jnp.vdot(resid, resid).real
        # Smoothness prior across views (2nd difference)
        if n >= 3:
            d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
            W = jnp.array([cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans], dtype=jnp.float32)
            loss = loss + jnp.sum((d2 * W) ** 2)
        return loss

    grad_all = jax.jit(jax.grad(align_loss, argnums=0))
    align_loss_jit = jax.jit(align_loss)

    # Gauss–Newton (Levenberg–Marquardt) single-view update
    def _pred_flat(T_i, vol):
        return forward_project_view_T(
            T_i,
            grid,
            detector,
            vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
        ).ravel()

    def _gn_update_one(p5_i, T_nom_i, y_i, vol):
        def f(p5):
            T_i = T_nom_i @ se3_from_5d(p5)
            return _pred_flat(T_i, vol) - y_i.ravel()
        # J^T r
        r = f(p5_i)
        _, vjp = jax.vjp(f, p5_i)
        g = vjp(r)[0]
        # J^T J via 5 JVPs
        eye5 = jnp.eye(5, dtype=jnp.float32)
        def jvp_col(v):
            return jax.jvp(f, (p5_i,), (v,))[1]
        cols = jax.vmap(jvp_col)(eye5)
        H = cols @ cols.T
        lam = jnp.float32(cfg.gn_damping)
        dp = jnp.linalg.solve(H + lam * jnp.eye(5, dtype=H.dtype), -g)
        return dp

    _gn_update_batch = jax.jit(jax.vmap(_gn_update_one, in_axes=(0, 0, 0, None)))

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
            L=cfg.recon_L,
            init_x=x,
            views_per_batch=(cfg.views_per_batch if cfg.views_per_batch > 0 else None),
            projector_unroll=int(cfg.projector_unroll),
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
        )
        if cfg.log_summary:
            if info_rec and "loss" in info_rec and info_rec["loss"]:
                lhist = info_rec["loss"]
                logging.info(
                    "FISTA summary (outer %d/%d): first=%.4e last=%.4e min=%.4e",
                    it + 1,
                    cfg.outer_iters,
                    float(lhist[0]),
                    float(lhist[-1]),
                    float(min(lhist)),
                )

        # Alignment step: Gauss–Newton or gradient descent
        loss_before = None
        if cfg.log_summary:
            try:
                loss_before = float(align_loss_jit(params5, x))
            except Exception:
                loss_before = None
        if cfg.opt_method.lower() == "gn":
            n = params5.shape[0]
            b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n
            dp_all = []
            for s in range(0, n, b):
                dp_chunk = _gn_update_batch(
                    params5[s : s + b], T_nom_all[s : s + b], projections[s : s + b], x
                )
                params5 = params5.at[s : s + b].add(dp_chunk)
                dp_all.append(dp_chunk)
            if cfg.log_summary and dp_all:
                try:
                    dp_cat = jnp.concatenate(dp_all, axis=0)
                    rot_mean = float(jnp.mean(jnp.abs(dp_cat[:, :3])))
                    trans_mean = float(jnp.mean(jnp.abs(dp_cat[:, 3:])))
                    logging.info(
                        "GN step stats (outer %d/%d): |drot|_mean=%.3e rad, |dtrans|_mean=%.3e",
                        it + 1,
                        cfg.outer_iters,
                        rot_mean,
                        trans_mean,
                    )
                except Exception:
                    pass
        else:
            scales = jnp.array(
                [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans], dtype=jnp.float32
            )
            g_params = grad_all(params5, x)
            rms = jnp.sqrt(jnp.mean(jnp.square(g_params), axis=0)) + 1e-6
            eff_scales = scales / rms
            # Simple 2-point line search on step factor to improve single-iter progress
            best_params = params5 - g_params * eff_scales
            best_loss = align_loss_jit(best_params, x)
            cand_params = params5 - 2.0 * g_params * eff_scales
            cand_loss = align_loss_jit(cand_params, x)
            params5 = jax.lax.cond(cand_loss < best_loss, lambda _: cand_params, lambda _: best_params, operand=None)
            if cfg.log_summary:
                try:
                    rot_rms = float(jnp.mean(rms[:3]))
                    trans_rms = float(jnp.mean(rms[3:]))
                    logging.info(
                        "GD grad RMS (outer %d/%d): rot=%.3e, trans=%.3e",
                        it + 1,
                        cfg.outer_iters,
                        rot_rms,
                        trans_rms,
                    )
                except Exception:
                    pass

        # Track overall data loss
        total_loss = float(align_loss_jit(params5, x))
        loss_hist.append(total_loss)
        if cfg.log_summary:
            if loss_before is not None:
                logging.info(
                    "Align loss (outer %d/%d): before=%.4e after=%.4e delta=%.4e",
                    it + 1,
                    cfg.outer_iters,
                    float(loss_before),
                    float(total_loss),
                    float(total_loss - loss_before),
                )

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

    for li, lvl in enumerate(levels):
        g = lvl["grid"]; d = lvl["detector"]; y = lvl["projections"]
        if x_init is not None:
            # Upsample previous x to current level as init
            f_up = prev_factor // lvl["factor"]
            x0 = upsample_volume(x_init, f_up, (g.nx, g.ny, g.nz))
        else:
            x0 = None

        # Optional translation seeding at the coarsest level via phase correlation
        params0 = params5
        if li == 0 and cfg.seed_translations:
            # quick seed recon to project nominal poses
            x_seed, _ = fista_tv(
                geometry,
                g,
                d,
                y,
                iters=max(3, cfg.recon_iters // 2),
                lambda_tv=cfg.lambda_tv,
                init_x=x0,
                projector_unroll=int(cfg.projector_unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=cfg.gather_dtype,
            )
            T_nom = jnp.stack(
                [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(y.shape[0])],
                axis=0,
            )
            from ..utils.phasecorr import phase_corr_shift
            vm_pred = jax.vmap(
                lambda T: forward_project_view_T(
                    T,
                    g,
                    d,
                    x_seed,
                    use_checkpoint=cfg.checkpoint_projector,
                    gather_dtype=cfg.gather_dtype,
                ),
                in_axes=0,
            )
            preds = vm_pred(T_nom)
            shift_uv = jax.vmap(phase_corr_shift)(preds, y)  # returns (du, dv)
            shifts = jnp.stack(shift_uv, axis=1).astype(jnp.float32)  # (n, 2)
            # Convert pixel shifts to world units using detector spacing
            dx = shifts[:, 0] * jnp.float32(d.du)
            dz = shifts[:, 1] * jnp.float32(d.dv)
            params0 = jnp.zeros((y.shape[0], 5), dtype=jnp.float32)
            params0 = params0.at[:, 3].set(dx)
            params0 = params0.at[:, 4].set(dz)

        # Run alignment at this level
        x_lvl, params5, info = align(
            geometry, g, d, y, cfg=cfg, init_x=x0, init_params5=params0
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
