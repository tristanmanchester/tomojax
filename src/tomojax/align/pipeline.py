from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
import time
from typing import Any, Dict, Tuple, Iterable, List, Optional

import jax
import jax.numpy as jnp
import optax

from ..core.geometry import Geometry, Grid, Detector
from ..core.projector import forward_project_view_T, get_detector_grid_device
from ..recon.fista_tv import fista_tv
from ..utils.logging import progress_iter, format_duration
from .parametrizations import se3_from_5d
from .losses import build_loss


@dataclass
class AlignConfig:
    outer_iters: int = 5
    recon_iters: int = 10
    lambda_tv: float = 0.005
    tv_prox_iters: int = 10
    # Reconstruction stopping criteria
    recon_rel_tol: float | None = None
    recon_patience: int = 2
    # Alignment step sizes
    lr_rot: float = 1e-3  # radians
    lr_trans: float = 1e-1  # world units
    # Memory/throughput knobs (hidden defaults)
    views_per_batch: int = 1  # stream one view at a time
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    # Solver & regularization (GN and GD only; LBFGS removed)
    opt_method: str = "gn"
    gn_damping: float = 1e-6
    w_rot: float = 0.0
    w_trans: float = 0.0
    seed_translations: bool = False
    # Logging
    log_summary: bool = False
    log_compact: bool = True  # print one compact line per outer when log_summary is enabled
    # Reconstruction Lipschitz (optional override to skip power-method)
    recon_L: float | None = None
    # Early stopping across outers (alignment phase)
    early_stop: bool = True
    early_stop_rel_impr: float = 1e-3  # stop if (before-after)/before < this
    early_stop_patience: int = 2
    # GN acceptance: only apply step if it reduces the alignment loss
    gn_accept_only_improving: bool = True
    gn_accept_tol: float = 0.0  # allow tiny increases if >0 (as fraction of before)
    # Data term / similarity
    loss_kind: str = "l2_otsu"
    loss_params: Optional[Dict[str, float]] = None
    # (LBFGS settings removed)


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

    # Precompute detector grid once (device arrays) to avoid repeated transfers/logging
    det_grid = get_detector_grid_device(detector)

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
            det_grid=det_grid,
        )
        return jax.vmap(f, in_axes=0)(T_batch)

    # Static smoothness weights to avoid rebuilding inside jitted loss
    W_weights = jnp.array(
        [cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans], dtype=jnp.float32
    )

    # Build per-view loss once (may precompute masks on targets)
    per_view_loss_fn, loss_state = build_loss(cfg.loss_kind, cfg.loss_params, projections)

    def align_loss(params5, vol):
        # Compose augmented poses
        # Current convention: per-view misalignment parameters act in the object
        # frame and are post-multiplied: T_world_from_obj_aug = T_nom @ T_delta.
        # This is consistent across parallel CT and laminography sample-frame.
        T_aug = T_nom_all @ jax.vmap(se3_from_5d)(params5)  # (n_views, 4, 4)
        # Use Python ints for sizes to keep them static
        n = int(params5.shape[0])
        nv = int(projections.shape[1])
        nu = int(projections.shape[2])
        b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n
        b = min(b, n)
        m = (n + b - 1) // b

        def body(loss_acc, i):
            i = jnp.int32(i)
            start = i * jnp.int32(b)
            remaining = jnp.maximum(0, jnp.int32(n) - start)
            valid = jnp.minimum(jnp.int32(b), remaining)
            shift = jnp.int32(b) - valid
            start_shifted = jnp.maximum(0, start - shift)
            T_chunk = jax.lax.dynamic_slice(T_aug, (start_shifted, 0, 0), (b, 4, 4))
            y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
            pred = _project_batch(T_chunk, vol)
            # Optional per-view mask slice (for Otsu-masked L2)
            mask_chunk = None
            if getattr(loss_state, "mask", None) is not None:
                mask_chunk = jax.lax.dynamic_slice(loss_state.mask, (start_shifted, 0, 0), (b, nv, nu))
            # Compute per-view losses, then zero-out invalid padded items
            lvec = per_view_loss_fn(pred, y_chunk, mask_chunk)  # (b,)
            idx = jnp.arange(b)
            vmask = (idx >= (jnp.int32(b) - valid)).astype(jnp.float32)
            loss_batch = jnp.sum(lvec * vmask)
            return (loss_acc + loss_batch, None)

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(body, loss0, jnp.arange(m))

        # Smoothness prior across views (2nd difference)
        loss = loss_tot
        if int(params5.shape[0]) >= 3:
            d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
            loss = loss + jnp.sum((d2 * W_weights) ** 2)
        return loss

    # Value function for whole batch (forward only) kept for logging and line search
    align_loss_jit = jax.jit(align_loss)

    # Memory-safe gradient: compute per-view grads in a Python loop, add smoothness analytic grad.
    # This avoids reverse-mode backprop across the full scan of all views at once.
    def _one_view_loss(p5_i, T_nom_i, y_i, vol, mask_i):
        T_i = T_nom_i @ se3_from_5d(p5_i)
        pred_i = forward_project_view_T(
            T_i,
            grid,
            detector,
            vol,
            use_checkpoint=cfg.checkpoint_projector,
            unroll=int(cfg.projector_unroll),
            gather_dtype=cfg.gather_dtype,
            det_grid=det_grid,
        )
        # Reuse the built per-view loss function; feed as a single-item batch
        lvec = per_view_loss_fn(pred_i[None, ...], y_i[None, ...], mask_i[None, ...])
        return lvec[0]

    one_view_val_and_grad = jax.jit(jax.value_and_grad(_one_view_loss))

    def loss_and_grad_manual(params5, vol):
        # Data term: sum over views
        n = int(params5.shape[0])
        total = jnp.float32(0.0)
        g = jnp.zeros_like(params5)
        for i in range(n):
            y_i = projections[i]
            T_nom_i = T_nom_all[i]
            # Provide a concrete mask slice even if unused by the loss
            if getattr(loss_state, "mask", None) is not None:
                mask_i = loss_state.mask[i]
            else:
                mask_i = jnp.zeros_like(y_i)
            li, gi = one_view_val_and_grad(params5[i], T_nom_i, y_i, vol, mask_i)
            total = total + li
            g = g.at[i].set(gi)
        # Smoothness prior gradient (second difference): tridiagonal conv with [-1, 2, -1]
        if int(params5.shape[0]) >= 3:
            d2 = params5[:-2] - 2.0 * params5[1:-1] + params5[2:]
            w = jnp.array([cfg.w_rot, cfg.w_rot, cfg.w_rot, cfg.w_trans, cfg.w_trans], jnp.float32)
            total = total + jnp.sum((d2 * w) ** 2)
            # Gradient contribution
            # For middle points i in [1..n-2]: grad += 2*w^2 * ( -1*(p[i-1]-2p[i]+p[i+1]) *(-2) + ... )
            # Easier via explicit accumulation
            ww = (w ** 2) * 2.0
            n = params5.shape[0]
            # i term contributes: +(-2)*d2[i] to grad[i+1], +(1)*d2[i] to grad[i], +(1)*d2[i] to grad[i+2]
            g = g.at[1:-1].add(-2.0 * d2 * ww)
            g = g.at[0:-2].add(1.0 * d2 * ww)
            g = g.at[2:].add(1.0 * d2 * ww)
        return total, g

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
            det_grid=det_grid,
        ).ravel()

    def _gn_update_one(p5_i, T_nom_i, y_i, vol, w_i):
        def f(p5):
            T_i = T_nom_i @ se3_from_5d(p5)
            r = _pred_flat(T_i, vol) - y_i.ravel()
            return w_i.ravel() * r
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

    _gn_update_batch = jax.jit(jax.vmap(_gn_update_one, in_axes=(0, 0, 0, None, 0)))

    # Reuse measured Lipschitz across outer iterations to avoid repeated power-method
    L_prev = cfg.recon_L
    small_impr_streak = 0
    opt_mode = str(cfg.opt_method).lower()
    outer_stats: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()

    def _log_outer_summary(stat: Dict[str, Any]) -> None:
        outer_idx = int(stat.get("outer_idx", 0))
        total_iters = int(cfg.outer_iters)
        total_time = format_duration(stat.get("outer_time"))
        elapsed = format_duration(stat.get("cumulative_time"))
        if cfg.log_compact:
            # Build compact one-liner with key fields
            parts: List[str] = [f"Outer {outer_idx}/{total_iters}"]
            # Recon summary
            rbits: List[str] = []
            rt = stat.get("recon_time")
            if rt is not None:
                rbits.append(f"{format_duration(rt)}")
            if stat.get("recon_retry"):
                rbits.append("retry")
            lm = stat.get("L_meas"); ln = stat.get("L_next")
            if (lm is not None) and (ln is not None):
                rbits.append(f"L {lm:.2e}->{ln:.2e}")
            ff = stat.get("fista_first"); fl = stat.get("fista_last"); fm = stat.get("fista_min")
            if (ff is not None) and (fl is not None):
                if fm is not None:
                    rbits.append(f"loss {ff:.2e}->{fl:.2e} (min {fm:.2e})")
                else:
                    rbits.append(f"loss {ff:.2e}->{fl:.2e}")
            if rbits:
                parts.append("recon " + " ".join(rbits))
            # Align summary
            abits: List[str] = []
            at = stat.get("align_time")
            if at is not None:
                abits.append(f"{format_duration(at)}")
            sk = stat.get("step_kind")
            if sk == "gn":
                rm = stat.get("rot_mean"); tm = stat.get("trans_mean")
                if rm is not None: abits.append(f"|drot| {rm:.2e}")
                if tm is not None: abits.append(f"|dtrans| {tm:.2e}")
            elif sk == "gd":
                rr = stat.get("rot_rms"); tr = stat.get("trans_rms")
                if rr is not None: abits.append(f"rotRMS {rr:.2e}")
                if tr is not None: abits.append(f"transRMS {tr:.2e}")
            lb = stat.get("loss_before"); la = stat.get("loss_after"); ld = stat.get("loss_delta"); rp = stat.get("loss_rel_pct")
            if (lb is not None) and (la is not None):
                rel = f" {rp:+.2f}%" if rp is not None else ""
                abits.append(f"loss {lb:.2e}->{la:.2e} (Δ {ld:+.2e}{rel})")
            if abits:
                parts.append("align " + " ".join(abits))
            parts.append(f"elapsed {elapsed}")
            logging.info(" | ".join(parts))
            return
        logging.info(
            "Outer %d/%d | total %s | elapsed %s",
            outer_idx,
            total_iters,
            total_time,
            elapsed,
        )

        recon_parts: List[str] = []
        recon_time = stat.get("recon_time")
        if recon_time is not None:
            recon_parts.append(f"time {format_duration(recon_time)}")
        if stat.get("recon_retry"):
            recon_parts.append("fallback retry")
        l_meas = stat.get("L_meas")
        l_next = stat.get("L_next")
        if (l_meas is not None) and (l_next is not None):
            recon_parts.append(f"L {l_meas:.3e}->{l_next:.3e}")
        f_first = stat.get("fista_first")
        f_last = stat.get("fista_last")
        f_min = stat.get("fista_min")
        if (f_first is not None) and (f_last is not None):
            if f_min is not None:
                recon_parts.append(f"loss {f_first:.3e}->{f_last:.3e} (min {f_min:.3e})")
            else:
                recon_parts.append(f"loss {f_first:.3e}->{f_last:.3e}")
        logging.info("  Recon | %s", " | ".join(recon_parts) if recon_parts else "-")

        align_parts: List[str] = []
        align_time = stat.get("align_time")
        if align_time is not None:
            align_parts.append(f"time {format_duration(align_time)}")
        step_kind = stat.get("step_kind")
        if step_kind == "gn":
            rot_mean = stat.get("rot_mean")
            trans_mean = stat.get("trans_mean")
            if rot_mean is not None:
                align_parts.append(f"|drot|_mean {rot_mean:.3e} rad")
            if trans_mean is not None:
                align_parts.append(f"|dtrans|_mean {trans_mean:.3e}")
        elif step_kind == "gd":
            rot_rms = stat.get("rot_rms")
            trans_rms = stat.get("trans_rms")
            if rot_rms is not None:
                align_parts.append(f"rot RMS {rot_rms:.3e}")
            if trans_rms is not None:
                align_parts.append(f"trans RMS {trans_rms:.3e}")
        loss_before = stat.get("loss_before")
        loss_after = stat.get("loss_after")
        loss_delta = stat.get("loss_delta")
        rel_pct = stat.get("loss_rel_pct")
        if (loss_before is not None) and (loss_after is not None):
            rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
            align_parts.append(
                f"loss {loss_before:.3e}->{loss_after:.3e} (Δ {loss_delta:+.3e}{rel_str})"
            )
        logging.info("  Align | %s", " | ".join(align_parts) if align_parts else "-")

    for it in progress_iter(range(cfg.outer_iters), total=cfg.outer_iters, desc="Align: outer iters"):
        outer_idx = it + 1
        stat: Dict[str, Any] = {"outer_idx": outer_idx}
        outer_start = time.perf_counter()

        # Reconstruction step
        class _GAll:
            def pose_for_view(self, i):
                T_nom = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
                T_al = se3_from_5d(params5[i])
                return tuple(map(tuple, T_nom @ T_al))

            def rays_for_view(self, i):
                return geometry.rays_for_view(i)

        def _run_fista_safe(vpb: int | None, unroll: int, gather: str, gm: str):
            return fista_tv(
                _GAll(),
                grid,
                detector,
                projections,
                iters=cfg.recon_iters,
                lambda_tv=cfg.lambda_tv,
                L=L_prev,
                init_x=x,
                views_per_batch=vpb,
                projector_unroll=int(unroll),
                checkpoint_projector=cfg.checkpoint_projector,
                gather_dtype=gather,
                grad_mode=gm,
                tv_prox_iters=int(cfg.tv_prox_iters),
                recon_rel_tol=cfg.recon_rel_tol,
                recon_patience=(
                    int(cfg.recon_patience) if cfg.recon_patience is not None else 0
                ),
            )

        vpb0 = (cfg.views_per_batch if cfg.views_per_batch > 0 else None)
        recon_retry = False
        recon_start = time.perf_counter()
        try:
            x, info_rec = _run_fista_safe(vpb0, int(cfg.projector_unroll), cfg.gather_dtype, "auto")
        except Exception as e:
            msg = str(e)
            if ("RESOURCE_EXHAUSTED" in msg) or ("Out of memory" in msg) or ("Allocator" in msg):
                logging.warning("FISTA OOM detected; retrying with safer settings (vpb=1, unroll=1, stream)")
                try:
                    recon_retry = True
                    x, info_rec = _run_fista_safe(1, 1, cfg.gather_dtype, "stream")
                except Exception as e2:
                    msg2 = str(e2)
                    if ("RESOURCE_EXHAUSTED" in msg2) or ("Out of memory" in msg2) or ("Allocator" in msg2):
                        logging.error(
                            "FISTA still OOM at finest level. Reduce memory pressure (smaller problem size or lower internal batching), or provide --recon-L to skip power-method."
                        )
                    raise
            else:
                raise
        # Ensure device work is finished before timing recon
        try:
            x.block_until_ready()
        except Exception:
            # Fallback if x is not a DeviceArray-like object
            jax.block_until_ready(x)
        recon_time = time.perf_counter() - recon_start
        stat["recon_time"] = recon_time
        stat["recon_retry"] = recon_retry
        # Capture and reuse measured L next iteration (with small safety margin)
        try:
            L_meas = float(info_rec.get("L", 0.0))
            if L_meas > 0.0:
                L_prev = 1.2 * L_meas
                stat["L_meas"] = L_meas
                stat["L_next"] = L_prev
        except Exception:
            pass
        if info_rec and "loss" in info_rec and info_rec["loss"]:
            try:
                lhist = info_rec["loss"]
                stat["fista_first"] = float(lhist[0])
                stat["fista_last"] = float(lhist[-1])
                stat["fista_min"] = float(min(lhist))
            except Exception:
                pass

        # Alignment step: Gauss–Newton, LBFGS, or gradient descent
        # Evaluate alignment loss before update (needed for GN acceptance / early stop)
        align_start = time.perf_counter()
        try:
            loss_before = float(align_loss_jit(params5, x))
        except Exception:
            loss_before = None
        stat["loss_before"] = loss_before
        loss_kind_norm = str(cfg.loss_kind).lower()
        # GN supported for LS-like losses: l2, l2_otsu (masked L2), pwls
        ls_like = loss_kind_norm in ("l2", "l2_otsu", "l2-otsu", "otsu-l2", "pwls", "edge_l2", "edge_aware_l2")
        if opt_mode == "gn" and ls_like:
            step_kind = "gn"
        else:
            step_kind = "gd"
        loss_after = None
        if step_kind == "gn":
            n = params5.shape[0]
            b = int(cfg.views_per_batch) if int(cfg.views_per_batch) > 0 else n
            dp_all = []
            params5_prev = params5
            for s in range(0, n, b):
                chunk_len = min(b, n - s)
                idx = slice(s, s + chunk_len)
                # Build per-pixel weights (sqrt of LS weight) for LS-like losses
                y_chunk = projections[idx]
                if loss_kind_norm in ("l2",):
                    w_chunk = jnp.ones_like(y_chunk)
                elif loss_kind_norm in ("l2_otsu", "l2-otsu", "otsu-l2") and getattr(loss_state, "mask", None) is not None:
                    # Loss implemented as 0.5 * sum (w * r)^2; here w = mask in [0,1]
                    w_chunk = loss_state.mask[idx]
                elif loss_kind_norm == "pwls":
                    a = jnp.float32((cfg.loss_params or {}).get("a", 1.0))
                    bpar = jnp.float32((cfg.loss_params or {}).get("b", 0.0))
                    w = 1.0 / (a * jnp.clip(y_chunk, 0.0) + bpar + 1e-6)
                    w_chunk = jnp.sqrt(w)
                elif loss_kind_norm in ("edge_l2", "edge_aware_l2"):
                    # Sobel gradients of target to form weights: r_w = sqrt(1 + ||∇y||) * r
                    kx = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], jnp.float32) / 8.0
                    ky = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], jnp.float32) / 8.0
                    y4 = y_chunk[..., None]  # (b, nv, nu, 1)
                    gx = jax.lax.conv_general_dilated(y4, kx[:, :, None, None], (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
                    gy = jax.lax.conv_general_dilated(y4, ky[:, :, None, None], (1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
                    mag = jnp.sqrt(gx[..., 0] ** 2 + gy[..., 0] ** 2)
                    w_chunk = jnp.sqrt(1.0 + mag)
                else:
                    w_chunk = jnp.ones_like(y_chunk)
                params_chunk = params5[idx]
                T_chunk = T_nom_all[idx]

                if chunk_len != b:
                    pad_count = b - chunk_len

                    def _pad_repeat_last(arr: jnp.ndarray) -> jnp.ndarray:
                        pad_vals = jnp.repeat(arr[-1:], pad_count, axis=0)
                        return jnp.concatenate([arr, pad_vals], axis=0)

                    params_chunk_in = _pad_repeat_last(params_chunk)
                    T_chunk_in = _pad_repeat_last(T_chunk)
                    y_chunk_in = _pad_repeat_last(y_chunk)
                    w_chunk_in = _pad_repeat_last(w_chunk)
                else:
                    params_chunk_in = params_chunk
                    T_chunk_in = T_chunk
                    y_chunk_in = y_chunk
                    w_chunk_in = w_chunk

                dp_chunk_full = _gn_update_batch(
                    params_chunk_in, T_chunk_in, y_chunk_in, x, w_chunk_in
                )
                dp_chunk = dp_chunk_full[:chunk_len]
                params5 = params5.at[idx].add(dp_chunk)
                dp_all.append(dp_chunk)
        # Compute post-update loss and accept/reject
            loss_after = float(align_loss_jit(params5, x)) if loss_before is not None else None
            if cfg.gn_accept_only_improving and (loss_before is not None) and (loss_after is not None):
                tol = float(cfg.gn_accept_tol) * abs(loss_before)
                if not (loss_after < loss_before - tol):
                    # Reject step
                    params5 = params5_prev
                    loss_after = loss_before
            # Log step stats
            if dp_all:
                try:
                    dp_cat = jnp.concatenate(dp_all, axis=0)
                    stat["rot_mean"] = float(jnp.mean(jnp.abs(dp_cat[:, :3])))
                    stat["trans_mean"] = float(jnp.mean(jnp.abs(dp_cat[:, 3:])))
                except Exception:
                    pass
        
        else:
            scales = jnp.array(
                [cfg.lr_rot, cfg.lr_rot, cfg.lr_rot, cfg.lr_trans, cfg.lr_trans], dtype=jnp.float32
            )
            # Keep a copy for line search; donated arg may be reused internally
            p5_in = params5
            _, g_params = loss_and_grad_manual(params5, x)
            rms = jnp.sqrt(jnp.mean(jnp.square(g_params), axis=0)) + 1e-6
            eff_scales = scales / rms
            # Simple 2-point line search on step factor to improve single-iter progress
            best_params = p5_in - g_params * eff_scales
            best_loss = align_loss_jit(best_params, x)
            cand_params = p5_in - 2.0 * g_params * eff_scales
            cand_loss = align_loss_jit(cand_params, x)
            params5 = jax.lax.cond(cand_loss < best_loss, lambda _: cand_params, lambda _: best_params, operand=None)
            loss_after = float(jnp.minimum(best_loss, cand_loss)) if loss_before is not None else None
            try:
                stat["rot_rms"] = float(jnp.mean(rms[:3]))
                stat["trans_rms"] = float(jnp.mean(rms[3:]))
            except Exception:
                pass
        stat["step_kind"] = step_kind
        stat["loss_after_step"] = loss_after
        # Ensure device work from alignment step is finished before timing
        try:
            # Prefer object method if available (propagates device errors correctly)
            params5.block_until_ready()  # type: ignore[attr-defined]
        except Exception:
            try:
                jax.block_until_ready(params5)
            except Exception:
                pass
        stat["align_time"] = time.perf_counter() - align_start

        # Track overall data loss
        total_loss = float(align_loss_jit(params5, x))
        loss_hist.append(total_loss)
        stat["loss_after"] = total_loss
        if loss_before is not None:
            delta = total_loss - loss_before
            stat["loss_delta"] = delta
            if math.isfinite(loss_before) and abs(loss_before) > 1e-12:
                stat["loss_rel_pct"] = (delta / loss_before) * 100.0
            else:
                stat["loss_rel_pct"] = None
            if math.isfinite(loss_before) and math.isfinite(total_loss):
                denom = max(abs(loss_before), 1e-12)
                rel_impr = (loss_before - total_loss) / denom
            else:
                rel_impr = None
        else:
            stat["loss_delta"] = None
            stat["loss_rel_pct"] = None
            rel_impr = None
        stat["rel_impr"] = rel_impr

        outer_time = time.perf_counter() - outer_start
        stat["outer_time"] = outer_time
        stat["cumulative_time"] = time.perf_counter() - wall_start
        outer_stats.append(stat)

        if cfg.log_summary:
            _log_outer_summary(stat)

        # Early stopping based on alignment improvement during GN/GD step
        if cfg.early_stop and (rel_impr is not None):
            rel_for_patience = rel_impr
            if (not math.isfinite(rel_for_patience)) or (rel_for_patience < 0.0):
                rel_for_patience = 0.0
            if rel_for_patience < float(cfg.early_stop_rel_impr):
                small_impr_streak += 1
            else:
                small_impr_streak = 0
            if small_impr_streak >= int(cfg.early_stop_patience):
                if cfg.log_summary:
                    logging.info(
                        "Early stop after %d outer iters (%s elapsed): rel_impr=%.3e < %.3e for %d consecutive outers",
                        outer_idx,
                        format_duration(stat.get("cumulative_time")),
                        float(rel_impr),
                        float(cfg.early_stop_rel_impr),
                        int(cfg.early_stop_patience),
                    )
                break
        elif cfg.early_stop:
            small_impr_streak = 0

    if cfg.log_summary and outer_stats:
        recon_total = sum(float(s.get("recon_time", 0.0)) for s in outer_stats if s.get("recon_time") is not None)
        align_total = sum(float(s.get("align_time", 0.0)) for s in outer_stats if s.get("align_time") is not None)
        wall_total = time.perf_counter() - wall_start
        logging.info(
            "Alignment completed in %s (recon %s, align %s over %d outer iters)",
            format_duration(wall_total),
            format_duration(recon_total),
            format_duration(align_total),
            len(outer_stats),
        )
        first_loss = outer_stats[0].get("loss_before") if outer_stats else None
        final_loss = outer_stats[-1].get("loss_after") if outer_stats else None
        if (first_loss is not None) and (final_loss is not None):
            total_delta = final_loss - first_loss
            rel_pct = (total_delta / first_loss) * 100.0 if abs(first_loss) > 1e-12 else None
            rel_str = f", {rel_pct:+.2f}%" if rel_pct is not None else ""
            logging.info(
                "  Loss %s -> %s (Δ %s%s)",
                f"{first_loss:.3e}",
                f"{final_loss:.3e}",
                f"{total_delta:+.3e}",
                rel_str,
            )
        best_loss = min(
            (s.get("loss_after") for s in outer_stats if s.get("loss_after") is not None),
            default=None,
        )
        if best_loss is not None and final_loss is not None and best_loss < final_loss:
            logging.info("  Best loss observed: %.3e", best_loss)

    # Provide last measured/reused L for potential reuse across levels
    info = {
        "loss": loss_hist,
        "L": (float(L_prev) if L_prev is not None else None),
        "outer_stats": outer_stats,
    }
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
                recon_rel_tol=cfg.recon_rel_tol,
                recon_patience=(
                    int(cfg.recon_patience) if cfg.recon_patience is not None else 0
                ),
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
        # Re-estimate L at each level using a fresh (streamed) power-method for stability
        cfg_level = replace(cfg, recon_L=None)
        x_lvl, params5, info = align(
            geometry, g, d, y, cfg=cfg_level, init_x=x0, init_params5=params0
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
