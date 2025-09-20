from __future__ import annotations

import argparse
import logging
import numpy as np
import jax.numpy as jnp
import os
from contextlib import nullcontext as _nullcontext

from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..align.pipeline import align, AlignConfig
from ..utils.logging import setup_logging, log_jax_env


def _init_jax_compilation_cache() -> None:
    """Enable JAX persistent compilation cache for faster re-runs.

    Directory precedence:
    - TOMOJAX_JAX_CACHE_DIR if set
    - ${XDG_CACHE_HOME:-~/.cache}/tomojax/jax_cache
    """
    try:
        # Avoid noisy logs on environments without this feature
        from jax.experimental import compilation_cache as cc  # type: ignore
    except Exception:
        return
    try:
        cache_dir = os.environ.get("TOMOJAX_JAX_CACHE_DIR")
        if not cache_dir:
            base = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
            cache_dir = os.path.join(base, "tomojax", "jax_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cc.initialize_cache(cache_dir)
        logging.info("JAX compilation cache: %s", cache_dir)
    except Exception:
        # Best-effort; skip on any failure silently
        pass


def build_geometry(meta: dict):
    grid_d = meta["grid"]; det_d = meta["detector"]; thetas = meta["thetas_deg"]; gtype = meta.get("geometry_type", "parallel")
    grid = Grid(**{k: grid_d[k] for k in ("nx","ny","nz","vx","vy","vz")})
    detector = Detector(**{k: det_d[k] for k in ("nu","nv","du","dv")}, det_center=tuple(det_d.get("det_center", (0.0,0.0))))
    if gtype == "parallel":
        geom = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0))
        tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(grid=grid, detector=detector, thetas_deg=thetas, tilt_deg=tilt_deg, tilt_about=tilt_about)
    return grid, detector, geom

def _transfer_guard_ctx(mode: str | None = None):
    # Allow overriding via env var to control verbosity: off|log|disallow
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log").lower()
    if mode in ("off", "none", "disable", "disabled"):
        return _nullcontext()
    try:
        import jax as _jax  # local import to avoid hard dep at import time
        tg = getattr(_jax, "transfer_guard", None)
        if tg is not None:
            return tg(mode)
        try:
            from jax.experimental import transfer_guard as _tg  # type: ignore
            return _tg(mode)
        except Exception:
            return _nullcontext()
    except Exception:
        return _nullcontext()


def main() -> None:
    p = argparse.ArgumentParser(description="Joint reconstruction + alignment on dataset (.nxs)")
    p.add_argument("--data", required=True, help="Input .nxs")
    p.add_argument("--outer-iters", type=int, default=5)
    p.add_argument("--recon-iters", type=int, default=10)
    p.add_argument("--lambda-tv", type=float, default=0.005)
    p.add_argument("--tv-prox-iters", type=int, default=10, help="Inner iterations for TV proximal operator")
    p.add_argument("--lr-rot", type=float, default=1e-3)
    p.add_argument("--lr-trans", type=float, default=1e-1)
    p.add_argument("--levels", type=int, nargs="+", default=None, help="Optional multires factors, e.g., 4 2 1")
    p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype (auto: bf16 on GPU/TPU, else fp32)",
    )
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true")
    ck.add_argument("--no-checkpoint-projector", dest="checkpoint_projector", action="store_false")
    p.set_defaults(checkpoint_projector=True)
    p.add_argument("--opt-method", choices=["gd", "gn"], default="gd", help="Alignment optimizer: gd or gn (gn only valid with --loss l2)")
    p.add_argument("--gn-damping", type=float, default=1e-3, help="Levenberg-Marquardt damping for GN")
    p.add_argument("--w-rot", type=float, default=1e-3, help="Smoothness weight for rotations")
    p.add_argument("--w-trans", type=float, default=1e-3, help="Smoothness weight for translations")
    p.add_argument("--seed-translations", action="store_true", help="Phase-correlation init for dx,dz at coarsest level")
    p.add_argument("--log-summary", action="store_true", help="Print per-outer summaries (FISTA loss, alignment loss before/after)")
    p.add_argument("--log-compact", dest="log_compact", action="store_true", default=True,
                   help="Use compact one-line per-outer summary when --log-summary is set (default: on)")
    p.add_argument("--no-log-compact", dest="log_compact", action="store_false")
    p.add_argument("--recon-L", type=float, default=None, help="Fixed Lipschitz constant for FISTA inside alignment (skip power-method)")
    # Data term / similarity
    p.add_argument(
        "--loss",
        choices=[
            "l2","charbonnier","huber","cauchy","barron","student_t","correntropy",
            "zncc","ssim","ms-ssim","mi","nmi","renyi_mi",
            "grad_l1","edge_l2","ngf","grad_orient","phasecorr","fft_mag","chamfer_edge",
            "l2_otsu","ssim_otsu","tversky","swd","mind","pwls","poisson"
        ],
        default="l2",
        help="Data term / similarity to optimize",
    )
    p.add_argument("--loss-param", action="append", default=[], help="Loss parameter as k=v (repeatable), e.g., delta=1.0, eps=1e-3, window=7, temp=0.5")
    # Early stopping controls (alignment phase)
    es = p.add_mutually_exclusive_group()
    es.add_argument("--early-stop", dest="early_stop", action="store_true", help="Enable early stopping across outers (default)")
    es.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="Disable early stopping across outers")
    p.set_defaults(early_stop=True)
    p.add_argument("--early-stop-rel", type=float, default=None, help="Relative improvement threshold for early stop (default 1e-3)")
    p.add_argument("--early-stop-patience", type=int, default=None, help="Consecutive outers below threshold before stopping (default 2)")
    p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help="JAX transfer guard mode during compute (default: off; use log/disallow when debugging)",
    )
    p.add_argument("--out", required=True, help="Output .nxs with recon and alignment params")
    p.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available")
    args = p.parse_args()

    setup_logging(); log_jax_env()
    _init_jax_compilation_cache()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"
    # Parse loss params (k=v -> float)
    loss_params: dict[str, float] = {}
    for kv in args.loss_param:
        if "=" not in kv:
            raise SystemExit(f"--loss-param must be k=v, got: {kv}")
        k, v = kv.split("=", 1)
        try:
            loss_params[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--loss-param value must be numeric: {kv}")

    meta = load_nxtomo(args.data)
    grid, detector, geom = build_geometry(meta)
    proj = jnp.asarray(meta["projections"], dtype=jnp.float32)

    # Hidden defaults: stream one view at a time; unroll=1
    vpb_est = 1

    # Resolve default gather dtype lazily at runtime
    from ..utils.memory import default_gather_dtype as _default_gather_dtype
    _gather = str(args.gather_dtype)
    if _gather == "auto":
        _gather = _default_gather_dtype()

    cfg = AlignConfig(
        outer_iters=args.outer_iters,
        recon_iters=args.recon_iters,
        lambda_tv=args.lambda_tv,
        tv_prox_iters=int(args.tv_prox_iters),
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
        views_per_batch=int(vpb_est),
        projector_unroll=1,
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=_gather,
        opt_method=str(args.opt_method),
        gn_damping=float(args.gn_damping),
        w_rot=float(args.w_rot),
        w_trans=float(args.w_trans),
        loss_kind=str(args.loss),
        loss_params=loss_params if loss_params else None,
        seed_translations=bool(args.seed_translations),
        log_summary=bool(args.log_summary),
        log_compact=bool(args.log_compact),
        recon_L=(float(args.recon_L) if args.recon_L is not None else None),
        early_stop=bool(args.early_stop),
        early_stop_rel_impr=(float(args.early_stop_rel) if args.early_stop_rel is not None else 1e-3),
        early_stop_patience=(int(args.early_stop_patience) if args.early_stop_patience is not None else 2),
    )
    if args.levels is not None and len(args.levels) > 0:
        from ..align.pipeline import align_multires
        with _transfer_guard_ctx(args.transfer_guard):
            x, params5, info = align_multires(geom, grid, detector, proj, factors=args.levels, cfg=cfg)
    else:
        with _transfer_guard_ctx(args.transfer_guard):
            x, params5, info = align(geom, grid, detector, proj, cfg=cfg)

    # Avoid copying projections back from device: reuse host array from metadata
    save_nxtomo(
        args.out,
        projections=meta["projections"],
        thetas_deg=np.asarray(meta["thetas_deg"]),
        grid=meta.get("grid"),
        detector=meta.get("detector"),
        geometry_type=meta.get("geometry_type", "parallel"),
        geometry_meta=meta.get("geometry_meta"),
        volume=np.asarray(x),
        align_params=np.asarray(params5),
        frame=str(meta.get("frame", "sample")),
    )
    logging.info("Saved alignment results to %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
