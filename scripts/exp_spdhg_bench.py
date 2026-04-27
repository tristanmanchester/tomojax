from __future__ import annotations

"""Ad hoc SPDHG experiment driver.

This script owns manual benchmark exploration for one experiment family. Keep this
surface task-specific; promote helpers to ``src/tomojax/bench/`` only after they have
multiple stable callers, and keep fixed profile policy in ``bench/`` instead of here.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Any
import sys

import numpy as np
import jax
import jax.numpy as jnp

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
except ModuleNotFoundError:
    def mean_squared_error(image_true: np.ndarray, image_test: np.ndarray) -> float:
        err = np.asarray(image_true, dtype=np.float32) - np.asarray(image_test, dtype=np.float32)
        return float(np.mean(err * err))

    def peak_signal_noise_ratio(
        image_true: np.ndarray,
        image_test: np.ndarray,
        *,
        data_range: float,
    ) -> float:
        mse = mean_squared_error(image_true, image_test)
        if mse == 0.0:
            return float("inf")
        return float(20.0 * np.log10(float(data_range)) - 10.0 * np.log10(mse))

    def structural_similarity(
        image_true: np.ndarray,
        image_test: np.ndarray,
        *,
        data_range: float,
    ) -> float:
        mse = mean_squared_error(image_true, image_test)
        if mse == 0.0:
            return 1.0
        denom = max(float(data_range) ** 2, 1e-6)
        return float(np.clip(1.0 - mse / denom, -1.0, 1.0))

from tomojax.data.simulate import SimConfig, make_phantom
from tomojax.core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.data.io_hdf5 import NXTomoMetadata, save_nxtomo
from tomojax.utils.fov import cylindrical_mask_xy
from tomojax.utils.subprocesses import run_command
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import fista_tv
from tomojax.recon.spdhg_tv import spdhg_tv, SPDHGConfig


@dataclass(frozen=True)
class GeometryBundle:
    data: Dict[str, Any]
    projections: jnp.ndarray
    grid: Grid
    detector: Detector
    geometry: ParallelGeometry | LaminographyGeometry
    ground_truth: np.ndarray | None


@dataclass(frozen=True)
class ReconstructionResults:
    volumes: Dict[str, np.ndarray]
    timing_sec: Dict[str, float]
    fista_info: Any
    spdhg_info: Any


def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def psnr3d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return float(peak_signal_noise_ratio(y, x, data_range=max(float(y.max()) - float(y.min()), 1e-6)))


def ssim_center_slices(x: np.ndarray, y: np.ndarray, n_slices: int = 5) -> float:
    # Average SSIM over central z slices; convert to float32
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    nz = x.shape[2]
    zs = np.linspace(nz // 4, 3 * nz // 4, num=n_slices, dtype=int)
    vals = []
    for zi in zs:
        vals.append(structural_similarity(y[:, :, zi], x[:, :, zi], data_range=float(y.max() - y.min())))
    return float(np.mean(vals))


def total_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, axis=0, append=x[-1:, :, :])
    dy = np.diff(x, axis=1, append=x[:, -1:, :])
    dz = np.diff(x, axis=2, append=x[:, :, -1:])
    tv = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz + 1e-8))
    return float(tv)


def save_volume(out_path: str, data: Dict[str, Any], vol: np.ndarray, frame: str = "sample") -> None:
    save_meta = NXTomoMetadata.from_dataset(data)
    save_meta.volume = np.asarray(vol)
    save_meta.frame = frame
    save_nxtomo(
        out_path,
        projections=np.asarray(data["projections"]),
        metadata=save_meta,
    )


def save_slice_png(out_path: str, vol: np.ndarray, title: str = "slice") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        with open(out_path, "wb") as f:
            f.write(b"matplotlib unavailable\n")
        return

    v = np.asarray(vol, dtype=np.float32)
    ny = v.shape[1]
    zi = v.shape[2] // 2
    yi = ny // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(v[:, :, zi].T, cmap="gray", origin="lower")
    axs[0].set_title(f"z-slice z={zi}")
    axs[1].imshow(v[:, yi, :].T, cmap="gray", origin="lower")
    axs[1].set_title(f"y-slice y={yi}")
    axs[2].imshow(v[ :, :, :].mean(axis=2).T, cmap="gray", origin="lower")
    axs[2].set_title("mean over z")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run 3D CT reconstruction benchmark (FBP/FISTA/SPDHG)")
    ap.add_argument("--outdir", default="runs/exp_spdhg_256", help="Output directory for runs and reports")
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--nu", type=int, default=256)
    ap.add_argument("--nv", type=int, default=256)
    ap.add_argument("--n-views", type=int, default=512)
    ap.add_argument("--phantom", default="random_shapes", choices=["shepp","cube","sphere","blobs","random_shapes"])
    ap.add_argument("--overwrite-data", action="store_true", help="Force re-simulation of dataset even if it exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", default="none", choices=["none","gaussian","poisson"])
    ap.add_argument("--noise-level", type=float, default=0.0)
    ap.add_argument("--roi", choices=["off","auto","cube","bbox"], default="auto")
    ap.add_argument("--mask-vol", choices=["off","cyl"], default="cyl")
    ap.add_argument("--progress", action="store_true")

    # Algorithm params (tuned for 256^3 on a single GPU; adjust as needed)
    ap.add_argument("--fista-iters", type=int, default=60)
    ap.add_argument("--fista-lambda", type=float, default=5e-3)
    ap.add_argument("--fista-tv-iters", type=int, default=10)

    ap.add_argument("--spdhg-iters", type=int, default=300)
    ap.add_argument("--spdhg-lambda", type=float, default=5e-3)
    ap.add_argument("--spdhg-block", type=int, default=64)
    ap.add_argument("--spdhg-theta", type=float, default=0.5)
    ap.add_argument("--spdhg-manual-steps", action="store_true", help="Use conservative manual tau/sigma instead of auto")
    ap.add_argument("--gather-dtype", choices=["auto","fp32","bf16","fp16"], default="auto")
    ap.add_argument("--sim-block", type=int, default=32, help="Views per chunk for simulation (chunked-vmap)")
    ap.add_argument("--fbp-on-cpu", action="store_true", help="Run FBP in a CPU subprocess if GPU FFT plan fails")

    # Random-shapes phantom controls (effective when phantom=random_shapes)
    ap.add_argument("--n-cubes", type=int, default=96)
    ap.add_argument("--n-spheres", type=int, default=96)
    ap.add_argument("--min-size", type=int, default=6)
    ap.add_argument("--max-size", type=int, default=28)
    ap.add_argument("--max-rot-deg", type=float, default=180.0)
    ap.add_argument("--min-value", type=float, default=0.1)
    ap.add_argument("--max-value", type=float, default=1.0)

    return ap.parse_args(argv)


def prepare_or_load_dataset(args: argparse.Namespace) -> str:
    sim_path = os.path.join(args.outdir, "dataset.nxs")
    if args.overwrite_data or not os.path.exists(sim_path):
        cfg = SimConfig(
            nx=args.nx, ny=args.ny, nz=args.nz,
            nu=args.nu, nv=args.nv, n_views=args.n_views,
            phantom=args.phantom,
            noise=args.noise, noise_level=args.noise_level,
            seed=args.seed,
            n_cubes=args.n_cubes,
            n_spheres=args.n_spheres,
            min_size=args.min_size,
            max_size=args.max_size,
            max_rot_deg=args.max_rot_deg,
            min_value=args.min_value,
            max_value=args.max_value,
        )
        print(f"[simulate] generating dataset {args.nx}^3, {args.n_views} views with {args.n_cubes}+{args.n_spheres} shapes → {sim_path}")
        try:
            # Chunked, memory-friendly simulation to avoid OOM from CUDA command buffers and giant vmaps
            # Build geometry and phantom like simulate(), but project in view chunks
            grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.vx, cfg.vy, cfg.vz)
            det = Detector(cfg.nu, cfg.nv, cfg.du, cfg.dv, det_center=(0.0, 0.0))
            total_deg = 180.0 if cfg.geometry == "parallel" else 360.0
            thetas = np.linspace(0.0, total_deg, cfg.n_views, endpoint=False).astype(np.float32)
            if cfg.geometry == "parallel":
                geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
                geometry_meta = None
            else:
                geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=thetas, tilt_deg=cfg.tilt_deg, tilt_about=cfg.tilt_about)
                geometry_meta = {"tilt_deg": float(cfg.tilt_deg), "tilt_about": str(cfg.tilt_about)}

            vol = make_phantom(cfg)
            from tomojax.utils.memory import default_gather_dtype
            gather = default_gather_dtype() if args.gather_dtype == "auto" else args.gather_dtype
            T_all = jnp.stack([jnp.asarray(geom.pose_for_view(i), jnp.float32) for i in range(cfg.n_views)], axis=0)
            det_grid = get_detector_grid_device(det)
            b = max(1, int(args.sim_block))

            # Jitted windowed projector: always slices (b,4,4) window and returns full window + valid count
            n = int(cfg.n_views)
            def project_window(start_idx: int):
                i = jnp.int32(start_idx)
                n32 = jnp.int32(n)
                b32 = jnp.int32(b)
                remaining = jnp.maximum(0, n32 - i)
                valid = jnp.minimum(b32, remaining)
                shift = b32 - valid
                start_shifted = jnp.maximum(0, i - shift)
                T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
                f = lambda T: forward_project_view_T(
                    T,
                    grid,
                    det,
                    vol,
                    use_checkpoint=True,
                    gather_dtype=gather,
                    det_grid=det_grid,
                )
                chunk_b = jax.vmap(f, in_axes=0)(T_chunk)
                return chunk_b, valid

            project_window_jit = jax.jit(project_window)

            projs_np = []
            for s in range(0, n, b):
                chunk_b, valid = project_window_jit(s)
                cb = np.asarray(chunk_b)
                v = int(valid)
                cb = cb[b - v : b, :, :] if v < b else cb
                projs_np.append(cb)
                if args.progress:
                    e = min(s + b, n)
                    print(f"simulate chunk {s}:{e}/{n}")
            proj = np.concatenate(projs_np, axis=0)
            # Noise
            rng = np.random.default_rng(cfg.seed)
            if cfg.noise == "gaussian" and cfg.noise_level > 0:
                proj = proj + rng.normal(scale=cfg.noise_level, size=proj.shape).astype(np.float32)
            elif cfg.noise == "poisson" and cfg.noise_level > 0:
                s = cfg.noise_level
                lam = np.maximum(0.0, proj) * s
                noisy = rng.poisson(lam=lam).astype(np.float32) / max(s, 1e-6)
                proj = noisy

            save_meta = NXTomoMetadata(
                thetas_deg=thetas,
                grid=grid.to_dict(),
                detector=det.to_dict(),
                geometry_type=cfg.geometry,
                geometry_meta=geometry_meta,
                volume=np.asarray(vol),
                frame="sample",
            )
            save_nxtomo(
                sim_path,
                projections=proj,
                metadata=save_meta,
            )
        except Exception as e:
            # Fallback: run simulate in a clean subprocess (separate JAX context) to avoid OOM
            print(f"[simulate] in-process chunked path failed ({e}); falling back to CLI simulate…")
            cmd = [
                sys.executable,
                "-m",
                "tomojax.cli.simulate",
                "--out", sim_path,
                "--nx", str(cfg.nx), "--ny", str(cfg.ny), "--nz", str(cfg.nz),
                "--nu", str(cfg.nu), "--nv", str(cfg.nv), "--n-views", str(cfg.n_views),
                "--geometry", cfg.geometry,
                "--phantom", cfg.phantom,
                "--n-cubes", str(cfg.n_cubes), "--n-spheres", str(cfg.n_spheres),
                "--min-size", str(cfg.min_size), "--max-size", str(cfg.max_size),
                "--min-value", str(cfg.min_value), "--max-value", str(cfg.max_value),
                "--max-rot-deg", str(cfg.max_rot_deg),
                "--seed", str(cfg.seed),
            ]
            env = os.environ.copy()
            env.pop("TOMOJAX_PROGRESS", None)  # prefer fast vmapped path
            # Optionally avoid command buffers if backend struggles
            env.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
            print("[simulate] launching:", " ".join(cmd))
            run_command(cmd, check=True, env=env)  # nosec B603
    else:
        print(f"[simulate] reusing dataset at {sim_path}")
    return sim_path


def load_geometry_bundle(sim_path: str) -> GeometryBundle:
    from tomojax.data.io_hdf5 import load_nxtomo

    data = load_nxtomo(sim_path)
    proj = jnp.asarray(data["projections"], dtype=jnp.float32)
    grid_d = data["grid"]
    det_d = data["detector"]
    grid = Grid(
        nx=grid_d["nx"],
        ny=grid_d["ny"],
        nz=grid_d["nz"],
        vx=grid_d["vx"],
        vy=grid_d["vy"],
        vz=grid_d["vz"],
    )
    det = Detector(
        nu=det_d["nu"],
        nv=det_d["nv"],
        du=det_d["du"],
        dv=det_d["dv"],
        det_center=tuple(det_d.get("det_center", (0.0, 0.0))),
    )
    if data.get("geometry_type", "parallel") == "parallel":
        geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=data["thetas_deg"])
    else:
        geometry_meta = data.get("geometry_meta", {})
        geom = LaminographyGeometry(
            grid=grid,
            detector=det,
            thetas_deg=data["thetas_deg"],
            tilt_deg=float(geometry_meta.get("tilt_deg", 30.0)),
            tilt_about=str(geometry_meta.get("tilt_about", "x")),
        )
    gt_raw = data.get("volume")
    gt = None if gt_raw is None else np.asarray(gt_raw)
    return GeometryBundle(
        data=data,
        projections=proj,
        grid=grid,
        detector=det,
        geometry=geom,
        ground_truth=gt,
    )


def prepare_volume_mask(
    args: argparse.Namespace,
    grid: Grid,
    detector: Detector,
) -> tuple[np.ndarray | None, jnp.ndarray | None]:
    vol_mask_np = None
    if args.mask_vol == "cyl":
        try:
            m_xy = cylindrical_mask_xy(grid, detector)
            vol_mask_np = np.asarray(m_xy, dtype=np.float32)[:, :, None]
        except Exception:
            vol_mask_np = None
    vol_mask = None if vol_mask_np is None else jnp.asarray(vol_mask_np)
    return vol_mask_np, vol_mask


def resolve_gather_dtype(args: argparse.Namespace) -> str:
    gather = str(args.gather_dtype)
    if gather == "auto":
        from tomojax.utils.memory import default_gather_dtype

        gather = default_gather_dtype()
    return gather


def run_reconstructions(
    args: argparse.Namespace,
    bundle: GeometryBundle,
    *,
    gather: str,
    vol_mask_np: np.ndarray | None,
    vol_mask: jnp.ndarray | None,
) -> ReconstructionResults:
    t0 = time.perf_counter()

    def run_fbp_gpu():
        return fbp(
            bundle.geometry, bundle.grid, bundle.detector, bundle.projections,
            filter_name="ramp",
            views_per_batch=1,
            projector_unroll=1,
            checkpoint_projector=True,
            gather_dtype=gather,
        )
    vol_fbp = None
    try:
        if args.fbp_on_cpu:
            raise RuntimeError("force_cpu")
        vol_fbp = run_fbp_gpu()
    except Exception as e:
        # Fallback: run FBP via CLI on CPU, then load volume back
        print(f"[fbp] GPU path failed ({e}); falling back to CPU subprocess…")
        fbp_tmp = os.path.join(args.outdir, "fbp_cpu_tmp.nxs")
        cmd = [
            sys.executable,
            "-m", "tomojax.cli.recon",
            "--data", os.path.join(args.outdir, "dataset.nxs"),
            "--algo", "fbp",
            "--filter", "ramp",
            "--out", fbp_tmp,
        ]
        env = os.environ.copy(); env["JAX_PLATFORM_NAME"] = "cpu"
        run_command(cmd, check=True, env=env)  # nosec B603
        import h5py
        with h5py.File(fbp_tmp, "r") as f:
            vol_fbp = np.asarray(f["/entry/processing/tomojax/volume"])  # zyx on disk by default
    if vol_mask_np is not None:
        vol_fbp = vol_fbp * vol_mask
    fbp_time = time.perf_counter() - t0
    save_volume(os.path.join(args.outdir, "fbp.nxs"), bundle.data, np.asarray(vol_fbp))
    save_slice_png(os.path.join(args.outdir, "fbp_slices.png"), np.asarray(vol_fbp), title="FBP slices")

    t0 = time.perf_counter()
    vol_fista, info_fista = fista_tv(
        bundle.geometry, bundle.grid, bundle.detector, bundle.projections,
        iters=int(args.fista_iters),
        lambda_tv=float(args.fista_lambda),
        views_per_batch=1,
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype=gather,
        tv_prox_iters=int(args.fista_tv_iters),
        vol_mask=vol_mask,
    )
    fista_time = time.perf_counter() - t0
    save_volume(os.path.join(args.outdir, "fista.nxs"), bundle.data, np.asarray(vol_fista))
    save_slice_png(os.path.join(args.outdir, "fista_slices.png"), np.asarray(vol_fista), title="FISTA slices")

    spdhg_cfg = SPDHGConfig(
        iters=int(args.spdhg_iters),
        lambda_tv=float(args.spdhg_lambda),
        theta=float(args.spdhg_theta),
        views_per_batch=int(max(1, args.spdhg_block)),
        seed=int(args.seed),
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype=gather,
        positivity=True,
        support=vol_mask if vol_mask is not None else None,
        log_every=1,
    )
    if args.spdhg_manual_steps:
        # Conservative steps (avoid heavy A-norm power method). Tuned to be safe.
        spdhg_cfg = spdhg_cfg.__class__(
            **{**asdict(spdhg_cfg), "tau": 0.02, "sigma_data": 0.25, "sigma_tv": 0.25}
        )

    t0 = time.perf_counter()
    vol_spdhg, info_spdhg = spdhg_tv(
        bundle.geometry,
        bundle.grid,
        bundle.detector,
        bundle.projections,
        config=spdhg_cfg,
    )
    spdhg_time = time.perf_counter() - t0
    save_volume(os.path.join(args.outdir, "spdhg.nxs"), bundle.data, np.asarray(vol_spdhg))
    save_slice_png(os.path.join(args.outdir, "spdhg_slices.png"), np.asarray(vol_spdhg), title="SPDHG slices")

    return ReconstructionResults(
        volumes={
            "fbp": np.asarray(vol_fbp),
            "fista": np.asarray(vol_fista),
            "spdhg": np.asarray(vol_spdhg),
        },
        timing_sec={"fbp": float(fbp_time), "fista": float(fista_time), "spdhg": float(spdhg_time)},
        fista_info=info_fista,
        spdhg_info=info_spdhg,
    )


def compute_metrics(
    args: argparse.Namespace,
    bundle: GeometryBundle,
    results: ReconstructionResults,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "dataset": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "nu": args.nu,
            "nv": args.nv,
            "n_views": args.n_views,
            "phantom": args.phantom,
            "noise": args.noise,
            "noise_level": args.noise_level,
        }
    }
    gt = bundle.ground_truth
    for name, vol in results.volumes.items():
        if gt is not None:
            metrics[name] = {
                "psnr": psnr3d(vol, gt),
                "ssim_center": ssim_center_slices(vol, gt, n_slices=5),
                "mse": float(mean_squared_error(gt.astype(np.float32), vol.astype(np.float32))),
                "tv": total_variation(vol),
            }
        else:
            metrics[name] = {
                "psnr": None,
                "ssim_center": None,
                "mse": None,
                "tv": total_variation(vol),
            }
    metrics["timing_sec"] = results.timing_sec
    metrics["fista_info"] = results.fista_info
    metrics["spdhg_info"] = results.spdhg_info
    return metrics


def write_report(
    args: argparse.Namespace,
    bundle: GeometryBundle,
    results: ReconstructionResults,
    metrics: Dict[str, Any],
) -> None:
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Difference images (central z slice)
    plt = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    gt = bundle.ground_truth
    if gt is not None:
        diff_path = os.path.join(args.outdir, "diff_center_z.png")
        if plt is None:
            with open(diff_path, "wb") as f:
                f.write(b"matplotlib unavailable\n")
        else:
            zc = gt.shape[2] // 2
            gt_slice = gt[:, :, zc].T
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            for ax, name in zip(axs, ["fbp", "fista", "spdhg"]):
                sl = results.volumes[name][:, :, zc].T
                im = ax.imshow(sl - gt_slice, cmap="coolwarm", vmin=-0.3, vmax=0.3, origin="lower")
                ax.set_title(f"{name} − GT (z={zc})")
                ax.axis("off")
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
            fig.tight_layout()
            fig.savefig(diff_path, dpi=150)
            plt.close(fig)

    # Simple text report
    with open(os.path.join(args.outdir, "REPORT.txt"), "w") as f:
        f.write("CT Reconstruction Benchmark (FBP/FISTA/SPDHG)\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n")
    print(f"[done] results written to {args.outdir}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dir(args.outdir)
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    sim_path = prepare_or_load_dataset(args)
    bundle = load_geometry_bundle(sim_path)
    vol_mask_np, vol_mask = prepare_volume_mask(args, bundle.grid, bundle.detector)
    gather = resolve_gather_dtype(args)
    results = run_reconstructions(
        args,
        bundle,
        gather=gather,
        vol_mask_np=vol_mask_np,
        vol_mask=vol_mask,
    )
    metrics = compute_metrics(args, bundle, results)
    write_report(args, bundle, results, metrics)


if __name__ == "__main__":
    main()
