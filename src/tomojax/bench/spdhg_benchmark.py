"""Reusable helpers for manual SPDHG reconstruction benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import numpy as np

try:
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
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


from tomojax.datasets import SimConfig
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.io import NXTomoMetadata, save_nxtomo


@dataclass(frozen=True)
class SpdhgGeometryBundle:
    data: dict[str, Any]
    projections: Any
    grid: Grid
    detector: Detector
    geometry: ParallelGeometry | LaminographyGeometry
    ground_truth: np.ndarray | None


@dataclass(frozen=True)
class SpdhgReconstructionResults:
    volumes: dict[str, np.ndarray]
    timing_sec: dict[str, float]
    fista_info: Any
    spdhg_info: Any


@dataclass(frozen=True)
class SpdhgDatasetSimulationPlan:
    sim_path: str
    cfg: SimConfig
    gather_dtype: str
    sim_block: int
    progress: bool


@dataclass(frozen=True)
class SpdhgSimulationGeometryBundle:
    grid: Grid
    detector: Detector
    geometry: ParallelGeometry | LaminographyGeometry
    geometry_meta: dict[str, Any] | None
    thetas_deg: np.ndarray
    volume: Any


_EXPECTED_FALLBACK_FAILURE_SNIPPETS = (
    "allocator",
    "cuda_error_out_of_memory",
    "cudnn_status_alloc_failed",
    "failed to allocate",
    "memory allocation",
    "out of memory",
    "resource_exhausted",
)


def is_expected_spdhg_fallback_failure(exc: Exception) -> bool:
    """Return whether a failed benchmark step should use the resource fallback path."""
    if isinstance(exc, MemoryError):
        return True
    if not isinstance(exc, (RuntimeError, TimeoutError, OSError)):
        return False
    msg = str(exc).lower()
    return any(snippet in msg for snippet in _EXPECTED_FALLBACK_FAILURE_SNIPPETS)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def psnr3d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return float(
        peak_signal_noise_ratio(y, x, data_range=max(float(y.max()) - float(y.min()), 1e-6))
    )


def ssim_center_slices(x: np.ndarray, y: np.ndarray, n_slices: int = 5) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    nz = x.shape[2]
    zs = np.linspace(nz // 4, 3 * nz // 4, num=n_slices, dtype=int)
    vals = [
        structural_similarity(y[:, :, zi], x[:, :, zi], data_range=float(y.max() - y.min()))
        for zi in zs
    ]
    return float(np.mean(vals))


def total_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, axis=0, append=x[-1:, :, :])
    dy = np.diff(x, axis=1, append=x[:, -1:, :])
    dz = np.diff(x, axis=2, append=x[:, :, -1:])
    tv = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz + 1e-8))
    return float(tv)


def save_volume(
    out_path: str, data: dict[str, Any], vol: np.ndarray, frame: str = "sample"
) -> None:
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
    axs[2].imshow(v[:, :, :].mean(axis=2).T, cmap="gray", origin="lower")
    axs[2].set_title("mean over z")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_spdhg_benchmark_metrics(
    args: Any,
    bundle: SpdhgGeometryBundle,
    results: SpdhgReconstructionResults,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
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


def write_spdhg_benchmark_report(
    args: Any,
    bundle: SpdhgGeometryBundle,
    results: SpdhgReconstructionResults,
    metrics: dict[str, Any],
) -> None:
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

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
            for ax, name in zip(axs, ["fbp", "fista", "spdhg"], strict=False):
                sl = results.volumes[name][:, :, zc].T
                im = ax.imshow(sl - gt_slice, cmap="coolwarm", vmin=-0.3, vmax=0.3, origin="lower")
                ax.set_title(f"{name} - GT (z={zc})")
                ax.axis("off")
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
            fig.tight_layout()
            fig.savefig(diff_path, dpi=150)
            plt.close(fig)

    with open(os.path.join(args.outdir, "REPORT.txt"), "w") as f:
        f.write("CT Reconstruction Benchmark (FBP/FISTA/SPDHG)\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n")
    print(f"[done] results written to {args.outdir}")


__all__ = [
    "SpdhgDatasetSimulationPlan",
    "SpdhgGeometryBundle",
    "SpdhgReconstructionResults",
    "SpdhgSimulationGeometryBundle",
    "compute_spdhg_benchmark_metrics",
    "ensure_dir",
    "is_expected_spdhg_fallback_failure",
    "psnr3d",
    "save_slice_png",
    "save_volume",
    "ssim_center_slices",
    "total_variation",
    "write_spdhg_benchmark_report",
]
