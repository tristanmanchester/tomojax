from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np


def _central_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    nx, ny, nz = volume.shape
    return {
        "xy": volume[:, :, nz // 2],
        "xz": volume[:, ny // 2, :],
        "yz": volume[nx // 2, :, :],
    }


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _error_limit(error_volume: np.ndarray) -> float:
    finite = error_volume[np.isfinite(error_volume)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, 99.0))
    return max(vmax, 1e-6)


def _text_lines(
    *,
    profile_name: str,
    out_path: Path,
    metrics: dict[str, Any],
    quality: dict[str, Any],
    fixture: dict[str, Any],
) -> list[str]:
    lines = [
        f"profile: {profile_name}",
        f"objective: {metrics.get('objective_name')}={metrics.get('objective_value')}",
        f"gt_mse: {quality.get('gt_mse')}",
        f"warm_mean_s: {metrics.get('warm_run_seconds_mean')}",
        f"peak_gpu_mb: {metrics.get('peak_gpu_memory_mb')}",
        f"gpu_scope: {metrics.get('gpu_memory_scope')}",
    ]
    if metrics.get("quality_threshold_value") is not None:
        lines.append(
            "threshold: "
            f"{metrics.get('quality_threshold_metric')}<={metrics.get('quality_threshold_value')}"
        )
        lines.append(f"threshold_met: {metrics.get('quality_threshold_met')}")
        lines.append(f"stopped_on_threshold: {metrics.get('stopped_on_threshold')}")
        lines.append(f"stopped_on_plateau: {metrics.get('stopped_on_plateau')}")
        lines.append(f"stopped_on_budget: {metrics.get('stopped_on_budget')}")
        lines.append(f"final_stop_reason: {metrics.get('final_stop_reason')}")
        lines.append(f"final_stop_level: {metrics.get('final_stop_level_factor')}")
        lines.append(
            "first_threshold_level: "
            f"{metrics.get('first_threshold_crossing_level_factor')}"
        )
        lines.append(
            f"warm_s_to_threshold: {metrics.get('warm_seconds_to_quality_threshold')}"
        )
    if quality.get("rot_rms_deg") is not None:
        lines.append(f"rot_rms_deg: {quality.get('rot_rms_deg')}")
    if quality.get("trans_rms_px") is not None:
        lines.append(f"trans_rms_px: {quality.get('trans_rms_px')}")
    if fixture.get("volume_shape") is not None:
        lines.append(f"volume_shape: {fixture.get('volume_shape')}")
    if fixture.get("n_views") is not None:
        lines.append(f"n_views: {fixture.get('n_views')}")
    lines.append(f"artifact: {out_path.name}")
    return lines


def save_alignment_summary(
    *,
    out_path: Path,
    profile_name: str,
    gt_volume: np.ndarray,
    baseline_volume: np.ndarray,
    final_volume: np.ndarray,
    loss_history: list[float],
    convergence_trace: list[dict[str, Any]] | None,
    convergence_metric_name: str | None,
    quality_threshold_value: float | None,
    metrics: dict[str, Any],
    quality: dict[str, Any],
    fixture: dict[str, Any],
) -> Path:
    gt_volume = np.asarray(gt_volume, dtype=np.float32)
    baseline_volume = np.asarray(baseline_volume, dtype=np.float32)
    final_volume = np.asarray(final_volume, dtype=np.float32)
    error_volume = np.abs(final_volume - gt_volume)

    gt_slices = _central_slices(gt_volume)
    baseline_slices = _central_slices(baseline_volume)
    final_slices = _central_slices(final_volume)
    error_slices = _central_slices(error_volume)

    gt_limits = _display_limits(gt_volume)
    baseline_limits = _display_limits(baseline_volume)
    final_limits = _display_limits(final_volume)
    err_vmax = _error_limit(error_volume)

    fig = plt.figure(figsize=(13.5, 13.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 1.15])

    row_titles = [
        ("Ground Truth", gt_slices, "gray", gt_limits),
        ("Nominal FBP Baseline", baseline_slices, "gray", baseline_limits),
        ("Aligned Result", final_slices, "gray", final_limits),
        ("Abs Error", error_slices, "magma", (0.0, err_vmax)),
    ]
    planes = ["xy", "xz", "yz"]

    for row_idx, (row_title, slices, cmap, limits) in enumerate(row_titles):
        for col_idx, plane in enumerate(planes):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(
                np.asarray(slices[plane]).T,
                cmap=cmap,
                origin="lower",
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
                aspect="auto",
            )
            if row_idx == 0:
                ax.set_title(plane.upper())
            if col_idx == 0:
                ax.set_ylabel(row_title)
            ax.set_xticks([])
            ax.set_yticks([])

    loss_ax = fig.add_subplot(gs[0:2, 3])
    trace = list(convergence_trace or [])
    if trace and convergence_metric_name:
        filtered = [
            (
                int(point.get("outer_idx", idx + 1)),
                float(point["quality_value"]),
                point.get("level_factor"),
            )
            for idx, point in enumerate(trace)
            if point.get("quality_value") is not None
        ]
        xs = np.asarray([x for x, _, _ in filtered], dtype=np.int32)
        ys = np.asarray([y for _, y, _ in filtered], dtype=np.float32)
        if xs.size == ys.size and xs.size > 0:
            loss_ax.set_title(f"{convergence_metric_name} vs Outer Iteration")
            # Draw a faint full trace to show continuity across resolution changes.
            loss_ax.plot(
                xs,
                ys,
                color="0.75",
                linewidth=1.0,
                alpha=0.8,
                zorder=1,
            )
            level_values: list[int | None] = []
            for _, _, level_factor in filtered:
                try:
                    level_values.append(int(level_factor) if level_factor is not None else None)
                except Exception:
                    level_values.append(None)
            unique_levels = []
            for level in level_values:
                if level not in unique_levels:
                    unique_levels.append(level)
            colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(unique_levels), 1)))
            for color, level in zip(colors, unique_levels):
                level_x = [x for x, _, lf in filtered if lf == level]
                level_y = [y for _, y, lf in filtered if lf == level]
                if not level_x:
                    continue
                label = (
                    f"{convergence_metric_name} (x{level})"
                    if level is not None
                    else convergence_metric_name
                )
                loss_ax.plot(
                    np.asarray(level_x, dtype=np.int32),
                    np.asarray(level_y, dtype=np.float32),
                    marker="o",
                    linewidth=2.0,
                    color=color,
                    label=label,
                    zorder=2,
                )
            if quality_threshold_value is not None:
                loss_ax.axhline(
                    float(quality_threshold_value),
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.2,
                    label="threshold",
                )
                met_idx = np.where(ys <= float(quality_threshold_value))[0]
                if met_idx.size > 0:
                    idx = int(met_idx[0])
                    loss_ax.scatter(
                        [xs[idx]],
                        [ys[idx]],
                        color="tab:red",
                        s=40,
                        zorder=3,
                    )
            loss_ax.set_xlabel("Outer Iteration")
            loss_ax.set_ylabel(convergence_metric_name)
            loss_ax.grid(True, alpha=0.25)
            loss_ax.legend(loc="best")
        else:
            loss_ax.text(0.5, 0.5, "No convergence trace", ha="center", va="center")
            loss_ax.set_xticks([])
            loss_ax.set_yticks([])
    elif loss_history:
        loss_ax.set_title("Alignment Loss")
        xs = np.arange(1, len(loss_history) + 1, dtype=np.int32)
        loss_ax.plot(xs, loss_history, marker="o", linewidth=1.5)
        loss_ax.set_xlabel("Outer Iteration")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True, alpha=0.25)
    else:
        loss_ax.set_title("Alignment Loss")
        loss_ax.text(0.5, 0.5, "No loss history", ha="center", va="center")
        loss_ax.set_xticks([])
        loss_ax.set_yticks([])

    text_ax = fig.add_subplot(gs[2:, 3])
    text_ax.axis("off")
    text_ax.set_title("Run Summary", loc="left")
    text_ax.text(
        0.0,
        1.0,
        "\n".join(_text_lines(
            profile_name=profile_name,
            out_path=out_path,
            metrics=metrics,
            quality=quality,
            fixture=fixture,
        )),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
