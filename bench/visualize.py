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
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _error_limit(error_volume: np.ndarray) -> float:
    finite = error_volume[np.isfinite(error_volume)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, 99.0))
    return max(vmax, 1e-6)


def _trace_points(
    trace: list[dict[str, Any]],
    *,
    value_key: str,
) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    filtered: list[tuple[int, float, int | None]] = []
    for idx, point in enumerate(trace):
        value = point.get(value_key)
        if value is None:
            continue
        try:
            outer_idx = int(point.get("outer_idx", idx + 1))
            metric_value = float(value)
            level_factor = point.get("level_factor")
            filtered.append(
                (
                    outer_idx,
                    metric_value,
                    int(level_factor) if level_factor is not None else None,
                )
            )
        except Exception:
            continue
    if not filtered:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32), []
    xs = np.asarray([x for x, _, _ in filtered], dtype=np.int32)
    ys = np.asarray([y for _, y, _ in filtered], dtype=np.float32)
    levels = [level for _, _, level in filtered]
    return xs, ys, levels


def _plot_trace_metric(
    ax,
    *,
    trace: list[dict[str, Any]],
    value_key: str,
    title: str,
    y_label: str,
    threshold: float | None = None,
) -> None:
    xs, ys, level_values = _trace_points(trace, value_key=value_key)
    if xs.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, f"No {y_label} trace", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.set_title(title)
    ax.plot(
        xs,
        ys,
        color="0.75",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )
    unique_levels: list[int | None] = []
    for level in level_values:
        if level not in unique_levels:
            unique_levels.append(level)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(unique_levels), 1)))
    for color, level in zip(colors, unique_levels):
        level_x = [x for x, lf in zip(xs, level_values) if lf == level]
        level_y = [y for y, lf in zip(ys, level_values) if lf == level]
        if not level_x:
            continue
        label = f"{y_label} (x{level})" if level is not None else y_label
        ax.plot(
            np.asarray(level_x, dtype=np.int32),
            np.asarray(level_y, dtype=np.float32),
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
            zorder=2,
        )
    if threshold is not None:
        ax.axhline(
            float(threshold),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="threshold",
        )
        met_idx = np.where(ys <= float(threshold))[0]
        if met_idx.size > 0:
            idx = int(met_idx[0])
            ax.scatter([xs[idx]], [ys[idx]], color="tab:red", s=40, zorder=3)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def _text_lines(
    *,
    profile_name: str,
    out_path: Path,
    metrics: dict[str, Any],
    quality: dict[str, Any],
    fixture: dict[str, Any],
) -> list[str]:
    level_summaries = list(metrics.get("warm_level_summaries") or [])
    lines = [
        f"profile: {profile_name}",
        f"objective: {metrics.get('objective_name')}={metrics.get('objective_value')}",
        f"gt_mse: {quality.get('gt_mse')}",
        f"warm_gt_mse_median: {metrics.get('warm_gt_mse_median')}",
        f"benchmark_valid: {metrics.get('benchmark_valid')}",
        f"reached_finest_level: {metrics.get('reached_finest_level')}",
        f"warm_mean_s: {metrics.get('warm_run_seconds_mean')}",
        f"warmup_s: {metrics.get('warmup_seconds')}",
        f"warmup_incomplete: {metrics.get('warmup_incomplete')}",
        f"primer_ran: {metrics.get('primer_ran')}",
        f"primer_s: {metrics.get('primer_seconds')}",
        f"time_budget_s: {metrics.get('time_budget_seconds')}",
        f"peak_gpu_mb: {metrics.get('peak_gpu_memory_mb')}",
        f"gpu_scope: {metrics.get('gpu_memory_scope')}",
    ]
    if metrics.get("representative_run_index") is not None:
        lines.append(
            f"representative_run_index: {metrics.get('representative_run_index')}"
        )
    if metrics.get("warmup_stop_reason") is not None:
        lines.append(f"warmup_stop_reason: {metrics.get('warmup_stop_reason')}")
    if metrics.get("primer_reached_finest_level") is not None:
        lines.append(
            "primer_reached_finest_level: "
            f"{metrics.get('primer_reached_finest_level')}"
        )
    if metrics.get("invalid_reason"):
        lines.append(f"invalid_reason: {metrics.get('invalid_reason')}")
    if metrics.get("finest_level_first_elapsed_seconds") is not None:
        lines.append(
            "finest_level_first_s: "
            f"{metrics.get('finest_level_first_elapsed_seconds')}"
        )
    if metrics.get("finest_level_first_outer_idx") is not None:
        lines.append(
            "finest_level_first_outer: "
            f"{metrics.get('finest_level_first_outer_idx')}"
        )
    if metrics.get("quality_threshold_value") is not None:
        lines.append(
            "threshold: "
            f"{metrics.get('quality_threshold_metric')}<={metrics.get('quality_threshold_value')}"
        )
        if metrics.get("quality_threshold_scope") is not None:
            lines.append(f"threshold_scope: {metrics.get('quality_threshold_scope')}")
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
    if metrics.get("quality_contract_met") is not None:
        lines.append(f"quality_contract_met: {metrics.get('quality_contract_met')}")
    if metrics.get("warm_seconds_to_quality_contract") is not None:
        lines.append(
            f"warm_s_to_contract: {metrics.get('warm_seconds_to_quality_contract')}"
        )
    if metrics.get("memory_guard_value_mb") is not None:
        lines.append(f"memory_guard_mb: {metrics.get('memory_guard_value_mb')}")
    if metrics.get("memory_guard_penalty") is not None:
        lines.append(f"memory_guard_penalty: {metrics.get('memory_guard_penalty')}")
    if metrics.get("objective_time_memguard") is not None:
        lines.append(f"objective_time_memguard: {metrics.get('objective_time_memguard')}")
    if quality.get("rot_rmse_deg") is not None:
        lines.append(f"rot_rmse_deg: {quality.get('rot_rmse_deg')}")
    if quality.get("trans_gf_rmse_px") is not None:
        lines.append(f"trans_gf_rmse_px: {quality.get('trans_gf_rmse_px')}")
    elif quality.get("trans_rmse_px") is not None:
        lines.append(f"trans_rmse_px: {quality.get('trans_rmse_px')}")
    if fixture.get("volume_shape") is not None:
        lines.append(f"volume_shape: {fixture.get('volume_shape')}")
    if fixture.get("n_views") is not None:
        lines.append(f"n_views: {fixture.get('n_views')}")
    if level_summaries:
        lines.append("level_split:")
        for item in level_summaries:
            lines.append(
                "  "
                f"x{item.get('level_factor')}: "
                f"{item.get('elapsed_seconds_total')}s, "
                f"{item.get('outer_iters_executed')} iters, "
                f"final_gt_mse={item.get('final_gt_mse')}"
            )
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
    representative_run_index: int | None = None,
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

    right_gs = gs[:, 3].subgridspec(3, 1, height_ratios=[1.0, 1.0, 1.5])
    mse_ax = fig.add_subplot(right_gs[0, 0])
    rmse_ax = fig.add_subplot(right_gs[1, 0])
    text_ax = fig.add_subplot(right_gs[2, 0])
    trace = list(convergence_trace or [])
    if trace and convergence_metric_name:
        _plot_trace_metric(
            mse_ax,
            trace=trace,
            value_key="quality_value",
            title=f"{convergence_metric_name} vs Outer Iteration",
            y_label=convergence_metric_name,
            threshold=quality_threshold_value,
        )
        _plot_trace_metric(
            rmse_ax,
            trace=trace,
            value_key="trans_gf_rmse_px",
            title="trans_gf_rmse_px vs Outer Iteration",
            y_label="trans_gf_rmse_px",
        )
    elif loss_history:
        mse_ax.set_title("Alignment Loss")
        xs = np.arange(1, len(loss_history) + 1, dtype=np.int32)
        mse_ax.plot(xs, loss_history, marker="o", linewidth=1.5)
        mse_ax.set_xlabel("Outer Iteration")
        mse_ax.set_ylabel("Loss")
        mse_ax.grid(True, alpha=0.25)
        rmse_ax.set_title("trans_gf_rmse_px")
        rmse_ax.text(0.5, 0.5, "No position RMSE trace", ha="center", va="center")
        rmse_ax.set_xticks([])
        rmse_ax.set_yticks([])
    else:
        mse_ax.set_title("Alignment Loss")
        mse_ax.text(0.5, 0.5, "No loss history", ha="center", va="center")
        mse_ax.set_xticks([])
        mse_ax.set_yticks([])
        rmse_ax.set_title("trans_gf_rmse_px")
        rmse_ax.text(0.5, 0.5, "No gauge-fixed position RMSE trace", ha="center", va="center")
        rmse_ax.set_xticks([])
        rmse_ax.set_yticks([])

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
