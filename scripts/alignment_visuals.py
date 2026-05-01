from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import numpy as np

from tomojax.recon.quicklook import scale_to_uint8


@dataclass(frozen=True)
class VisualScenario:
    slug: str
    title: str
    geometry_dofs: tuple[str, ...]
    hidden_det_u_px: float = 0.0
    hidden_det_v_px: float = 0.0
    hidden_detector_roll_deg: float = 0.0
    hidden_axis_rot_x_deg: float = 0.0
    hidden_axis_rot_y_deg: float = 0.0
    nominal_tilt_deg: float = 30.0
    true_tilt_deg: float = 30.0


@dataclass(frozen=True)
class VisualProfile:
    views: int
    levels: tuple[int, ...]
    outer_iters: int
    early_stop: bool
    early_stop_profile: str = "compute_saving"


@dataclass(frozen=True)
class AlignmentVisualizationPayload:
    scenario: VisualScenario
    profile: VisualProfile
    theta_span: float
    truth: np.ndarray
    naive_fbp: np.ndarray
    calibrated_fbp: np.ndarray
    aligned_tv: np.ndarray
    estimates: dict[str, Any]
    metrics: dict[str, float]
    diagnostics: Any
    outer_stats: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class NaiveVisualizationPayload:
    scenario: VisualScenario
    truth: np.ndarray
    naive_fbp: np.ndarray


def slice_xy(volume: np.ndarray) -> np.ndarray:
    return volume[:, :, volume.shape[2] // 2].T


def slice_xz(volume: np.ndarray) -> np.ndarray:
    return volume[:, volume.shape[1] // 2, :].T


def slice_yz(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2, :, :].T


def ortho_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "XY": slice_xy(volume),
        "XZ": slice_xz(volume),
        "YZ": slice_yz(volume),
    }


def scale_gray(image: np.ndarray) -> np.ndarray:
    return scale_to_uint8(image, lower_percentile=1.0, upper_percentile=99.7)


def to_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    return arr.astype(np.uint8)


def scale_shared_gray(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.nanmin(image))
        upper = float(np.nanmax(image))
    if upper <= lower:
        return np.zeros((*image.shape, 3), dtype=np.uint8)
    scaled = np.clip((np.asarray(image, dtype=np.float32) - lower) / (upper - lower), 0.0, 1.0)
    gray = np.rint(scaled * 255.0).astype(np.uint8)
    return to_rgb(gray)


def scale_diverging(image: np.ndarray, clip: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if not np.isfinite(clip) or clip <= 1e-12:
        clip = float(np.nanpercentile(np.abs(arr), 99.0))
    if not np.isfinite(clip) or clip <= 1e-12:
        return np.full((*arr.shape, 3), 245, dtype=np.uint8)
    x = np.clip(arr / clip, -1.0, 1.0)
    rgb = np.empty((*arr.shape, 3), dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    rgb[pos, 0] = 255.0
    rgb[pos, 1] = 255.0 * (1.0 - x[pos])
    rgb[pos, 2] = 255.0 * (1.0 - x[pos])
    rgb[neg, 0] = 255.0 * (1.0 + x[neg])
    rgb[neg, 1] = 255.0 * (1.0 + x[neg])
    rgb[neg, 2] = 255.0
    return np.rint(rgb).astype(np.uint8)


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    image = to_rgb(image)
    if image.shape[0] == height:
        return image
    out = np.full((height, image.shape[1], 3), 20, dtype=np.uint8)
    out[: image.shape[0], : image.shape[1], :] = image
    return out


def hstack_rgb(images: list[np.ndarray], *, pad: int = 6) -> np.ndarray:
    height = max(im.shape[0] for im in images)
    padded = [pad_to_height(im, height) for im in images]
    width = sum(im.shape[1] for im in padded) + pad * (len(padded) - 1)
    out = np.full((height, width, 3), 20, dtype=np.uint8)
    x = 0
    for image in padded:
        out[:, x : x + image.shape[1], :] = image
        x += image.shape[1] + pad
    return out


def vstack_rgb(images: list[np.ndarray], *, pad: int = 8) -> np.ndarray:
    images = [to_rgb(im) for im in images]
    width = max(im.shape[1] for im in images)
    height = sum(im.shape[0] for im in images) + pad * (len(images) - 1)
    out = np.full((height, width, 3), 20, dtype=np.uint8)
    y = 0
    for image in images:
        out[y : y + image.shape[0], : image.shape[1], :] = image
        y += image.shape[0] + pad
    return out


def text_panel(width: int, height: int, lines: Sequence[str], *, title: str = "") -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    y = 8
    if title:
        draw.text((10, y), title[:60], fill=(245, 245, 245))
        y += 22
    for line in lines:
        if y > height - 18:
            break
        draw.text((10, y), str(line)[:72], fill=(220, 220, 220))
        y += 18
    return np.asarray(img, dtype=np.uint8)


def _metric_text(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(val):
        return "n/a"
    return f"{val:.4g}"


def _scenario_truth_payload(scenario: VisualScenario) -> dict[str, float]:
    return {
        "det_u_px": float(scenario.hidden_det_u_px),
        "det_v_px": float(scenario.hidden_det_v_px),
        "detector_roll_deg": float(scenario.hidden_detector_roll_deg),
        "axis_rot_x_deg": float(scenario.hidden_axis_rot_x_deg),
        "axis_rot_y_deg": float(scenario.hidden_axis_rot_y_deg),
        "nominal_tilt_deg": float(scenario.nominal_tilt_deg),
        "true_tilt_deg": float(scenario.true_tilt_deg),
    }


def _geometry_status_label(diagnostics: Any) -> str:
    if not isinstance(diagnostics, dict):
        return ""
    overall = diagnostics.get("overall_status")
    if isinstance(overall, str) and overall:
        return overall
    blocks = diagnostics.get("blocks")
    if not isinstance(blocks, list):
        return ""
    statuses = [
        str(block.get("status"))
        for block in blocks
        if isinstance(block, dict) and block.get("status")
    ]
    if not statuses:
        return ""
    if "ill_conditioned" in statuses:
        return "ill_conditioned"
    if "underconverged" in statuses:
        return "underconverged"
    if all(status == "converged" for status in statuses):
        return "converged"
    return ",".join(statuses)


def diagnostics_panel(
    scenario: VisualScenario,
    *,
    profile: VisualProfile,
    theta_span: float,
    estimates: dict[str, Any],
    metrics: dict[str, float],
    diagnostics: Any,
    width: int,
    height: int,
) -> np.ndarray:
    status = _geometry_status_label(diagnostics)
    hidden = _scenario_truth_payload(scenario)
    lines = [
        f"title: {scenario.title}",
        f"dofs: {','.join(scenario.geometry_dofs) or 'none'}",
        f"status: {status or 'control'}",
        f"span/views: {theta_span:g} deg / {profile.views}",
        f"levels: {' '.join(str(v) for v in profile.levels)}",
        f"iters/early: {profile.outer_iters} / {profile.early_stop} ({profile.early_stop_profile})",
        "",
        "hidden",
        f"det_u={hidden['det_u_px']:.3g} det_v={hidden['det_v_px']:.3g}",
        f"roll={hidden['detector_roll_deg']:.3g}",
        f"axis=({hidden['axis_rot_x_deg']:.3g},{hidden['axis_rot_y_deg']:.3g})",
        f"tilt true={hidden['true_tilt_deg']:.3g}",
        "",
        "estimated",
        f"det_u={estimates.get('det_u_px', 0.0):.3g} det_v={estimates.get('det_v_px', 0.0):.3g}",
        f"roll={estimates.get('detector_roll_deg', 0.0):.3g}",
        f"axis=({estimates.get('axis_rot_x_deg', 0.0):.3g},{estimates.get('axis_rot_y_deg', 0.0):.3g})",
        "",
        "NMSE",
        f"naive={_metric_text(metrics.get('naive_volume_nmse'))}",
        f"calib={_metric_text(metrics.get('calibrated_volume_nmse'))}",
        f"aligned={_metric_text(metrics.get('aligned_tv_volume_nmse'))}",
    ]
    return text_panel(width, height, lines, title=scenario.slug)


def loss_panel(outer_stats: Sequence[Mapping[str, Any]], *, width: int = 360, height: int = 220) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), "geometry loss / initial loss by level", fill=(20, 20, 20))

    stats = [
        dict(stat)
        for stat in outer_stats
        if isinstance(stat, Mapping) and stat.get("geometry_block")
    ]
    if not stats:
        draw.text((10, 42), "no geometry loss stats", fill=(80, 80, 80))
        return np.asarray(img, dtype=np.uint8)

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for stat in stats:
        key = (int(stat.get("level_factor", 0)), str(stat.get("geometry_block", "geometry")))
        groups.setdefault(key, []).append(stat)

    left, top, right, bottom = 46, 34, width - 14, height - 34
    draw.rectangle((left, top, right, bottom), outline=(180, 180, 180))
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
    ]
    draw.text((10, top - 5), "1.0", fill=(80, 80, 80))
    draw.text((12, bottom - 8), "0", fill=(80, 80, 80))
    max_len = max(len(v) for v in groups.values())
    legend_y = height - 28
    for idx, ((level, block), items) in enumerate(sorted(groups.items(), reverse=True)):
        color = colors[idx % len(colors)]
        first = float(items[0].get("geometry_loss_before", items[0].get("geometry_loss_after", 1.0)))
        if not np.isfinite(first) or abs(first) < 1e-12:
            first = 1.0
        points: list[tuple[int, int]] = []
        for j, stat in enumerate(items):
            loss = float(stat.get("geometry_loss_after", stat.get("geometry_loss_before", first)))
            y_norm = np.clip(loss / first, 0.0, 1.2)
            x = int(left + (right - left) * (j / max(1, max_len - 1)))
            y = int(bottom - (bottom - top) * min(float(y_norm), 1.0))
            points.append((x, y))
            accepted = bool(stat.get("geometry_accepted", False))
            r = 3
            if accepted:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            else:
                draw.line((x - r, y - r, x + r, y + r), fill=color)
                draw.line((x - r, y + r, x + r, y - r), fill=color)
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        label = f"L{level} {block}"[:22]
        lx = 10 + (idx % 2) * 170
        ly = legend_y + (idx // 2) * 14
        if ly < height - 8:
            draw.text((lx, ly), label, fill=color)
    return np.asarray(img, dtype=np.uint8)


def shared_intensity_limits(volumes: Sequence[np.ndarray]) -> tuple[float, float]:
    samples = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in volumes])
    lower = float(np.nanpercentile(samples, 1.0))
    upper = float(np.nanpercentile(samples, 99.7))
    return lower, upper


def difference_clip(images: Sequence[np.ndarray]) -> float:
    samples = np.concatenate([np.abs(np.asarray(v, dtype=np.float32)).reshape(-1) for v in images])
    return float(np.nanpercentile(samples, 99.0))


def image_grid(
    rows: Sequence[Sequence[np.ndarray]],
    *,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    scale: int = 3,
    pad: int = 8,
) -> np.ndarray:
    try:
        import matplotlib
    except ModuleNotFoundError:
        return _pil_image_grid(
            rows,
            title=title,
            row_labels=row_labels,
            col_labels=col_labels,
            scale=scale,
            pad=pad,
        )

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    image_rows = [[to_rgb(image) for image in row] for row in rows]

    if len(row_labels) != len(image_rows):
        raise ValueError("row_labels must match image grid row count")
    if len(col_labels) != max(len(row) for row in image_rows):
        raise ValueError("col_labels must match image grid column count")

    nrows = len(image_rows)
    ncols = len(col_labels)
    cell_px = max(max(image.shape[:2]) for row in image_rows for image in row) * max(scale, 1)
    title_font = max(10.0, min(18.0, cell_px * 0.044))
    col_font = max(9.0, min(15.0, cell_px * 0.036))
    row_font = max(8.5, min(13.0, cell_px * 0.032))
    title_px = int(round(title_font * 2.2))
    col_label_px = int(round(col_font * 2.1))
    gutter_px = max(4, int(pad))
    fig_w = ncols * cell_px + (ncols - 1) * gutter_px
    fig_h = title_px + gutter_px + col_label_px + gutter_px + nrows * cell_px + (nrows - 1) * gutter_px
    dpi = 100
    fig = Figure(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi, facecolor=(0.045, 0.045, 0.045))
    canvas = FigureCanvasAgg(fig)

    fig.text(
        0.0,
        1.0 - 7 / fig_h,
        title,
        color=(0.9, 0.9, 0.9),
        fontsize=title_font,
        fontweight="semibold",
        va="top",
        ha="left",
    )

    for col, label in enumerate(col_labels):
        x0 = col * (cell_px + gutter_px)
        fig.text(
            (x0 + 2) / fig_w,
            1.0 - (title_px + gutter_px + 5) / fig_h,
            label,
            color=(0.78, 0.78, 0.78),
            fontsize=col_font,
            fontweight="semibold",
            va="top",
            ha="left",
        )

    image_top = title_px + gutter_px + col_label_px + gutter_px
    for row_idx, row in enumerate(image_rows):
        y0 = image_top + row_idx * (cell_px + gutter_px)
        for col_idx, image in enumerate(row):
            x0 = col_idx * (cell_px + gutter_px)
            ax = fig.add_axes(
                [
                    x0 / fig_w,
                    1.0 - (y0 + cell_px) / fig_h,
                    cell_px / fig_w,
                    cell_px / fig_h,
                ]
            )
            ax.imshow(image, interpolation="nearest")
            ax.set_axis_off()
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.35)
                spine.set_edgecolor((0.19, 0.19, 0.19))
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.96,
                    row_labels[row_idx],
                    color=(0.94, 0.94, 0.94),
                    fontsize=row_font,
                    fontweight="semibold",
                    va="top",
                    ha="left",
                    transform=ax.transAxes,
                    bbox={
                        "boxstyle": "round,pad=0.22,rounding_size=0.06",
                        "facecolor": (0.02, 0.02, 0.02, 0.50),
                        "edgecolor": "none",
                    },
                )

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[:, :, :3])


def _pil_image_grid(
    rows: Sequence[Sequence[np.ndarray]],
    *,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    scale: int = 3,
    pad: int = 8,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image_rows = [[to_rgb(image) for image in row] for row in rows]
    if len(row_labels) != len(image_rows):
        raise ValueError("row_labels must match image grid row count")
    if len(col_labels) != max(len(row) for row in image_rows):
        raise ValueError("col_labels must match image grid column count")

    nrows = len(image_rows)
    ncols = len(col_labels)
    cell_px = max(max(image.shape[:2]) for row in image_rows for image in row) * max(scale, 1)
    title_px = 34
    col_label_px = 24
    gutter_px = max(4, int(pad))
    width = ncols * cell_px + (ncols - 1) * gutter_px
    height = title_px + gutter_px + col_label_px + gutter_px + nrows * cell_px + (nrows - 1) * gutter_px
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 7), title[:120], fill=(230, 230, 230))

    for col, label in enumerate(col_labels):
        x = col * (cell_px + gutter_px) + 2
        draw.text((x, title_px + gutter_px + 5), label, fill=(198, 198, 198))

    image_top = title_px + gutter_px + col_label_px + gutter_px
    for row_idx, row in enumerate(image_rows):
        y = image_top + row_idx * (cell_px + gutter_px)
        for col_idx, image in enumerate(row):
            x = col_idx * (cell_px + gutter_px)
            tile = Image.fromarray(image).resize((cell_px, cell_px), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + cell_px - 1, y + cell_px - 1), outline=(48, 48, 48))
            if col_idx == 0:
                draw.rectangle((x + 4, y + 4, x + 130, y + 24), fill=(5, 5, 5))
                draw.text((x + 8, y + 7), row_labels[row_idx][:20], fill=(240, 240, 240))
    return np.asarray(canvas, dtype=np.uint8)


def save_ortho_slices(
    out_dir: Path,
    prefix: str,
    slices: Mapping[str, np.ndarray],
    *,
    lower: float | None = None,
    upper: float | None = None,
    diff_clip: float | None = None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for plane, image in slices.items():
        key = f"{prefix}_{plane.lower()}"
        path = out_dir / f"{key}.png"
        if diff_clip is None:
            if lower is None or upper is None:
                panel = scale_gray(image)
            else:
                panel = scale_shared_gray(image, lower, upper)
        else:
            panel = scale_diverging(image, diff_clip)
        iio.imwrite(path, panel)
        paths[key] = str(path)
    return paths


def write_alignment_visuals(
    payload: AlignmentVisualizationPayload,
    *,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = payload.scenario
    truth = payload.truth
    naive_fbp = payload.naive_fbp
    calibrated_fbp = payload.calibrated_fbp
    aligned_tv = payload.aligned_tv
    truth_s = ortho_slices(truth)
    naive_s = ortho_slices(naive_fbp)
    calibrated_s = ortho_slices(calibrated_fbp)
    aligned_s = ortho_slices(aligned_tv)
    diff_calib_truth = ortho_slices(calibrated_fbp - truth)
    diff_aligned_truth = ortho_slices(aligned_tv - truth)
    diff_aligned_naive = ortho_slices(aligned_tv - naive_fbp)
    lower, upper = shared_intensity_limits([truth, naive_fbp, calibrated_fbp, aligned_tv])
    diff_clip = difference_clip(
        list(diff_calib_truth.values())
        + list(diff_aligned_truth.values())
        + list(diff_aligned_naive.values())
    )
    inspection_panel = image_grid(
        [
            [scale_shared_gray(truth_s[name], lower, upper) for name in ("XY", "XZ", "YZ")],
            [scale_shared_gray(naive_s[name], lower, upper) for name in ("XY", "XZ", "YZ")],
            [scale_shared_gray(aligned_s[name], lower, upper) for name in ("XY", "XZ", "YZ")],
            [scale_diverging((naive_s[name] - truth_s[name]), diff_clip) for name in ("XY", "XZ", "YZ")],
            [
                scale_diverging((aligned_s[name] - truth_s[name]), diff_clip)
                for name in ("XY", "XZ", "YZ")
            ],
        ],
        title=f"{scenario.slug}: {scenario.title}",
        row_labels=("GT", "naive", "aligned TV", "naive-GT", "aligned-GT"),
        col_labels=("XY", "XZ", "YZ"),
        scale=3,
        pad=8,
    )
    rendered_loss_panel = loss_panel(payload.outer_stats, width=360, height=220)
    rendered_diagnostics_panel = diagnostics_panel(
        scenario,
        profile=payload.profile,
        theta_span=payload.theta_span,
        estimates=payload.estimates,
        metrics=payload.metrics,
        diagnostics=payload.diagnostics,
        width=360,
        height=420,
    )
    truth_xy = scale_gray(slice_xy(truth))
    naive_xy = scale_gray(slice_xy(naive_fbp))
    calibrated_xy = scale_gray(slice_xy(calibrated_fbp))
    tv_xy = scale_gray(slice_xy(aligned_tv))
    truth_orthos = hstack_rgb([scale_gray(truth_s[name]) for name in ("XY", "XZ", "YZ")])
    calibrated_orthos = hstack_rgb([scale_gray(calibrated_s[name]) for name in ("XY", "XZ", "YZ")])
    diff_calib_truth_panel = hstack_rgb(
        [scale_diverging(diff_calib_truth[name], diff_clip) for name in ("XY", "XZ", "YZ")]
    )
    diff_aligned_truth_panel = hstack_rgb(
        [scale_diverging(diff_aligned_truth[name], diff_clip) for name in ("XY", "XZ", "YZ")]
    )
    diff_aligned_naive_panel = hstack_rgb(
        [scale_diverging(diff_aligned_naive[name], diff_clip) for name in ("XY", "XZ", "YZ")]
    )
    paths = {
        "truth_xy": str(out_dir / "truth_xy.png"),
        "naive_fbp_xy": str(out_dir / "naive_fbp_xy.png"),
        "calibrated_fbp_xy": str(out_dir / "calibrated_fbp_xy.png"),
        "aligned_tv_xy": str(out_dir / "aligned_tv_xy.png"),
        "before_after_panel": str(out_dir / "before_after_panel.png"),
        "inspection_panel": str(out_dir / "inspection_panel.png"),
        "loss_panel": str(out_dir / "loss_panel.png"),
        "diagnostics_panel": str(out_dir / "diagnostics_panel.png"),
        "truth_orthos": str(out_dir / "truth_orthos.png"),
        "calibrated_orthos": str(out_dir / "calibrated_fbp_orthos.png"),
        "absolute_difference_xy": str(out_dir / "absolute_difference_xy.png"),
        "difference_calibrated_truth_orthos": str(out_dir / "difference_calibrated_truth_orthos.png"),
        "difference_aligned_truth_orthos": str(out_dir / "difference_aligned_truth_orthos.png"),
        "difference_aligned_naive_orthos": str(out_dir / "difference_aligned_naive_orthos.png"),
    }
    paths.update(save_ortho_slices(out_dir, "truth", truth_s, lower=lower, upper=upper))
    paths.update(save_ortho_slices(out_dir, "naive_fbp", naive_s, lower=lower, upper=upper))
    paths.update(save_ortho_slices(out_dir, "calibrated_fbp", calibrated_s, lower=lower, upper=upper))
    paths.update(save_ortho_slices(out_dir, "aligned_tv", aligned_s, lower=lower, upper=upper))
    paths.update(
        save_ortho_slices(
            out_dir,
            "difference_calibrated_truth",
            diff_calib_truth,
            diff_clip=diff_clip,
        )
    )
    paths.update(
        save_ortho_slices(
            out_dir,
            "difference_aligned_truth",
            diff_aligned_truth,
            diff_clip=diff_clip,
        )
    )
    paths.update(
        save_ortho_slices(
            out_dir,
            "difference_aligned_naive",
            diff_aligned_naive,
            diff_clip=diff_clip,
        )
    )
    paths.update(
        save_ortho_slices(
            out_dir,
            "difference_naive_truth",
            ortho_slices(naive_fbp - truth),
            diff_clip=diff_clip,
        )
    )
    iio.imwrite(paths["truth_xy"], truth_xy)
    iio.imwrite(paths["naive_fbp_xy"], naive_xy)
    iio.imwrite(paths["calibrated_fbp_xy"], calibrated_xy)
    iio.imwrite(paths["aligned_tv_xy"], tv_xy)
    iio.imwrite(paths["before_after_panel"], inspection_panel)
    iio.imwrite(paths["inspection_panel"], inspection_panel)
    iio.imwrite(paths["loss_panel"], rendered_loss_panel)
    iio.imwrite(paths["diagnostics_panel"], rendered_diagnostics_panel)
    iio.imwrite(paths["truth_orthos"], truth_orthos)
    iio.imwrite(paths["calibrated_orthos"], calibrated_orthos)
    iio.imwrite(paths["absolute_difference_xy"], diff_calib_truth_panel)
    iio.imwrite(paths["difference_calibrated_truth_orthos"], diff_calib_truth_panel)
    iio.imwrite(paths["difference_aligned_truth_orthos"], diff_aligned_truth_panel)
    iio.imwrite(paths["difference_aligned_naive_orthos"], diff_aligned_naive_panel)
    return paths


def write_naive_visuals(
    payload: NaiveVisualizationPayload,
    *,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = payload.scenario
    truth = payload.truth
    naive_fbp = payload.naive_fbp
    truth_s = ortho_slices(truth)
    naive_s = ortho_slices(naive_fbp)
    diff_s = ortho_slices(naive_fbp - truth)
    lower, upper = shared_intensity_limits([truth, naive_fbp])
    diff_clip = difference_clip(list(diff_s.values()))
    inspection_panel = image_grid(
        [
            [scale_shared_gray(truth_s[name], lower, upper) for name in ("XY", "XZ", "YZ")],
            [scale_shared_gray(naive_s[name], lower, upper) for name in ("XY", "XZ", "YZ")],
            [scale_diverging(diff_s[name], diff_clip) for name in ("XY", "XZ", "YZ")],
        ],
        title=f"{scenario.slug}: {scenario.title}",
        row_labels=("GT", "naive", "naive-GT"),
        col_labels=("XY", "XZ", "YZ"),
        scale=3,
        pad=8,
    )
    rendered_diagnostics_panel = text_panel(
        360,
        220,
        [
            scenario.title,
            f"dofs: {','.join(scenario.geometry_dofs) or 'none'}",
            f"hidden det_u={scenario.hidden_det_u_px:g}",
            f"hidden roll={scenario.hidden_detector_roll_deg:g}",
            f"hidden axis=({scenario.hidden_axis_rot_x_deg:g},{scenario.hidden_axis_rot_y_deg:g})",
        ],
        title=scenario.slug,
    )
    truth_xy = scale_gray(slice_xy(truth))
    naive_xy = scale_gray(slice_xy(naive_fbp))
    truth_orthos = hstack_rgb([scale_gray(truth_s[name]) for name in ("XY", "XZ", "YZ")])
    naive_orthos = hstack_rgb([scale_gray(naive_s[name]) for name in ("XY", "XZ", "YZ")])
    diff_panel = hstack_rgb([scale_diverging(diff_s[name], diff_clip) for name in ("XY", "XZ", "YZ")])
    paths = {
        "truth_xy": str(out_dir / "truth_xy.png"),
        "naive_fbp_xy": str(out_dir / "naive_fbp_xy.png"),
        "calibrated_fbp_xy": "",
        "aligned_tv_xy": "",
        "before_after_panel": str(out_dir / "truth_vs_naive_panel.png"),
        "inspection_panel": str(out_dir / "truth_vs_naive_panel.png"),
        "loss_panel": "",
        "diagnostics_panel": str(out_dir / "diagnostics_panel.png"),
        "truth_orthos": str(out_dir / "truth_orthos.png"),
        "calibrated_orthos": str(out_dir / "naive_fbp_orthos.png"),
        "absolute_difference_xy": str(out_dir / "truth_naive_absolute_difference_xy.png"),
        "difference_calibrated_truth_orthos": "",
        "difference_aligned_truth_orthos": "",
        "difference_aligned_naive_orthos": str(out_dir / "difference_naive_truth_orthos.png"),
    }
    paths.update(save_ortho_slices(out_dir, "truth", truth_s, lower=lower, upper=upper))
    paths.update(save_ortho_slices(out_dir, "naive_fbp", naive_s, lower=lower, upper=upper))
    paths.update(save_ortho_slices(out_dir, "difference_naive_truth", diff_s, diff_clip=diff_clip))
    iio.imwrite(paths["truth_xy"], truth_xy)
    iio.imwrite(paths["naive_fbp_xy"], naive_xy)
    iio.imwrite(paths["before_after_panel"], inspection_panel)
    iio.imwrite(paths["inspection_panel"], inspection_panel)
    iio.imwrite(paths["diagnostics_panel"], rendered_diagnostics_panel)
    iio.imwrite(paths["truth_orthos"], truth_orthos)
    iio.imwrite(paths["calibrated_orthos"], naive_orthos)
    iio.imwrite(paths["absolute_difference_xy"], diff_panel)
    iio.imwrite(paths["difference_aligned_naive_orthos"], diff_panel)
    return paths


def resize_for_master(image: np.ndarray, *, width: int) -> np.ndarray:
    from PIL import Image

    arr = to_rgb(image)
    if arr.shape[1] <= width:
        return arr
    scale = float(width) / float(arr.shape[1])
    height = max(1, int(round(arr.shape[0] * scale)))
    pil = Image.fromarray(arr)
    return np.asarray(pil.resize((int(width), height), Image.Resampling.LANCZOS), dtype=np.uint8)
