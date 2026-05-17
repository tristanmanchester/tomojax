"""Report helpers for manual SPDHG experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_spdhg_experiment_report(indir: Path, *, out: Path | None = None) -> Path:
    """Build a Markdown report from an SPDHG experiment metrics directory."""
    root = Path(indir)
    metrics_path = root / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"expected JSON object in {metrics_path}")

    out_md = Path(out) if out is not None else root / "REPORT.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(spdhg_experiment_markdown(metrics), encoding="utf-8")
    return out_md


def spdhg_experiment_markdown(metrics: dict[str, Any]) -> str:
    """Return a Markdown report for SPDHG experiment metrics."""
    dataset = metrics.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    lines = [
        "# CT Benchmark Report\n",
        (
            f"Dataset: {dataset.get('nx')}x{dataset.get('ny')}x{dataset.get('nz')}, "
            f"views={dataset.get('n_views')}, phantom={dataset.get('phantom')}\n"
        ),
        "\n## Metrics\n",
        _metric_row(metrics, "fbp"),
        _metric_row(metrics, "fista"),
        _metric_row(metrics, "spdhg"),
        "\n## Timing (seconds)\n",
    ]
    timing = metrics.get("timing_sec", {})
    if not isinstance(timing, dict):
        timing = {}
    lines.append(
        f"- FBP: {timing.get('fbp')}\n"
        f"- FISTA: {timing.get('fista')}\n"
        f"- SPDHG: {timing.get('spdhg')}"
    )
    lines.append(
        "\n## Figures\n"
        "See slices: fbp_slices.png, fista_slices.png, spdhg_slices.png; "
        "differences: diff_center_z.png\n"
    )
    return "\n".join(lines)


def _metric_row(metrics: dict[str, Any], name: str) -> str:
    summary = metrics.get(name, {})
    if not isinstance(summary, dict):
        summary = {}
    return (
        f"- {name.upper()}: PSNR={summary.get('psnr')}, "
        f"SSIM_center={summary.get('ssim_center')}, "
        f"MSE={summary.get('mse')}, TV={summary.get('tv')}"
    )


__all__ = ["build_spdhg_experiment_report", "spdhg_experiment_markdown"]
