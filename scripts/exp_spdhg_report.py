from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize benchmark results and emit a Markdown report")
    ap.add_argument("--indir", default="runs/exp_spdhg_256", help="Directory with metrics.json and images")
    ap.add_argument("--out", default=None, help="Output Markdown path (default: indir/REPORT.md)")
    args = ap.parse_args()

    metrics_path = os.path.join(args.indir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(metrics_path)
    with open(metrics_path, "r") as f:
        m: Dict[str, Any] = json.load(f)

    out_md = args.out or os.path.join(args.indir, "REPORT.md")
    lines = []
    ds = m.get("dataset", {})
    lines.append(f"# CT Benchmark Report\n")
    lines.append(f"Dataset: {ds.get('nx')}x{ds.get('ny')}x{ds.get('nz')}, views={ds.get('n_views')}, phantom={ds.get('phantom')}\n")
    lines.append("\n## Metrics\n")
    def fmt_row(name):
        s = m.get(name, {})
        return f"- {name.upper()}: PSNR={s.get('psnr')}, SSIM_center={s.get('ssim_center')}, MSE={s.get('mse')}, TV={s.get('tv')}"
    lines.append(fmt_row("fbp"))
    lines.append(fmt_row("fista"))
    lines.append(fmt_row("spdhg"))
    lines.append("\n## Timing (seconds)\n")
    t = m.get("timing_sec", {})
    lines.append(f"- FBP: {t.get('fbp')}\n- FISTA: {t.get('fista')}\n- SPDHG: {t.get('spdhg')}")
    lines.append("\n## Figures\n")
    lines.append("See slices: fbp_slices.png, fista_slices.png, spdhg_slices.png; differences: diff_center_z.png\n")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

