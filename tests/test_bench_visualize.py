from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

visualize = importlib.import_module("visualize")


def test_save_alignment_summary_writes_png(tmp_path: Path) -> None:
    gt_volume = np.linspace(0.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
    final_volume = gt_volume * 0.9
    out_path = tmp_path / "align.summary.png"

    written = visualize.save_alignment_summary(
        out_path=out_path,
        profile_name="smoke_align",
        gt_volume=gt_volume,
        final_volume=final_volume,
        loss_history=[5.0, 2.5, 1.25],
        metrics={
            "objective_name": "gt_mse",
            "objective_value": 0.123,
            "warm_run_seconds_mean": 1.5,
            "peak_gpu_memory_mb": 256.0,
            "gpu_memory_scope": "process",
        },
        quality={
            "gt_mse": 0.123,
            "rot_rms_deg": 0.4,
            "trans_rms_px": 1.1,
        },
        fixture={
            "volume_shape": [3, 3, 3],
            "n_views": 12,
        },
    )

    assert written == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
