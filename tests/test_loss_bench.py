from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tomojax.cli import loss_bench
from tomojax.data.io_hdf5 import LoadedNXTomo


def test_loss_bench_skipped_loss_does_not_compute_metrics_or_emit_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expdir = tmp_path / "exp"
    mis_path = expdir / "misaligned.nxs"
    gt_path = expdir / "gt.nxs"

    meta_mis = {
        "align_params": np.zeros((2, 5), dtype=np.float32),
        "detector": {"nu": 4, "nv": 4, "du": 1.0, "dv": 1.0, "det_center": (0.0, 0.0)},
        "grid": {"nx": 4, "ny": 4, "nz": 1, "vx": 1.0, "vy": 1.0, "vz": 1.0},
        "thetas_deg": np.asarray([0.0, 90.0], dtype=np.float32),
        "geometry_type": "parallel",
        "projections": np.zeros((2, 4, 4), dtype=np.float32),
    }

    monkeypatch.setattr(loss_bench, "_make_gt_dataset", lambda *args, **kwargs: str(gt_path))
    monkeypatch.setattr(
        loss_bench, "_make_misaligned_dataset", lambda *args, **kwargs: str(mis_path)
    )
    monkeypatch.setattr(
        loss_bench,
        "load_nxtomo",
        lambda path: LoadedNXTomo.from_dataset(meta_mis),
    )
    monkeypatch.setattr(loss_bench, "setup_logging", lambda: None)
    monkeypatch.setattr(loss_bench, "log_jax_env", lambda: None)

    def fail_metrics(*args, **kwargs):
        raise AssertionError("metrics should not be computed for skipped losses")

    monkeypatch.setattr(loss_bench, "_metrics_abs", fail_metrics)
    monkeypatch.setattr(loss_bench, "_metrics_relative", fail_metrics)
    monkeypatch.setattr(loss_bench, "_metrics_gauge_fixed", fail_metrics)
    monkeypatch.setattr(loss_bench, "_gt_projection_mse", fail_metrics)

    monkeypatch.setattr(
        "sys.argv",
        [
            "loss_bench",
            "--expdir",
            str(expdir),
            "--losses",
            "mi_kde",
            "--gt-metric",
            "none",
        ],
    )

    loss_bench.main()

    results = json.loads((expdir / "results.json").read_text(encoding="utf-8"))
    assert results["results"] == [
        {
            "loss": "mi_kde",
            "status": "skipped",
            "seconds": pytest.approx(results["results"][0]["seconds"]),
            "log": "logs/mi_kde.log",
            "output": None,
            "error": None,
        }
    ]
