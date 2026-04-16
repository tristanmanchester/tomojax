from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
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
    monkeypatch.setattr(loss_bench, "_project_gt_with_estimated_poses", fail_metrics)

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


def test_loss_bench_runs_supported_workflow_and_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expdir = tmp_path / "exp"
    gt_path = expdir / "gt.nxs"
    mis_path = expdir / "misaligned.nxs"
    out_path = expdir / "align_l2.nxs"
    saved_outputs: dict[Path, LoadedNXTomo] = {}

    meta_mis = LoadedNXTomo.from_dataset(
        {
            "align_params": np.zeros((4, 5), dtype=np.float32),
            "detector": {"nu": 6, "nv": 1, "du": 1.0, "dv": 1.0, "det_center": (0.0, 0.0)},
            "grid": {"nx": 6, "ny": 6, "nz": 1, "vx": 1.0, "vy": 1.0, "vz": 1.0},
            "thetas_deg": np.asarray([0.0, 45.0, 90.0, 135.0], dtype=np.float32),
            "geometry_type": "parallel",
            "projections": np.zeros((4, 1, 6), dtype=np.float32),
            "volume": np.zeros((6, 6, 1), dtype=np.float32),
            "frame": "sample",
        }
    )

    monkeypatch.setattr(loss_bench, "_make_gt_dataset", lambda *args, **kwargs: str(gt_path))
    monkeypatch.setattr(
        loss_bench, "_make_misaligned_dataset", lambda *args, **kwargs: str(mis_path)
    )

    monkeypatch.setattr(loss_bench, "setup_logging", lambda: None)
    monkeypatch.setattr(loss_bench, "log_jax_env", lambda: None)

    def fake_load_nxtomo(path: str | Path) -> LoadedNXTomo:
        resolved = Path(path)
        if resolved == mis_path:
            return meta_mis
        if resolved in saved_outputs:
            return saved_outputs[resolved]
        raise AssertionError(f"unexpected load_nxtomo path: {resolved}")

    def fake_save_nxtomo(path: str | Path, *, projections, metadata) -> None:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.touch()
        saved_outputs[resolved] = LoadedNXTomo.from_dataset(
            {
                **LoadedNXTomo(projections=np.asarray(projections), metadata=metadata).to_dataset_dict()
            }
        )

    monkeypatch.setattr(loss_bench, "load_nxtomo", fake_load_nxtomo)
    monkeypatch.setattr(loss_bench, "save_nxtomo", fake_save_nxtomo)

    def fake_align_multires(geom, grid, det, projections, *, factors, cfg):
        x = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
        params = jnp.zeros((projections.shape[0], 5), dtype=jnp.float32)
        info = {
            "loss": [1.0],
            "factors": list(factors),
            "outer_stats": [{}],
            "stopped_by_observer": False,
            "observer_action": "continue",
            "total_outer_iters": 1,
            "wall_time_total": 0.0,
        }
        return x, params, info

    import tomojax.align.pipeline as align_pipeline

    monkeypatch.setattr(align_pipeline, "align_multires", fake_align_multires)

    monkeypatch.setattr(
        "sys.argv",
        [
            "loss_bench",
            "--expdir",
            str(expdir),
            "--nx",
            "6",
            "--ny",
            "6",
            "--nz",
            "1",
            "--nu",
            "6",
            "--nv",
            "1",
            "--n-views",
            "4",
            "--outer-iters",
            "1",
            "--recon-iters",
            "1",
            "--losses",
            "l2",
            "--gt-metric",
            "none",
        ],
    )

    loss_bench.main()

    results = json.loads((expdir / "results.json").read_text(encoding="utf-8"))
    record = results["results"][0]
    assert record["loss"] == "l2"
    assert record["status"] == "ok"
    assert record["log"] == "logs/l2.log"
    assert record["output"] == "align_l2.nxs"
    assert record["error"] is None
    assert "gt_mse" not in record
    for key in (
        "seconds",
        "rot_rmse_deg",
        "rot_mse_deg",
        "trans_rmse_px",
        "trans_mse_px",
        "rot_mae_deg",
        "trans_mae_px",
        "rot_rel_rmse_deg",
        "trans_rel_rmse_px",
        "rot_gf_rmse_deg",
        "trans_gf_rmse_px",
    ):
        assert isinstance(record[key], int | float)
        assert np.isfinite(record[key])

    csv_lines = (expdir / "results.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].split(",") == [
        "loss",
        "status",
        "seconds",
        "rot_rmse_deg",
        "rot_mse_deg",
        "trans_rmse_px",
        "trans_mse_px",
        "rot_mae_deg",
        "trans_mae_px",
        "rot_rel_rmse_deg",
        "trans_rel_rmse_px",
        "rot_gf_rmse_deg",
        "trans_gf_rmse_px",
        "gt_mse",
        "log",
        "output",
        "error",
    ]
    assert len(csv_lines) == 2
    assert csv_lines[1].split(",")[0:2] == ["l2", "ok"]
    assert csv_lines[1].split(",")[13:17] == ["", "logs/l2.log", "align_l2.nxs", "None"]

    assert out_path.exists()
    saved = saved_outputs[out_path]
    assert saved.align_params is not None
    assert saved.volume is not None
    np.testing.assert_allclose(np.asarray(saved.align_params), 0.0)
    np.testing.assert_allclose(np.asarray(saved.volume), 0.0)
