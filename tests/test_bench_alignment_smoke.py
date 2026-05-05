from __future__ import annotations

import argparse
import logging
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tomojax.bench import alignment_smoke


def test_alignment_smoke_metrics_include_mse_rmse_and_psnr() -> None:
    truth = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    recon = np.asarray([0.0, 1.5, 2.0], dtype=np.float32)

    metrics = alignment_smoke._metrics(recon, truth)

    assert metrics["mse"] > 0.0
    assert metrics["rmse"] > 0.0
    assert metrics["psnr_db"] > 0.0


def test_alignment_smoke_parses_alignment_log_summary() -> None:
    log = "\n".join(
        [
            "Recon (FISTA) | time 1.25s",
            "Align | time 2.50s | loss 10.0->7.5 (-25.0%)",
            "Loss 10.0 -> 7.5 (-25.0%)",
            "Recon (FISTA) | time 1.75s",
            "Align | time 3.50s | loss 7.5->6.0 (-20.0%)",
        ]
    )

    parsed = alignment_smoke._parse_alignment_log(log)

    assert parsed["initial_loss"] == 10.0
    assert parsed["final_loss"] == 6.0
    assert parsed["delta_percent"] == -40.0
    assert parsed["recon_time_sec"] == 3.0
    assert parsed["align_time_sec"] == 6.0
    assert len(parsed["align_steps"]) == 2


def test_alignment_smoke_pose_recovery_metrics_apply_translation_gauge(tmp_path) -> None:
    truth_path = tmp_path / "misaligned.nxs"
    params_path = tmp_path / "params.json"
    theta = np.asarray(
        [
            [0.1, -0.2, 0.3, 1.0, 3.0],
            [0.2, -0.1, 0.4, 3.0, 7.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(truth_path, "w") as handle:
        handle.create_dataset("entry/processing/tomojax/align/thetas", data=theta)
    params_path.write_text(
        """
{
  "schema": "tomojax.alignment_params.v1",
  "views": [
    {"view_index": 0, "alpha_rad": 0.1, "beta_rad": -0.2, "phi_rad": 0.3, "dx_px": -1.0, "dz_px": -2.0},
    {"view_index": 1, "alpha_rad": 0.2, "beta_rad": -0.1, "phi_rad": 0.4, "dx_px": 1.0, "dz_px": 2.0}
  ],
  "gauge_fix": {"mode": "mean_translation"}
}
""",
        encoding="utf-8",
    )

    metrics = alignment_smoke._pose_recovery_metrics(
        truth_path=truth_path,
        params_json=params_path,
    )

    assert metrics["translation_gauge"] == "mean_translation"
    assert metrics["rot_rmse_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["trans_rmse_px"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["initial_trans_rmse_px"] > 0.0


def test_alignment_smoke_projection_residual_l2_rmse_per_ray() -> None:
    metrics = alignment_smoke._projection_residual_metrics(
        final_loss=8.0,
        loss_kind="l2",
        projection_shape=(2, 2, 4),
    )

    assert metrics["n_rays"] == 16
    assert metrics["rmse_per_ray"] == pytest.approx(1.0)


def test_alignment_smoke_in_process_align_preserves_cli_shape(monkeypatch) -> None:
    from tomojax.cli import align as align_cli

    calls: list[str] = []
    old_argv = list(sys.argv)

    monkeypatch.setattr(align_cli, "_build_parser", lambda: argparse.ArgumentParser())
    monkeypatch.setattr(
        align_cli,
        "parse_args_with_config",
        lambda parser, argv, required: (
            argparse.Namespace(progress=False),
            {"effective_options": {}},
        ),
    )
    monkeypatch.setattr(align_cli, "log_jax_env", lambda: logging.info("JAX backend: test"))
    monkeypatch.setattr(align_cli, "_init_jax_compilation_cache", lambda: None)
    monkeypatch.setattr(
        align_cli,
        "_build_align_cli_run_plan",
        lambda parser, args, config_metadata: SimpleNamespace(name="plan"),
    )
    monkeypatch.setattr(
        align_cli,
        "_make_align_cli_checkpoint_callbacks",
        lambda plan: SimpleNamespace(single=lambda *a, **k: None, multires=lambda *a, **k: None),
    )
    monkeypatch.setattr(
        align_cli,
        "_execute_alignment_plan",
        lambda plan, single_checkpoint_callback, multires_checkpoint_callback: SimpleNamespace(
            name="execution"
        ),
    )

    def fake_write(plan, execution):
        calls.append(f"{plan.name}:{execution.name}")
        logging.info("Saved alignment results to out.nxs")

    monkeypatch.setattr(align_cli, "_write_alignment_outputs", fake_write)

    result = alignment_smoke._run_align_in_process(["--data", "in.nxs", "--out", "out.nxs"])

    assert result.returncode == 0
    assert calls == ["plan:execution"]
    assert "JAX backend: test" in result.stdout
    assert "Saved alignment results to out.nxs" in result.stdout
    assert sys.argv == old_argv
