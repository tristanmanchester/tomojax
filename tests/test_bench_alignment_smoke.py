from __future__ import annotations

import argparse
import logging
import sys
from types import SimpleNamespace

import numpy as np

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
