from __future__ import annotations

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
