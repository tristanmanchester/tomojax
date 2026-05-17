from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np

from tomojax.bench import article_visuals
from tomojax.bench.article_alignment_runs import (
    article_scenario_catalog_for_kind,
    article_theta_span_deg,
    diagnostic_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


article_runner = _load_module(
    "generate_alignment_before_after_128_under_test",
    ROOT / "scripts" / "generate_alignment_before_after_128.py",
)


def test_article_visual_scaling_handles_nonfinite_values(tmp_path: Path) -> None:
    image = np.asarray([[np.nan, -np.inf], [0.25, np.inf]], dtype=np.float32)

    gray = article_visuals.scale_shared_gray(image, np.nan, np.nan)
    diverging = article_visuals.scale_diverging(image, np.nan)
    loss = article_visuals.loss_panel(
        [
            {"geometry_block": "setup", "level_factor": 4, "geometry_loss_before": 1.0},
            {
                "geometry_block": "setup",
                "level_factor": 4,
                "geometry_loss_before": 1.0,
                "geometry_loss_after": np.nan,
            },
            {
                "geometry_block": "setup",
                "level_factor": 4,
                "geometry_loss_before": 1.0,
                "geometry_loss_after": 0.5,
                "geometry_accepted": True,
            },
        ]
    )

    assert gray.dtype == np.uint8
    assert diverging.dtype == np.uint8
    assert loss.dtype == np.uint8
    assert np.isfinite(gray).all()
    assert np.isfinite(diverging).all()
    assert np.isfinite(loss).all()

    out = tmp_path / "nonfinite_panel.png"
    iio.imwrite(out, gray)
    assert out.stat().st_size > 0


def test_scenario_finite_report_marks_nonfinite_alignment_volume() -> None:
    result = article_runner.ScenarioComputationResult(
        theta_span=180.0,
        naive_fbp=np.zeros((2, 2, 2), dtype=np.float32),
        calibrated_fbp=np.ones((2, 2, 2), dtype=np.float32),
        aligned_tv=np.full((2, 2, 2), np.nan, dtype=np.float32),
        provenance="estimated",
        supplied={},
        estimates={},
        metrics={
            "naive_volume_nmse": 0.1,
            "calibrated_volume_nmse": 0.2,
            "aligned_tv_volume_nmse": np.nan,
        },
        info={},
        diagnostics={},
        geometry_objectives=[],
        schedule_metadata={},
        executed_stages=[],
        solver_metadata={},
    )

    report = article_runner._scenario_finite_report(result)

    assert report["all_required_finite"] is False
    assert report["first_nonfinite"]["name"] == "aligned_tv"
    aligned_summary = next(v for v in report["volumes"] if v["name"] == "aligned_tv")
    assert aligned_summary["finite_fraction"] == 0.0


def test_article_alignment_run_contracts_live_behind_bench_module() -> None:
    profile = diagnostic_profile()
    scenarios = article_scenario_catalog_for_kind("default")

    assert profile.name == "diagnostic_32"
    assert profile.levels == (4, 2, 1)
    assert scenarios
    assert all(scenario.phantom_key == "phantom94" for scenario in scenarios)
    assert {scenario.slug for scenario in scenarios}
    assert all(article_theta_span_deg(scenario) in {180.0, 360.0} for scenario in scenarios)
