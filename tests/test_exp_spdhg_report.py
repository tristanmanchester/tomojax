from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_main_writes_markdown_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    report_mod = _load_module("exp_spdhg_report_test", "scripts/exp_spdhg_report.py")
    metrics = {
        "dataset": {"nx": 8, "ny": 9, "nz": 10, "n_views": 11, "phantom": "cube"},
        "fbp": {"psnr": 1.0, "ssim_center": 0.2, "mse": 3.0, "tv": 4.0},
        "fista": {"psnr": 5.0, "ssim_center": 0.6, "mse": 7.0, "tv": 8.0},
        "spdhg": {"psnr": 9.0, "ssim_center": 1.0, "mse": 11.0, "tv": 12.0},
        "timing_sec": {"fbp": 0.1, "fista": 0.2, "spdhg": 0.3},
    }
    indir = tmp_path / "exp_spdhg_256"
    indir.mkdir()
    (indir / "metrics.json").write_text(json.dumps(metrics))

    monkeypatch.setattr(sys, "argv", ["exp_spdhg_report", "--indir", str(indir)])

    report_mod.main()

    out_path = indir / "REPORT.md"
    text = out_path.read_text()
    captured = capsys.readouterr()
    assert "# CT Benchmark Report" in text
    assert "Dataset: 8x9x10, views=11, phantom=cube" in text
    assert "- FISTA: PSNR=5.0, SSIM_center=0.6, MSE=7.0, TV=8.0" in text
    assert "- SPDHG: 0.3" in text
    assert f"Wrote {out_path}" in captured.out


def test_main_raises_when_metrics_file_is_missing(monkeypatch, tmp_path: Path) -> None:
    report_mod = _load_module(
        "exp_spdhg_report_missing_metrics_test", "scripts/exp_spdhg_report.py"
    )
    indir = tmp_path / "exp_spdhg_missing"
    indir.mkdir()

    monkeypatch.setattr(sys, "argv", ["exp_spdhg_report", "--indir", str(indir)])

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        report_mod.main()
