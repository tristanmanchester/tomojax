from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import warnings

import h5py
import numpy as np
import pytest

from tomojax.bench.spdhg_benchmark import (
    SpdhgGeometryBundle,
    SpdhgReconstructionResults,
    compute_spdhg_benchmark_metrics,
    is_expected_spdhg_fallback_failure,
)
from tomojax.io import LoadedNXTomo, NXTomoMetadata, load_nxtomo

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _loaded_dataset(volume: np.ndarray) -> LoadedNXTomo:
    volume = np.asarray(volume, dtype=np.float32)
    projections = np.ones((2, volume.shape[0], volume.shape[1]), dtype=np.float32)
    return LoadedNXTomo(
        projections=projections,
        metadata=NXTomoMetadata(
            thetas_deg=np.array([0.0, 90.0], dtype=np.float32),
            grid={
                "nx": int(volume.shape[0]),
                "ny": int(volume.shape[1]),
                "nz": int(volume.shape[2]),
                "vx": 1.0,
                "vy": 1.0,
                "vz": 1.0,
            },
            detector={
                "nu": int(volume.shape[0]),
                "nv": int(volume.shape[1]),
                "du": 1.0,
                "dv": 1.0,
                "det_center": (0.0, 0.0),
            },
            geometry_type="parallel",
            volume=volume,
            frame="sample",
        ),
    )


def test_helpers_compute_metrics_and_write_outputs(tmp_path: Path) -> None:
    bench_mod = _load_module("exp_spdhg_bench_helpers_test", "scripts/exp_spdhg_bench.py")
    volume = np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    mildly_shifted = np.clip(volume * 0.95, 0.0, 1.0)
    shifted = np.clip(volume * 0.75, 0.0, 1.0)

    out_dir = tmp_path / "nested" / "outputs"
    bench_mod.ensure_dir(str(out_dir))

    mildly_shifted_psnr = bench_mod.psnr3d(mildly_shifted, volume)
    shifted_psnr = bench_mod.psnr3d(shifted, volume)
    ssim = bench_mod.ssim_center_slices(volume, volume, n_slices=3)
    tv = bench_mod.total_variation(shifted)

    png_path = out_dir / "slices.png"
    nxs_path = out_dir / "volume.nxs"
    bench_mod.save_slice_png(str(png_path), shifted, title="Shifted slices")
    bench_mod.save_volume(str(nxs_path), _loaded_dataset(volume), shifted, frame="recon")

    saved = load_nxtomo(str(nxs_path))

    assert out_dir.exists()
    assert mildly_shifted_psnr > shifted_psnr
    assert np.isclose(ssim, 1.0)
    assert tv > 0.0
    assert png_path.exists()
    assert png_path.stat().st_size > 0
    assert np.allclose(saved.volume, shifted)
    assert saved.frame == "recon"
    assert np.allclose(saved.projections, np.ones((2, 8, 8), dtype=np.float32))


def test_spdhg_benchmark_contracts_live_behind_bench_module() -> None:
    volume = np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    args = _load_module("exp_spdhg_bench_contract_args_test", "scripts/exp_spdhg_bench.py").parse_args(
        [
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--nu",
            "8",
            "--nv",
            "8",
            "--n-views",
            "2",
        ]
    )
    dataset = _loaded_dataset(volume)
    bundle = SpdhgGeometryBundle(
        data=dataset,
        projections=dataset.projections,
        grid=dataset.metadata.grid,
        detector=dataset.metadata.detector,
        geometry=None,  # type: ignore[arg-type]
        ground_truth=volume,
    )
    results = SpdhgReconstructionResults(
        volumes={"fbp": volume * 0.99, "fista": volume * 0.9, "spdhg": volume * 0.8},
        timing_sec={"fbp": 1.0, "fista": 2.0, "spdhg": 3.0},
        fista_info={"solver": "fista"},
        spdhg_info={"solver": "spdhg"},
    )

    metrics = compute_spdhg_benchmark_metrics(args, bundle, results)

    assert is_expected_spdhg_fallback_failure(MemoryError("oom")) is True
    assert is_expected_spdhg_fallback_failure(RuntimeError("RESOURCE_EXHAUSTED")) is True
    assert is_expected_spdhg_fallback_failure(ValueError("bad geometry")) is False
    assert metrics["dataset"]["n_views"] == 2
    assert metrics["fbp"]["mse"] > 0.0
    assert metrics["fista_info"] == {"solver": "fista"}
    assert metrics["spdhg_info"] == {"solver": "spdhg"}


def test_prepare_or_load_dataset_falls_back_only_for_expected_resource_errors(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    bench_mod = _load_module("exp_spdhg_bench_fallback_expected_test", "scripts/exp_spdhg_bench.py")
    args = bench_mod.parse_args(
        [
            "--outdir",
            str(tmp_path),
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--nu",
            "8",
            "--nv",
            "8",
            "--n-views",
            "2",
            "--overwrite-data",
        ]
    )
    fallback_calls = []

    def fake_generate_dataset(_plan):
        raise RuntimeError("RESOURCE_EXHAUSTED: simulated GPU allocation failure")

    def fake_run_simulate_fallback(plan):
        fallback_calls.append(plan.sim_path)

    monkeypatch.setattr(bench_mod, "generate_dataset", fake_generate_dataset)
    monkeypatch.setattr(bench_mod, "run_simulate_fallback", fake_run_simulate_fallback)

    sim_path = bench_mod.prepare_or_load_dataset(args)
    captured = capsys.readouterr()

    assert sim_path == str(tmp_path / "dataset.nxs")
    assert fallback_calls == [str(tmp_path / "dataset.nxs")]
    assert "falling back to CLI simulate" in captured.out


def test_prepare_or_load_dataset_reraises_unexpected_errors(monkeypatch, tmp_path: Path) -> None:
    bench_mod = _load_module(
        "exp_spdhg_bench_fallback_unexpected_test", "scripts/exp_spdhg_bench.py"
    )
    args = bench_mod.parse_args(
        [
            "--outdir",
            str(tmp_path),
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--nu",
            "8",
            "--nv",
            "8",
            "--n-views",
            "2",
            "--overwrite-data",
        ]
    )
    fallback_calls = []

    def fake_generate_dataset(_plan):
        raise ValueError("projection shape mismatch")

    def fake_run_simulate_fallback(_plan):
        fallback_calls.append(True)

    monkeypatch.setattr(bench_mod, "generate_dataset", fake_generate_dataset)
    monkeypatch.setattr(bench_mod, "run_simulate_fallback", fake_run_simulate_fallback)

    with pytest.raises(ValueError, match="projection shape mismatch"):
        bench_mod.prepare_or_load_dataset(args)

    assert fallback_calls == []


def test_run_reconstructions_reraises_unexpected_fbp_errors(monkeypatch, tmp_path: Path) -> None:
    bench_mod = _load_module("exp_spdhg_bench_fbp_unexpected_test", "scripts/exp_spdhg_bench.py")
    gt_volume = np.ones((8, 8, 8), dtype=np.float32)
    dataset = _loaded_dataset(gt_volume)
    grid = bench_mod.Grid(8, 8, 8, 1.0, 1.0, 1.0)
    detector = bench_mod.Detector(8, 8, 1.0, 1.0, det_center=(0.0, 0.0))
    geometry = bench_mod.ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.array([0.0, 90.0], dtype=np.float32),
    )
    bundle = bench_mod.GeometryBundle(
        data=dataset,
        projections=bench_mod.jnp.asarray(dataset.projections, dtype=bench_mod.jnp.float32),
        grid=grid,
        detector=detector,
        geometry=geometry,
        ground_truth=gt_volume,
    )
    args = bench_mod.parse_args(
        [
            "--outdir",
            str(tmp_path),
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--nu",
            "8",
            "--nv",
            "8",
            "--n-views",
            "2",
        ]
    )
    run_calls = []

    def fake_fbp(*_args, **_kwargs):
        raise ValueError("bad detector geometry")

    def fake_run_command(*_args, **_kwargs):
        run_calls.append(True)

    monkeypatch.setattr(bench_mod, "fbp", fake_fbp)
    monkeypatch.setattr(bench_mod, "run_command", fake_run_command)

    with pytest.raises(ValueError, match="bad detector geometry"):
        bench_mod.run_reconstructions(
            args,
            bundle,
            gather="fp32",
            vol_mask_np=None,
            vol_mask=None,
        )

    assert run_calls == []


def test_run_reconstructions_falls_back_for_expected_fbp_resource_errors(
    monkeypatch, tmp_path: Path
) -> None:
    bench_mod = _load_module("exp_spdhg_bench_fbp_expected_test", "scripts/exp_spdhg_bench.py")
    gt_volume = np.ones((8, 8, 8), dtype=np.float32)
    dataset = _loaded_dataset(gt_volume)
    grid = bench_mod.Grid(8, 8, 8, 1.0, 1.0, 1.0)
    detector = bench_mod.Detector(8, 8, 1.0, 1.0, det_center=(0.0, 0.0))
    geometry = bench_mod.ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.array([0.0, 90.0], dtype=np.float32),
    )
    bundle = bench_mod.GeometryBundle(
        data=dataset,
        projections=bench_mod.jnp.asarray(dataset.projections, dtype=bench_mod.jnp.float32),
        grid=grid,
        detector=detector,
        geometry=geometry,
        ground_truth=gt_volume,
    )
    args = bench_mod.parse_args(
        [
            "--outdir",
            str(tmp_path),
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--nu",
            "8",
            "--nv",
            "8",
            "--n-views",
            "2",
        ]
    )
    run_calls = []

    def fake_fbp(*_args, **_kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: simulated GPU allocation failure")

    def fake_run_command(cmd: list[str], *, check: bool, env: dict[str, str]):
        run_calls.append((cmd, check, env))
        out_path = Path(cmd[cmd.index("--out") + 1])
        with h5py.File(out_path, "w") as handle:
            handle.create_dataset(
                "/entry/processing/tomojax/volume",
                data=np.full(gt_volume.shape, 2.0, dtype=np.float32),
            )

    monkeypatch.setattr(bench_mod, "fbp", fake_fbp)
    monkeypatch.setattr(bench_mod, "run_command", fake_run_command)
    monkeypatch.setattr(
        bench_mod,
        "fista_tv",
        lambda *_args, **_kwargs: (np.full(gt_volume.shape, 3.0, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        bench_mod,
        "spdhg_tv",
        lambda *_args, **_kwargs: (np.full(gt_volume.shape, 4.0, dtype=np.float32), {}),
    )
    monkeypatch.setattr(bench_mod, "save_volume", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bench_mod, "save_slice_png", lambda *_args, **_kwargs: None)

    results = bench_mod.run_reconstructions(
        args,
        bundle,
        gather="fp32",
        vol_mask_np=None,
        vol_mask=None,
    )

    assert len(run_calls) == 1
    assert run_calls[0][2]["JAX_PLATFORM_NAME"] == "cpu"
    assert np.allclose(results.volumes["fbp"], np.full(gt_volume.shape, 2.0, dtype=np.float32))


def test_main_reuses_dataset_and_writes_benchmark_artifacts(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    bench_mod = _load_module("exp_spdhg_bench_main_test", "scripts/exp_spdhg_bench.py")
    gt_volume = np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    dataset = _loaded_dataset(gt_volume)
    dataset_path = tmp_path / "dataset.nxs"
    dataset_path.touch()
    run_calls: list[tuple[list[str], dict[str, str]]] = []
    spdhg_configs = []

    def fake_load_nxtomo(path: str) -> LoadedNXTomo:
        assert path == str(dataset_path)
        return dataset

    def fake_run_command(
        cmd: list[str], *, check: bool, env: dict[str, str]
    ):  # pragma: no cover - exercised via assertions
        run_calls.append((list(cmd), dict(env)))
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as handle:
            handle.create_dataset(
                "/entry/processing/tomojax/volume",
                data=np.full(gt_volume.shape, 2.0, dtype=np.float32),
            )

    def fake_fista_tv(*_args, **_kwargs):
        return np.full(gt_volume.shape, 3.0, dtype=np.float32), {"solver": "fista"}

    def fake_spdhg_tv(*_args, config, **_kwargs):
        spdhg_configs.append(config)
        return np.full(gt_volume.shape, 4.0, dtype=np.float32), {"solver": "spdhg"}

    monkeypatch.setattr(bench_mod, "load_nxtomo", fake_load_nxtomo)
    monkeypatch.setattr(bench_mod, "run_command", fake_run_command)
    monkeypatch.setattr(bench_mod, "fista_tv", fake_fista_tv)
    monkeypatch.setattr(bench_mod, "spdhg_tv", fake_spdhg_tv)
    monkeypatch.setattr(
        bench_mod,
        "cylindrical_mask_xy",
        lambda grid, det: np.ones((grid.nx, grid.ny), dtype=np.float32),
    )
    monkeypatch.setattr(
        bench_mod.time,
        "perf_counter",
        iter([1.0, 2.0, 3.0, 5.0, 8.0, 13.0]).__next__,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exp_spdhg_bench",
            "--outdir",
            str(tmp_path),
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--nu",
            "8",
            "--nv",
            "8",
            "--n-views",
            "2",
            "--fbp-on-cpu",
            "--spdhg-manual-steps",
        ],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        bench_mod.main()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    report_text = (tmp_path / "REPORT.txt").read_text()
    captured = capsys.readouterr()
    saved_fbp = load_nxtomo(str(tmp_path / "fbp.nxs"))
    saved_fista = load_nxtomo(str(tmp_path / "fista.nxs"))
    saved_spdhg = load_nxtomo(str(tmp_path / "spdhg.nxs"))

    assert "[simulate] reusing dataset" in captured.out
    assert "[done] results written" in captured.out
    assert len(run_calls) == 1
    assert run_calls[0][0][:3] == [sys.executable, "-m", "tomojax.cli.recon"]
    assert run_calls[0][1]["JAX_PLATFORM_NAME"] == "cpu"
    assert metrics["timing_sec"] == {"fbp": 1.0, "fista": 2.0, "spdhg": 5.0}
    assert metrics["fista_info"] == {"solver": "fista"}
    assert metrics["spdhg_info"] == {"solver": "spdhg"}
    assert metrics["fbp"]["mse"] is not None
    assert spdhg_configs[0].tau == 0.02
    assert spdhg_configs[0].sigma_data == 0.25
    assert spdhg_configs[0].sigma_tv == 0.25
    assert np.allclose(saved_fbp.volume, np.full(gt_volume.shape, 2.0, dtype=np.float32))
    assert np.allclose(saved_fista.volume, np.full(gt_volume.shape, 3.0, dtype=np.float32))
    assert np.allclose(saved_spdhg.volume, np.full(gt_volume.shape, 4.0, dtype=np.float32))
    assert (tmp_path / "fbp_slices.png").exists()
    assert (tmp_path / "fista_slices.png").exists()
    assert (tmp_path / "spdhg_slices.png").exists()
    assert (tmp_path / "diff_center_z.png").exists()
    assert "CT Reconstruction Benchmark" in report_text
