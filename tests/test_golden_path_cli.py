from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np

from tomojax.cli import main as main_cli
import tomojax.cli.align as align_cli
from tomojax.io import load_dataset


def test_golden_path_tiff_ingest_validate_inspect_recon_align(monkeypatch, tmp_path: Path) -> None:
    stack = tmp_path / "stack"
    stack.mkdir()
    iio.imwrite(stack / "0001.tif", np.ones((2, 4), dtype=np.float32))
    iio.imwrite(stack / "0002.tif", np.full((2, 4), 2.0, dtype=np.float32))
    angles = tmp_path / "angles.csv"
    angles.write_text("angle\n0\n90\n", encoding="utf-8")
    scan = tmp_path / "scan.nxs"
    recon = tmp_path / "recon.nxs"
    aligned = tmp_path / "aligned.nxs"
    quicklook = tmp_path / "recon.png"
    align_manifest = tmp_path / "align-manifest.json"
    align_calls: list[tuple[tuple[int, int, int], str, str]] = []

    def fake_align(
        geom,
        recon_grid,
        recon_detector,
        projections,
        *,
        cfg,
        resume_state=None,
        checkpoint_callback=None,
    ):
        del geom, recon_detector, resume_state, checkpoint_callback
        align_calls.append((tuple(projections.shape), cfg.schedule, cfg.align_profile))
        x = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        params5 = jnp.zeros((int(projections.shape[0]), 5), dtype=jnp.float32)
        return x, params5, {"loss": [0.0], "outer_stats": [], "active_dofs": ["det_u"]}

    def fake_align_multires(
        geom,
        recon_grid,
        recon_detector,
        projections,
        *,
        factors,
        cfg,
        resume_state=None,
        checkpoint_callback=None,
    ):
        assert factors == [1]
        return fake_align(
            geom,
            recon_grid,
            recon_detector,
            projections,
            cfg=cfg,
            resume_state=resume_state,
            checkpoint_callback=checkpoint_callback,
        )

    monkeypatch.setattr(align_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(align_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(align_cli, "_init_jax_compilation_cache", lambda: None)
    monkeypatch.setattr(align_cli, "align", fake_align)
    monkeypatch.setattr(align_cli, "align_multires", fake_align_multires)

    commands = [
        [
            "ingest",
            str(stack),
            "--angles",
            str(angles),
            "--out",
            str(scan),
            "--du",
            "1",
            "--dv",
            "1",
            "--grid",
            "4",
            "4",
            "2",
        ],
        ["validate", str(scan)],
        ["inspect", str(scan)],
        [
            "recon",
            str(scan),
            "--out",
            str(recon),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--grid",
            "4",
            "4",
            "2",
            "--views-per-batch",
            "1",
            "--quicklook",
            str(quicklook),
        ],
        ["validate", str(recon)],
        [
            "align",
            str(scan),
            "--out",
            str(aligned),
            "--mode",
            "cor",
            "--roi",
            "off",
            "--grid",
            "4",
            "4",
            "2",
            "--outer-iters",
            "1",
            "--recon-iters",
            "1",
            "--views-per-batch",
            "1",
            "--save-manifest",
            str(align_manifest),
        ],
        ["validate", str(aligned)],
    ]

    for command in commands:
        assert main_cli.main(command) == 0

    reconstructed = load_dataset(recon)
    assert reconstructed.volume is not None
    assert reconstructed.volume.shape == (4, 4, 2)
    assert quicklook.stat().st_size > 0

    aligned_dataset = load_dataset(aligned)
    assert aligned_dataset.volume is not None
    assert aligned_dataset.volume.shape == (4, 4, 2)
    assert aligned_dataset.align_params is not None
    assert aligned_dataset.align_params.shape == (2, 5)
    assert align_calls == [((2, 2, 4), "cor", "lightning")]
    assert align_manifest.stat().st_size > 0
