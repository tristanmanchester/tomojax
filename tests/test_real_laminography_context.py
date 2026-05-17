from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tomojax.bench.real_laminography_context import RealLaminoRunContext


def test_real_lamino_run_context_resolves_paths_and_writes_stage_products(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}

    def fake_write_stage_products(**kwargs):
        calls.update(kwargs)
        return {"orthos": str(kwargs["stage_dir"] / "orthos.png")}

    monkeypatch.setattr(
        "tomojax.bench.real_laminography_context.write_real_lamino_stage_products",
        fake_write_stage_products,
    )
    args = SimpleNamespace(
        out=str(tmp_path),
        preview_z=7,
        stack_z_range="5:9",
        snapshot_max_cols=4,
    )
    ctx = RealLaminoRunContext(args)
    ctx.naive_slice = np.ones((2, 2), dtype=np.float32)
    stage_dir = ctx.stage_dir("00_baseline")
    volume = np.zeros((2, 2, 2), dtype=np.float32)

    products = ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=volume,
        grid=object(),
        full_nz=12,
        input_reference=None,
        suffix="aligned",
    )

    assert ctx.run_root == tmp_path
    assert ctx.status_path == tmp_path / "status.json"
    assert ctx.preview_global_z == 7
    assert ctx.stack_z_range == (5, 9)
    assert ctx.stage_dir("00_baseline") == tmp_path / "00_baseline"
    assert products == {"orthos": str(stage_dir / "orthos.png")}
    assert calls["stage_dir"] == stage_dir
    assert calls["volume"] is volume
    assert calls["preview_global_z"] == 7
    assert calls["stack_z_range"] == (5, 9)
    assert calls["snapshot_max_cols"] == 4
    assert calls["input_reference"] is None
    assert calls["fallback_reference"] is ctx.naive_slice
    assert calls["suffix"] == "aligned"
