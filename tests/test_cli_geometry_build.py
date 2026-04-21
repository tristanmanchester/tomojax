from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import jax.numpy as jnp
import pytest

from tomojax.data.geometry_meta import (
    build_geometry_from_meta,
    build_nominal_geometry_from_meta,
)
from tomojax.data.io_hdf5 import LoadedNXTomo, NXTomoMetadata, load_nxtomo, save_nxtomo
from tomojax.cli import recon as recon_cli
from tomojax.core.geometry import Grid, ParallelGeometry
from tomojax.utils.memory import ViewsPerBatchEstimate


def _parallel_meta(**updates):
    meta = {
        "geometry_type": "parallel",
        "thetas_deg": np.asarray([0.0, 45.0], dtype=np.float32),
        "detector": {
            "nu": 9,
            "nv": 7,
            "du": 1.25,
            "dv": 2.5,
            "det_center": [0.0, 0.0],
        },
        "grid": {
            "nx": 8,
            "ny": 10,
            "nz": 6,
            "vx": 0.8,
            "vy": 1.1,
            "vz": 1.4,
            "vol_origin": [1.0, 2.0, 3.0],
            "vol_center": [4.0, 5.0, 6.0],
        },
    }
    meta.update(updates)
    return meta


def _write_recon_input(path: Path, meta: dict[str, object]) -> None:
    projections = np.asarray(meta["projections"], dtype=np.float32)
    image_key = np.asarray(
        meta.get("image_key", np.zeros((projections.shape[0],), dtype=np.int32)),
        dtype=np.int32,
    )
    metadata = NXTomoMetadata(
        thetas_deg=np.asarray(meta["thetas_deg"], dtype=np.float32),
        image_key=image_key,
        grid=meta["grid"],
        detector=meta["detector"],
        geometry_type=str(meta["geometry_type"]),
    )
    save_nxtomo(path, projections=projections, metadata=metadata)


def test_build_geometry_from_meta_applies_saved_angle_offsets():
    meta = _parallel_meta(
        angle_offset_deg=np.asarray([2.0, -3.0], dtype=np.float32),
    )

    grid, detector, geom = build_geometry_from_meta(meta, apply_saved_alignment=True)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"] + meta["angle_offset_deg"],
    )

    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_geometry_from_meta_ignores_nonfinite_angle_offsets():
    meta = _parallel_meta(
        angle_offset_deg=np.asarray([np.nan, 5.0], dtype=np.float32),
    )

    grid, detector, geom = build_geometry_from_meta(meta, apply_saved_alignment=True)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    for i in range(2):
        pose = np.asarray(geom.pose_for_view(i), dtype=np.float32)
        assert np.isfinite(pose).all()
        np.testing.assert_allclose(
            pose,
            np.asarray(expected.pose_for_view(i), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


def test_build_geometry_from_meta_skips_double_applying_saved_angle_offsets():
    meta = _parallel_meta(
        angle_offset_deg=np.asarray([2.0, -3.0], dtype=np.float32),
        misalign_spec={"schedule": "already-baked"},
    )

    grid, detector, geom = build_geometry_from_meta(meta, apply_saved_alignment=True)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_geometry_from_meta_rejects_unsupported_geometry_types():
    meta = _parallel_meta(geometry_type="custom")

    with pytest.raises(ValueError, match="Unsupported geometry_type"):
        build_geometry_from_meta(meta)


def test_align_build_geometry_uses_grid_override_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, geom = build_nominal_geometry_from_meta(meta, grid_override=(11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert (grid.vx, grid.vy, grid.vz) == (1.25, 1.25, 2.5)
    expected = ParallelGeometry(grid=grid, detector=detector, thetas_deg=meta["thetas_deg"])
    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_align_build_geometry_accepts_positional_grid_override_for_compatibility():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, _, _ = build_nominal_geometry_from_meta(meta, (11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)


def test_recon_build_geometry_infers_grid_from_detector_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, _ = build_nominal_geometry_from_meta(meta)

    assert (grid.nx, grid.ny, grid.nz) == (detector.nu, detector.nu, detector.nv)
    assert (grid.vx, grid.vy, grid.vz) == (detector.du, detector.du, detector.dv)


def test_build_geometry_from_meta_uses_volume_shape_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, _ = build_geometry_from_meta(meta, volume_shape=(11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert (grid.vx, grid.vy, grid.vz) == (detector.du, detector.du, detector.dv)


def test_recon_build_geometry_accepts_positional_grid_override_for_compatibility():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, _, _ = build_nominal_geometry_from_meta(meta, (11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)


def test_recon_build_geometry_preserves_grid_origin_and_center():
    meta = _parallel_meta()

    grid, _, _ = build_nominal_geometry_from_meta(meta)

    assert grid.vol_origin == (1.0, 2.0, 3.0)
    assert grid.vol_center == (4.0, 5.0, 6.0)


def test_recon_build_geometry_preserves_grid_origin_and_center_with_override():
    meta = _parallel_meta()

    grid, _, _ = build_nominal_geometry_from_meta(meta, (11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert grid.vol_origin == (1.0, 2.0, 3.0)
    assert grid.vol_center == (4.0, 5.0, 6.0)


def test_recon_build_geometry_preserves_full_grid_override_metadata():
    meta = _parallel_meta()
    override_grid = Grid(nx=11, ny=13, nz=15, vx=0.8, vy=1.1, vz=1.4)

    grid, _, geom = build_nominal_geometry_from_meta(meta, override_grid)

    assert grid == override_grid
    assert grid.vol_origin is None
    assert grid.vol_center is None
    assert geom.grid == override_grid


def test_recon_cli_real_io_grid_override_preserves_grid_origin_and_center(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 7, 9), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
        geometry_meta=None,
    )
    in_path = tmp_path / "grid_origin_in.nxs"
    out_path = tmp_path / "grid_origin_out.nxs"
    _write_recon_input(in_path, meta)

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(
        recon_cli,
        "fbp",
        lambda geom, recon_grid, detector, proj, **kwargs: jnp.zeros(
            (recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--grid",
            "11",
            "13",
            "15",
            "--frame",
            "lab",
            "--volume-axes",
            "xyz",
            "--out",
            str(out_path),
        ],
    )

    recon_cli.main()

    out = load_nxtomo(out_path)
    assert out.grid is not None
    assert out.volume is not None
    assert out.frame == "lab"
    assert out.volume_axes_order == "xyz"
    assert out.disk_volume_axes_order == "xyz"
    assert out.volume_axes_source == "attr"
    assert out.volume.shape == (11, 13, 15)
    assert out.grid["nx"] == 11
    assert out.grid["ny"] == 13
    assert out.grid["nz"] == 15
    assert out.grid["vol_origin"] == [1.0, 2.0, 3.0]
    assert out.grid["vol_center"] == [4.0, 5.0, 6.0]


def test_recon_cli_real_io_grid_override_preserves_roi_centering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 16, 16), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
        geometry_meta=None,
        detector={
            "nu": 16,
            "nv": 16,
            "du": 1.0,
            "dv": 1.0,
            "det_center": [0.0, 0.0],
        },
        grid={
            "nx": 32,
            "ny": 32,
            "nz": 16,
            "vx": 1.0,
            "vy": 1.0,
            "vz": 1.0,
            "vol_origin": [11.0, 12.0, 13.0],
            "vol_center": [1.0, 2.0, 3.0],
        },
    )
    in_path = tmp_path / "roi_center_in.nxs"
    out_path = tmp_path / "roi_center_out.nxs"
    _write_recon_input(in_path, meta)

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(
        recon_cli,
        "fbp",
        lambda geom, recon_grid, detector, proj, **kwargs: jnp.zeros(
            (recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "auto",
            "--grid",
            "20",
            "20",
            "12",
            "--frame",
            "lab",
            "--volume-axes",
            "xyz",
            "--out",
            str(out_path),
        ],
    )

    recon_cli.main()

    out = load_nxtomo(out_path)
    assert out.grid is not None
    assert out.volume is not None
    assert out.frame == "lab"
    assert out.volume_axes_order == "xyz"
    assert out.disk_volume_axes_order == "xyz"
    assert out.volume_axes_source == "attr"
    assert out.volume.shape == (20, 20, 12)
    assert out.grid["nx"] == 20
    assert out.grid["ny"] == 20
    assert out.grid["nz"] == 12
    assert "vol_origin" not in out.grid
    assert "vol_center" not in out.grid


def test_recon_cli_writes_quicklook_png(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 3, 4), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
        detector={
            "nu": 4,
            "nv": 3,
            "du": 1.0,
            "dv": 1.0,
            "det_center": [0.0, 0.0],
        },
        grid={
            "nx": 4,
            "ny": 5,
            "nz": 3,
            "vx": 1.0,
            "vy": 1.0,
            "vz": 1.0,
        },
    )
    in_path = tmp_path / "quicklook_in.nxs"
    out_path = tmp_path / "quicklook_out.nxs"
    quicklook_path = tmp_path / "previews" / "quicklook.png"
    volume = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    _write_recon_input(in_path, meta)

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(
        recon_cli,
        "fbp",
        lambda geom, grid, detector, proj, **kwargs: jnp.asarray(volume),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--out",
            str(out_path),
            "--quicklook",
            str(quicklook_path),
        ],
    )

    recon_cli.main()

    out = load_nxtomo(out_path)
    assert out.volume is not None
    np.testing.assert_allclose(out.volume, volume)
    assert quicklook_path.exists()
    preview = iio.imread(quicklook_path)
    assert preview.dtype == np.uint8
    assert preview.shape == (5, 4)
    assert preview.max() > preview.min()


def test_recon_cli_auto_views_per_batch_uses_estimator_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((5, 7, 9), dtype=np.float32),
        thetas_deg=np.linspace(0.0, 180.0, 5, endpoint=False, dtype=np.float32),
        image_key=np.zeros((5,), dtype=np.int32),
    )
    in_path = tmp_path / "auto_vpb_in.nxs"
    out_path = tmp_path / "auto_vpb_out.nxs"
    _write_recon_input(in_path, meta)
    captured: dict[str, object] = {}

    def fake_estimate(**kwargs):
        captured["estimate_kwargs"] = kwargs
        return ViewsPerBatchEstimate(
            views_per_batch=3,
            free_bytes=123_456,
            fallback_used=False,
        )

    def fake_fbp(geom, recon_grid, detector, proj, **kwargs):
        captured["fbp_kwargs"] = kwargs
        return jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)

    caplog.set_level("INFO")
    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(recon_cli, "estimate_views_per_batch_info", fake_estimate)
    monkeypatch.setattr(recon_cli, "fbp", fake_fbp)
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--views-per-batch",
            "auto",
            "--out",
            str(out_path),
        ],
    )

    recon_cli.main()

    estimate_kwargs = captured["estimate_kwargs"]
    assert estimate_kwargs["n_views"] == 5
    assert estimate_kwargs["grid_nxyz"] == (8, 10, 6)
    assert estimate_kwargs["det_nuv"] == (7, 9)
    assert estimate_kwargs["algo"] == "fbp"
    assert estimate_kwargs["gather_dtype"] == "fp32"
    assert estimate_kwargs["fallback_batch"] == 1
    assert captured["fbp_kwargs"]["views_per_batch"] == 3
    assert "views_per_batch=3 (mode=auto, algo=fbp)" in caplog.text


def test_recon_cli_auto_views_per_batch_warns_on_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 7, 9), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
    )
    in_path = tmp_path / "auto_vpb_fallback_in.nxs"
    out_path = tmp_path / "auto_vpb_fallback_out.nxs"
    _write_recon_input(in_path, meta)
    captured: dict[str, object] = {}

    def fake_estimate(**kwargs):
        return ViewsPerBatchEstimate(
            views_per_batch=1,
            free_bytes=None,
            fallback_used=True,
            fallback_reason="available memory could not be determined",
        )

    def fake_fbp(geom, recon_grid, detector, proj, **kwargs):
        captured["views_per_batch"] = kwargs["views_per_batch"]
        return jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)

    caplog.set_level("WARNING")
    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(recon_cli, "estimate_views_per_batch_info", fake_estimate)
    monkeypatch.setattr(recon_cli, "fbp", fake_fbp)
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--views-per-batch",
            "auto",
            "--out",
            str(out_path),
        ],
    )

    recon_cli.main()

    assert captured["views_per_batch"] == 1
    assert "Could not determine available memory" in caplog.text


def test_recon_cli_explicit_views_per_batch_skips_estimator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 7, 9), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
    )
    in_path = tmp_path / "explicit_vpb_in.nxs"
    out_path = tmp_path / "explicit_vpb_out.nxs"
    _write_recon_input(in_path, meta)
    captured: dict[str, object] = {}

    def fail_estimate(**kwargs):
        raise AssertionError("explicit views_per_batch should not call estimator")

    def fake_fbp(geom, recon_grid, detector, proj, **kwargs):
        captured["views_per_batch"] = kwargs["views_per_batch"]
        return jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(recon_cli, "estimate_views_per_batch_info", fail_estimate)
    monkeypatch.setattr(recon_cli, "fbp", fake_fbp)
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--views-per-batch",
            "4",
            "--out",
            str(out_path),
        ],
    )

    recon_cli.main()

    assert captured["views_per_batch"] == 4


def test_recon_cli_default_views_per_batch_keeps_fbp_conservative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 7, 9), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
    )
    in_path = tmp_path / "default_vpb_in.nxs"
    out_path = tmp_path / "default_vpb_out.nxs"
    _write_recon_input(in_path, meta)
    captured: dict[str, object] = {}

    def fake_fbp(geom, recon_grid, detector, proj, **kwargs):
        captured["views_per_batch"] = kwargs["views_per_batch"]
        return jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(recon_cli, "fbp", fake_fbp)
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(in_path),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--gather-dtype",
            "fp32",
            "--out",
            str(out_path),
        ],
    )

    recon_cli.main()

    assert captured["views_per_batch"] == 1


def test_recon_cli_spdhg_default_and_explicit_views_per_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    meta = _parallel_meta(
        projections=np.zeros((2, 7, 9), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
    )
    captured: list[int] = []

    def fake_spdhg(geom, recon_grid, detector, proj, *, init_x, config):
        captured.append(config.views_per_batch)
        vol = jnp.zeros((recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32)
        return vol, {}

    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(recon_cli, "spdhg_tv", fake_spdhg)

    for suffix, extra_args in (("default", []), ("explicit", ["--views-per-batch", "5"])):
        in_path = tmp_path / f"spdhg_{suffix}_vpb_in.nxs"
        out_path = tmp_path / f"spdhg_{suffix}_vpb_out.nxs"
        _write_recon_input(in_path, meta)
        monkeypatch.setattr(
            "sys.argv",
            [
                "recon",
                "--data",
                str(in_path),
                "--algo",
                "spdhg",
                "--roi",
                "off",
                "--gather-dtype",
                "fp32",
                "--out",
                str(out_path),
                *extra_args,
            ],
        )
        recon_cli.main()

    assert captured == [16, 5]


def test_recon_build_geometry_keeps_nominal_geometry_for_saved_alignment_metadata():
    align_params = np.asarray(
        [[0.0, 0.0, 0.0, 1.25, -0.5], [0.1, -0.2, 0.3, 0.0, 0.25]],
        dtype=np.float32,
    )
    angle_offset = np.asarray([5.0, -3.0], dtype=np.float32)
    meta = _parallel_meta(align_params=align_params, angle_offset_deg=angle_offset)

    grid, detector, geom = build_nominal_geometry_from_meta(meta)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(geom.pose_for_view(i), dtype=np.float32),
            np.asarray(expected.pose_for_view(i), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


def test_recon_build_geometry_skips_double_applying_baked_angle_offsets():
    angle_offset = np.asarray([7.0, -4.0], dtype=np.float32)
    meta = _parallel_meta(
        angle_offset_deg=angle_offset,
        misalign_spec={"kind": "scheduled"},
    )

    grid, detector, geom = build_nominal_geometry_from_meta(meta)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(0), dtype=np.float32),
        np.asarray(expected.pose_for_view(0), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
