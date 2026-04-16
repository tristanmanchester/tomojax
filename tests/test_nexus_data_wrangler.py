from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import h5py
import numpy as np
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


def _axes_attr_values(value) -> list[str]:
    arr = np.asarray(value).reshape(-1)
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in arr.tolist()
    ]


def test_spatial_bin_and_padding_helpers() -> None:
    wrangler_mod = _load_module("nexus_data_wrangler_helpers_test", "scripts/nexus_data_wrangler.py")

    arr3d = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    arr2d = np.arange(15, dtype=np.float32).reshape(3, 5)

    binned3d = wrangler_mod._spatial_bin(arr3d, bin_y=2, bin_x=2)
    binned2d = wrangler_mod._spatial_bin(arr2d, bin_y=3, bin_x=2)
    padded = wrangler_mod._pad_to_multiples(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        mult_y=3,
        mult_x=4,
        mode="edge",
    )

    assert np.allclose(
        binned3d,
        np.array([[[2.5, 4.5], [10.5, 12.5]]], dtype=np.float32),
    )
    assert np.allclose(binned2d, np.array([[5.5, 7.5]], dtype=np.float32))
    assert padded.shape == (3, 4)
    assert np.allclose(padded[0], np.array([1.0, 2.0, 3.0, 3.0], dtype=np.float32))
    assert np.allclose(padded[-1], np.array([4.0, 5.0, 6.0, 6.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Unsupported array ndim"):
        wrangler_mod._spatial_bin(np.zeros((2, 2, 2, 2), dtype=np.float32), 2, 2)


def test_flat_dark_correct_handles_missing_darks_and_missing_flats(
    capsys,
) -> None:
    wrangler_mod = _load_module(
        "nexus_data_wrangler_contrast_test", "scripts/nexus_data_wrangler.py"
    )
    data = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0]],
            [[4.0, 8.0], [8.0, 16.0]],
        ],
        dtype=np.float32,
    )

    absorption = wrangler_mod.flat_dark_correct_to_absorption(
        data=data,
        image_key=np.array([0, 1], dtype=np.int32),
    )

    assert np.allclose(
        absorption,
        np.full((1, 2, 2), np.log(2.0), dtype=np.float32),
    )
    assert "No dark fields found" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="No flat fields found"):
        wrangler_mod.flat_dark_correct_to_absorption(
            data=data,
            image_key=np.array([0, 2], dtype=np.int32),
        )


def test_write_nexus_h5_writes_expected_structure(tmp_path: Path, capsys) -> None:
    wrangler_mod = _load_module("nexus_data_wrangler_writer_test", "scripts/nexus_data_wrangler.py")
    output_path = tmp_path / "wrangled.nxs"

    wrangler_mod.write_nexus_h5(
        output_path=str(output_path),
        projections=np.full((2, 3, 4), 0.5, dtype=np.float32),
        angles_deg=np.array([0.0, 90.0], dtype=np.float32),
        pixel_size_pixels_x=1.5,
        pixel_size_pixels_y=2.5,
        tilt_deg=42.0,
        tilt_about="z",
        grid=(4, 4, 3),
        voxels=(1.5, 1.5, 2.5),
        image_key=np.array([0, 0], dtype=np.int32),
        sample_name="specimen",
        source_name="Beamline",
        source_type="experiment",
        source_probe="x-ray",
    )

    captured = capsys.readouterr()
    assert f"Wrote corrected absorption data to: {output_path}" in captured.out

    with h5py.File(output_path, "r") as handle:
        entry = handle["entry"]
        grid_meta = json.loads(entry.attrs["grid_meta_json"])
        geometry_meta = json.loads(entry["geometry"].attrs["geometry_meta_json"])
        detector = entry["instrument/detector"]
        rotation = entry["sample/transformations/rotation_angle"]
        summary = json.loads(rotation.attrs["summary"])
        volume = entry["processing/tomojax/volume"]

        assert entry.attrs["definition"] == "NXtomo"
        assert grid_meta == {"nx": 4, "ny": 4, "nz": 3, "vx": 1.5, "vy": 1.5, "vz": 2.5}
        assert geometry_meta == {"tilt_deg": 42.0, "tilt_about": "z"}
        assert detector["data"].shape == (2, 3, 4)
        assert np.allclose(detector["image_key"][...], np.array([0, 0], dtype=np.int32))
        assert detector["x_pixel_size"][0] == np.float32(1.5)
        assert detector["y_pixel_size"][0] == np.float32(2.5)
        assert entry["sample/name"][()].decode("utf-8") == "specimen"
        assert np.allclose(rotation[...], np.array([0.0, 90.0], dtype=np.float32))
        assert summary == {"start_deg": 0.0, "count": 2, "endpoint": False, "step_deg": 90.0}
        assert entry["data/projections"].id == detector["data"].id
        assert volume.shape == (3, 4, 4)
        assert volume.chunks == (3, 4, 4)
        assert np.allclose(volume[...], np.zeros((3, 4, 4), dtype=np.float32))
        assert volume.attrs["long_name"] == "ground_truth_volume"
        assert _axes_attr_values(entry["processing/tomojax"].attrs["volume_axes_order"]) == [
            "zyx"
        ]


def test_main_derives_detector_grid_and_voxels_after_binning_and_padding(
    monkeypatch, capsys
) -> None:
    wrangler_mod = _load_module("nexus_data_wrangler_main_test", "scripts/nexus_data_wrangler.py")
    write_calls: list[dict[str, object]] = []

    def fake_load_raw(*_args):
        return (
            np.zeros((4, 4, 4), dtype=np.float32),
            np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32),
            np.array([0, 1, 2, 0], dtype=np.int32),
        )

    def fake_correct_to_absorption(*_args, **_kwargs):
        return np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)

    def fake_write_nexus_h5(**kwargs):
        write_calls.append(kwargs)

    monkeypatch.setattr(wrangler_mod, "load_raw", fake_load_raw)
    monkeypatch.setattr(
        wrangler_mod, "flat_dark_correct_to_absorption", fake_correct_to_absorption
    )
    monkeypatch.setattr(wrangler_mod, "write_nexus_h5", fake_write_nexus_h5)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nexus_data_wrangler",
            "--in",
            "raw.nxs",
            "--out",
            "wrangled.nxs",
            "--bin-y",
            "2",
            "--bin-x",
            "3",
            "--pixel-size",
            "1.5",
            "--pad-y-multiple",
            "3",
            "--pad-x-multiple",
            "4",
        ],
    )

    wrangler_mod.main()

    captured = capsys.readouterr()
    write_kwargs = write_calls[0]
    assert "Applied spatial binning: by=2, bx=3 -> new shape (2, 2, 2)" in captured.out
    assert "Padded to multiples (y=3, x=4): (2, 2, 2) -> (2, 3, 4)" in captured.out
    assert write_kwargs["grid"] == (4, 4, 3)
    assert write_kwargs["voxels"] == (4.5, 4.5, 3.0)
    assert write_kwargs["pixel_size_pixels_x"] == 4.5
    assert write_kwargs["pixel_size_pixels_y"] == 3.0
    assert np.allclose(write_kwargs["angles_deg"], np.array([0.0, 270.0], dtype=np.float32))
    assert np.allclose(
        write_kwargs["image_key"], np.zeros((2,), dtype=np.int32)
    )
    assert np.asarray(write_kwargs["projections"]).shape == (2, 3, 4)


def test_main_rejects_projection_count_mismatch(monkeypatch) -> None:
    wrangler_mod = _load_module(
        "nexus_data_wrangler_mismatch_test", "scripts/nexus_data_wrangler.py"
    )

    monkeypatch.setattr(
        wrangler_mod,
        "load_raw",
        lambda *_args: (
            np.zeros((3, 4, 4), dtype=np.float32),
            np.array([0.0, 90.0, 180.0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        wrangler_mod,
        "flat_dark_correct_to_absorption",
        lambda *_args, **_kwargs: np.zeros((1, 4, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["nexus_data_wrangler", "--in", "raw.nxs", "--out", "wrangled.nxs"],
    )

    with pytest.raises(RuntimeError, match="Projection count mismatch"):
        wrangler_mod.main()
