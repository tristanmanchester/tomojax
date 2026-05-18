from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tomojax.io import PreprocessConfig, load_dataset, preprocess_nxtomo, preprocess_tiff_stack

from ._helpers import write_angle_csv, write_raw_nxtomo, write_tiff_stack


def _preprocess_tiff_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    projections = tmp_path / "projections"
    flats = tmp_path / "flats"
    darks = tmp_path / "darks"
    write_tiff_stack(projections, [5.0, 9.0])
    write_tiff_stack(flats, [11.0])
    write_tiff_stack(darks, [1.0])
    angles = tmp_path / "angles.csv"
    write_angle_csv(angles, [0.0, 90.0])
    return projections, flats, darks, angles


def test_tiff_preprocess_defaults_to_absorption_with_provenance(tmp_path: Path) -> None:
    projections, flats, darks, angles = _preprocess_tiff_fixture(tmp_path)
    out_path = tmp_path / "corrected.nxs"

    result = preprocess_tiff_stack(projections, flats, darks, angles, out_path)

    assert result.output_domain == "absorption"
    loaded = load_dataset(out_path)
    np.testing.assert_allclose(loaded.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)
    np.testing.assert_allclose(loaded.angles_deg, [0.0, 90.0])
    with h5py.File(out_path, "r") as handle:
        group = handle["entry/processing/tomojax/preprocess"]
        assert group.attrs["output_domain"] == "absorption"
        assert json.loads(group.attrs["frame_counts"])["sample"] == 2


def test_tiff_preprocess_can_write_transmission(tmp_path: Path) -> None:
    projections, flats, darks, angles = _preprocess_tiff_fixture(tmp_path)
    out_path = tmp_path / "transmission.nxs"
    config = PreprocessConfig(output_domain="transmission", output_dtype="float64")

    result = preprocess_tiff_stack(projections, flats, darks, angles, out_path, config)

    assert result.output_domain == "transmission"
    loaded = load_dataset(out_path)
    assert loaded.projections.dtype == np.float64
    np.testing.assert_allclose(loaded.projections[:, 0, 0], [0.4, 0.8], rtol=1e-6)


def test_nxtomo_preprocess_splits_image_key_and_preserves_angles(tmp_path: Path) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "corrected.nxs"
    write_raw_nxtomo(raw)

    result = preprocess_nxtomo(raw, corrected)

    assert result.sample_count == 2
    assert result.flat_count == 1
    assert result.dark_count == 1
    assert result.output_shape == (2, 2, 2)
    loaded = load_dataset(corrected)
    np.testing.assert_allclose(loaded.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)
    np.testing.assert_allclose(loaded.angles_deg, [0.0, 90.0])
    assert loaded.detector is not None
    assert loaded.detector.nu == 2
    assert loaded.detector.nv == 2


def test_nxtomo_preprocess_supports_view_selection_crop_and_transmission(tmp_path: Path) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "selected.nxs"
    write_raw_nxtomo(raw)
    config = PreprocessConfig(
        output_domain="transmission",
        select_views="1",
        crop="0:1,0:1",
    )

    result = preprocess_nxtomo(raw, corrected, config)

    assert result.output_shape == (1, 1, 1)
    loaded = load_dataset(corrected)
    np.testing.assert_allclose(loaded.projections[:, 0, 0], [0.8], rtol=1e-6)
    np.testing.assert_allclose(loaded.angles_deg, [90.0])
    with h5py.File(corrected, "r") as handle:
        group = handle["entry/processing/tomojax/preprocess"]
        assert json.loads(group.attrs["crop_bounds"]) == {"x0": 0, "x1": 1, "y0": 0, "y1": 1}
        assert json.loads(group.attrs["final_projection_shape"]) == [1, 1, 1]
