from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tomojax.io import PreprocessConfig, load_dataset, preprocess_nxtomo, preprocess_tiff_stack

from ._helpers import write_angle_csv, write_raw_nxtomo, write_tiff_stack

pytestmark = pytest.mark.surface


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


def _remove_detector_metadata(raw_path: Path) -> None:
    with h5py.File(raw_path, "a") as handle:
        detector = handle["entry/instrument/detector"]
        if "detector_meta_json" in detector.attrs:
            del detector.attrs["detector_meta_json"]


def _replace_scalar_dataset(group: h5py.Group, name: str, value: float) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.asarray(value, dtype=np.float32))


def test_tiff_preprocess_defaults_to_absorption_with_provenance(tmp_path: Path) -> None:
    projections, flats, darks, angles = _preprocess_tiff_fixture(tmp_path)
    out_path = tmp_path / "corrected.nxs"

    result = preprocess_tiff_stack(
        projections,
        flats_path=flats,
        darks_path=darks,
        angles_path=angles,
        output_path=out_path,
    )

    assert result.output_domain == "absorption"
    loaded = load_dataset(out_path)
    np.testing.assert_allclose(loaded.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)
    np.testing.assert_allclose(loaded.angles, [0.0, 90.0])
    with h5py.File(out_path, "r") as handle:
        group = handle["entry/processing/tomojax/preprocess"]
        assert group.attrs["output_domain"] == "absorption"
        assert json.loads(group.attrs["frame_counts"])["sample"] == 2


def test_tiff_preprocess_can_write_transmission(tmp_path: Path) -> None:
    projections, flats, darks, angles = _preprocess_tiff_fixture(tmp_path)
    out_path = tmp_path / "transmission.nxs"
    config = PreprocessConfig(output_domain="transmission", output_dtype="float64")

    result = preprocess_tiff_stack(
        projections,
        flats_path=flats,
        darks_path=darks,
        angles_path=angles,
        output_path=out_path,
        config=config,
    )

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
    np.testing.assert_allclose(loaded.angles, [0.0, 90.0])
    assert loaded.detector is not None
    assert loaded.detector.nu == 2
    assert loaded.detector.nv == 2


def test_nxtomo_preprocess_defaults_detector_metadata_when_absent(tmp_path: Path) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "corrected.nxs"
    write_raw_nxtomo(raw)
    _remove_detector_metadata(raw)
    with h5py.File(raw, "a") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["x_pixel_size"]
        del detector["y_pixel_size"]

    preprocess_nxtomo(raw, corrected)

    loaded = load_dataset(corrected)
    assert loaded.detector is not None
    assert loaded.detector.nu == 2
    assert loaded.detector.nv == 2
    assert loaded.detector.du == 1.0
    assert loaded.detector.dv == 1.0
    assert loaded.detector.center == (0.0, 0.0)


def test_nxtomo_preprocess_uses_pixel_size_datasets_without_detector_json(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "corrected.nxs"
    write_raw_nxtomo(raw)
    _remove_detector_metadata(raw)
    with h5py.File(raw, "a") as handle:
        detector = handle["entry/instrument/detector"]
        _replace_scalar_dataset(detector, "x_pixel_size", 0.5)
        _replace_scalar_dataset(detector, "y_pixel_size", 0.75)

    preprocess_nxtomo(raw, corrected)

    loaded = load_dataset(corrected)
    assert loaded.detector is not None
    assert loaded.detector.du == 0.5
    assert loaded.detector.dv == 0.75
    assert loaded.detector.center == (0.0, 0.0)


def test_nxtomo_preprocess_prefers_detector_json_over_pixel_size_datasets(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "corrected.nxs"
    write_raw_nxtomo(raw)
    with h5py.File(raw, "a") as handle:
        detector = handle["entry/instrument/detector"]
        _replace_scalar_dataset(detector, "x_pixel_size", 0.5)
        _replace_scalar_dataset(detector, "y_pixel_size", 0.75)
        detector.attrs["detector_meta_json"] = json.dumps(
            {
                "nu": 2,
                "nv": 2,
                "du": 2.0,
                "dv": 3.0,
                "det_center": [4.0, -5.0],
            }
        )

    preprocess_nxtomo(raw, corrected)

    loaded = load_dataset(corrected)
    assert loaded.detector is not None
    assert loaded.detector.du == 2.0
    assert loaded.detector.dv == 3.0
    assert loaded.detector.center == (4.0, -5.0)


def test_nxtomo_preprocess_shifts_detector_center_after_crop(tmp_path: Path) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "cropped.nxs"
    write_raw_nxtomo(raw)
    with h5py.File(raw, "a") as handle:
        detector = handle["entry/instrument/detector"]
        detector.attrs["detector_meta_json"] = json.dumps(
            {
                "nu": 2,
                "nv": 2,
                "du": 2.0,
                "dv": 3.0,
                "det_center": [10.0, 20.0],
            }
        )
    config = PreprocessConfig(crop="0:1,0:1")

    preprocess_nxtomo(raw, corrected, config=config)

    loaded = load_dataset(corrected)
    assert loaded.detector is not None
    assert loaded.detector.nu == 1
    assert loaded.detector.nv == 1
    assert loaded.detector.center == (9.0, 18.5)


def test_nxtomo_preprocess_supports_view_selection_crop_and_transmission(tmp_path: Path) -> None:
    raw = tmp_path / "raw.nxs"
    corrected = tmp_path / "selected.nxs"
    write_raw_nxtomo(raw)
    config = PreprocessConfig(
        output_domain="transmission",
        select_views="1",
        crop="0:1,0:1",
    )

    result = preprocess_nxtomo(raw, corrected, config=config)

    assert result.output_shape == (1, 1, 1)
    loaded = load_dataset(corrected)
    np.testing.assert_allclose(loaded.projections[:, 0, 0], [0.8], rtol=1e-6)
    np.testing.assert_allclose(loaded.angles, [90.0])
    with h5py.File(corrected, "r") as handle:
        group = handle["entry/processing/tomojax/preprocess"]
        assert json.loads(group.attrs["crop_bounds"]) == {"x0": 0, "x1": 1, "y0": 0, "y1": 1}
        assert json.loads(group.attrs["final_projection_shape"]) == [1, 1, 1]


def _write_float_tiffs(directory: Path, images: np.ndarray) -> None:
    import imageio.v3 as iio

    directory.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        iio.imwrite(directory / f"p_{i:03d}.tif", image.astype(np.float32))


def test_tiff_preprocess_removes_detector_stripes_and_linearises_beam_hardening(
    tmp_path: Path,
) -> None:
    views, nv, nu = 40, 3, 64
    u = np.arange(nu)[None, None, :]
    centre = 32 + 12 * np.sin(np.linspace(0, 2 * np.pi, views))[:, None, None]
    truth = 2.0 * np.maximum(0.0, 1.0 - ((u - centre) / 18.0) ** 2) * np.ones((1, nv, 1))
    gain = np.ones(nu)
    gain[[10, 31, 45]] = [0.97, 1.03, 0.98]  # pixels the flats do not describe
    _write_float_tiffs(tmp_path / "proj", 1000.0 * np.exp(-truth) * gain)
    _write_float_tiffs(tmp_path / "flats", np.full((1, nv, nu), 1000.0))
    _write_float_tiffs(tmp_path / "darks", np.zeros((1, nv, nu)))
    write_angle_csv(tmp_path / "angles.csv", np.linspace(0.0, 360.0, views, endpoint=False))

    def corrected(name: str, **config: object) -> np.ndarray:
        preprocess_tiff_stack(
            tmp_path / "proj",
            flats_path=tmp_path / "flats",
            darks_path=tmp_path / "darks",
            angles_path=tmp_path / "angles.csv",
            output_path=tmp_path / f"{name}.nxs",
            config=PreprocessConfig(**config),  # type: ignore[arg-type]
        )
        return load_dataset(tmp_path / f"{name}.nxs").projections

    plain = corrected("plain")
    cleaned = corrected("cleaned", stripe_width=9)
    # Columns 10 and 31 stand out from their neighbours; column 45's offset is
    # smaller than the gradient across it and stays.
    error = np.abs(cleaned - truth).max(axis=(0, 1))
    assert error[[10, 31]].max() < 2e-3
    assert np.abs(plain - truth).max(axis=(0, 1))[[10, 31]].min() > 0.02
    assert np.delete(error, [10, 31, 45]).max() < 2e-3
    hardened = corrected("hardened", beam_hardening=(1.0, 0.05))
    np.testing.assert_allclose(hardened, plain + 0.05 * plain**2, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="absorption"):
        corrected("raw", output_domain="transmission", stripe_width=9)
