from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from tomojax.cli import preprocess as preprocess_cli
from tomojax.data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo
from tomojax.data.preprocess import PreprocessConfig, preprocess_nxtomo


def _attr_to_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _attr_to_str(value.reshape(-1)[0])
    return str(value)


def _raw_path(tmp_path, *, frames: np.ndarray, image_key: np.ndarray) -> str:
    path = tmp_path / "raw.nxs"
    n_frames = int(frames.shape[0])
    metadata = NXTomoMetadata(
        thetas_deg=np.linspace(0.0, 30.0 * (n_frames - 1), n_frames, dtype=np.float32),
        image_key=np.asarray(image_key, dtype=np.int32),
        grid={"nx": 4, "ny": 4, "nz": 2, "vx": 1.0, "vy": 1.0, "vz": 1.0},
        detector={
            "nu": int(frames.shape[2]),
            "nv": int(frames.shape[1]),
            "du": 1.25,
            "dv": 2.5,
            "det_center": [0.5, -0.5],
        },
        geometry_type="lamino",
        geometry_meta={"tilt_deg": 35.0, "tilt_about": "x"},
        sample_name="specimen",
        source_name="Beamline",
        source_type="experiment",
        source_probe="x-ray",
    )
    save_nxtomo(str(path), frames, metadata=metadata)
    return str(path)


def test_preprocess_known_values_writes_transmission_and_provenance(tmp_path):
    frames = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],  # dark
            [[6.0, 10.0], [3.0, 11.0]],  # sample 1
            [[11.0, 11.0], [11.0, 11.0]],  # flat
            [[9.0, 5.0], [11.0, 2.0]],  # sample 2
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 1, 0]))
    out_path = tmp_path / "corrected.nxs"

    result = preprocess_nxtomo(raw_path, out_path)

    assert result.output_domain == "transmission"
    assert result.output_shape == (2, 2, 2)
    corrected = load_nxtomo(str(out_path))
    expected = np.array(
        [
            [[0.5, 0.9], [0.2, 1.0]],
            [[0.8, 0.4], [1.0, 0.1]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(corrected.projections, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(corrected.image_key, np.zeros((2,), dtype=np.int32))
    np.testing.assert_allclose(corrected.thetas_deg, np.array([30.0, 90.0], dtype=np.float32))
    assert corrected.detector == {
        "nu": 2,
        "nv": 2,
        "du": 1.25,
        "dv": 2.5,
        "det_center": [0.5, -0.5],
    }
    assert corrected.geometry_type == "lamino"
    assert corrected.geometry_meta == {"tilt_deg": 35.0, "tilt_about": "x"}
    assert corrected.sample_name == "specimen"
    assert corrected.source == {"name": "Beamline", "type": "experiment", "probe": "x-ray"}

    with h5py.File(out_path, "r") as handle:
        preprocess = handle["/entry/processing/tomojax/preprocess"]
        assert preprocess.attrs["NX_class"] == "NXcollection"
        assert int(preprocess.attrs["schema_version"]) == 1
        assert _attr_to_str(preprocess.attrs["output_domain"]) == "transmission"
        assert json.loads(preprocess.attrs["frame_counts"]) == {
            "dark": 1,
            "flat": 1,
            "sample": 2,
        }
        np.testing.assert_allclose(preprocess["flat_mean"][...], np.full((2, 2), 11.0))
        np.testing.assert_allclose(preprocess["dark_mean"][...], np.ones((2, 2)))


def test_preprocess_log_outputs_absorption(tmp_path):
    frames = np.array(
        [
            np.full((2, 2), 2.0, dtype=np.float32),
            np.full((2, 2), 6.0, dtype=np.float32),
            np.full((2, 2), 10.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 1]))
    out_path = tmp_path / "absorption.nxs"

    preprocess_nxtomo(
        raw_path,
        out_path,
        PreprocessConfig(log=True, epsilon=1e-6, clip_min=1e-6),
    )

    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(
        corrected.projections,
        np.full((1, 2, 2), -np.log(0.5), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    with h5py.File(out_path, "r") as handle:
        assert _attr_to_str(
            handle["/entry/processing/tomojax/preprocess"].attrs["output_domain"]
        ) == "absorption"


def test_preprocess_missing_flats_requires_explicit_override(tmp_path):
    frames = np.array(
        [
            np.full((1, 2), 2.0, dtype=np.float32),
            np.full((1, 2), 6.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0]))

    with pytest.raises(ValueError, match="No flat fields found"):
        preprocess_nxtomo(raw_path, tmp_path / "bad.nxs")

    preprocess_nxtomo(
        raw_path,
        tmp_path / "override_flat.nxs",
        PreprocessConfig(assume_flat_field=10.0),
    )
    corrected = load_nxtomo(str(tmp_path / "override_flat.nxs"))
    np.testing.assert_allclose(corrected.projections, np.full((1, 1, 2), 0.5))


def test_preprocess_missing_darks_requires_explicit_override(tmp_path):
    frames = np.array(
        [
            np.full((1, 2), 4.0, dtype=np.float32),
            np.full((1, 2), 8.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([0, 1]))

    with pytest.raises(ValueError, match="No dark fields found"):
        preprocess_nxtomo(raw_path, tmp_path / "bad.nxs")

    preprocess_nxtomo(
        raw_path,
        tmp_path / "override_dark.nxs",
        PreprocessConfig(assume_dark_field=0.0),
    )
    corrected = load_nxtomo(str(tmp_path / "override_dark.nxs"))
    np.testing.assert_allclose(corrected.projections, np.full((1, 1, 2), 0.5))


def test_preprocess_warns_and_repairs_nonpositive_values(tmp_path, caplog):
    frames = np.array(
        [
            np.full((1, 2), 5.0, dtype=np.float32),  # dark
            np.array([[3.0, 5.0]], dtype=np.float32),  # sample <= dark
            np.full((1, 2), 4.0, dtype=np.float32),  # flat < dark
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 1]))

    caplog.set_level("WARNING")
    result = preprocess_nxtomo(
        raw_path,
        tmp_path / "safe.nxs",
        PreprocessConfig(log=True, epsilon=1e-3, clip_min=1e-3),
    )

    assert result.warning_counts["nonpositive_flat_denominator"] == 2
    assert result.warning_counts["nonpositive_transmission"] == 2
    assert any("Flat-dark denominator" in record.message for record in caplog.records)
    assert any("Transmission values" in record.message for record in caplog.records)
    corrected = load_nxtomo(str(tmp_path / "safe.nxs"))
    assert np.isfinite(corrected.projections).all()
    np.testing.assert_allclose(corrected.projections, np.full((1, 1, 2), -np.log(1e-3)))


def test_preprocess_output_dtype_float64(tmp_path):
    frames = np.array(
        [
            np.full((1, 2), 1.0, dtype=np.float32),
            np.full((1, 2), 5.0, dtype=np.float32),
            np.full((1, 2), 9.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 1]))
    out_path = tmp_path / "float64.nxs"

    preprocess_nxtomo(raw_path, out_path, PreprocessConfig(output_dtype="float64"))

    with h5py.File(out_path, "r") as handle:
        assert handle["/entry/instrument/detector/data"].dtype == np.dtype("float64")
        assert handle["/entry/processing/tomojax/preprocess/flat_mean"].dtype == np.dtype(
            "float64"
        )


def test_preprocess_finds_unique_nonstandard_image_key(tmp_path):
    raw_path = tmp_path / "raw_equivalent_key.nxs"
    with h5py.File(raw_path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset(
            "data",
            data=np.array(
                [
                    np.full((1, 2), 1.0, dtype=np.float32),
                    np.full((1, 2), 5.0, dtype=np.float32),
                    np.full((1, 2), 9.0, dtype=np.float32),
                ]
            ),
        )
        sample = entry.create_group("sample")
        rotation = sample.create_group("transformations").create_dataset(
            "rotation_angle",
            data=np.array([0.0, 10.0, 20.0], dtype=np.float32),
        )
        rotation.attrs["units"] = "degree"
        entry.create_group("acquisition").create_dataset(
            "image_key",
            data=np.array([2, 0, 1], dtype=np.int32),
        )

    out_path = tmp_path / "corrected_equivalent_key.nxs"
    preprocess_nxtomo(str(raw_path), out_path)

    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(corrected.projections, np.full((1, 1, 2), 0.5))
    with h5py.File(out_path, "r") as handle:
        preprocess = handle["/entry/processing/tomojax/preprocess"]
        assert _attr_to_str(preprocess.attrs["image_key_path"]) == "/entry/acquisition/image_key"


def test_preprocess_cli_smoke(tmp_path, capsys):
    frames = np.array(
        [
            np.full((1, 2), 1.0, dtype=np.float32),
            np.full((1, 2), 5.0, dtype=np.float32),
            np.full((1, 2), 9.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 1]))
    out_path = tmp_path / "cli.nxs"

    status = preprocess_cli.main([raw_path, str(out_path), "--epsilon", "1e-6"])

    assert status == 0
    assert "Wrote corrected transmission projections" in capsys.readouterr().out
    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(corrected.projections, np.full((1, 1, 2), 0.5))
