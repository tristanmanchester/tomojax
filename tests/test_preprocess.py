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
        assert (
            _attr_to_str(handle["/entry/processing/tomojax/preprocess"].attrs["output_domain"])
            == "absorption"
        )


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
        assert handle["/entry/processing/tomojax/preprocess/flat_mean"].dtype == np.dtype("float64")


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


def test_preprocess_reject_views_keeps_projections_angles_and_metadata_aligned(tmp_path):
    frames = np.array(
        [
            np.full((1, 1), 1.0, dtype=np.float32),  # dark
            np.full((1, 1), 2.0, dtype=np.float32),  # sample view 0
            np.full((1, 1), 3.0, dtype=np.float32),  # sample view 1
            np.full((1, 1), 11.0, dtype=np.float32),  # flat
            np.full((1, 1), 4.0, dtype=np.float32),  # sample view 2
            np.full((1, 1), 5.0, dtype=np.float32),  # sample view 3
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 1, 0, 0]))
    out_path = tmp_path / "reject.nxs"

    result = preprocess_nxtomo(raw_path, out_path, PreprocessConfig(reject_views="1:3"))

    assert result.sample_count == 2
    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(
        corrected.projections[:, 0, 0],
        np.array([0.1, 0.4], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(corrected.thetas_deg, np.array([30.0, 150.0], dtype=np.float32))
    np.testing.assert_array_equal(corrected.image_key, np.zeros((2,), dtype=np.int32))

    with h5py.File(out_path, "r") as handle:
        provenance = handle["/entry/processing/tomojax/preprocess"].attrs
        assert json.loads(provenance["final_sample_view_indices"]) == [0, 3]
        assert json.loads(provenance["final_raw_frame_indices"]) == [1, 5]


def test_preprocess_select_and_reject_files_are_deduped_in_sample_view_order(tmp_path):
    frames = np.array(
        [
            np.full((1, 1), 1.0, dtype=np.float32),
            np.full((1, 1), 2.0, dtype=np.float32),
            np.full((1, 1), 3.0, dtype=np.float32),
            np.full((1, 1), 4.0, dtype=np.float32),
            np.full((1, 1), 5.0, dtype=np.float32),
            np.full((1, 1), 11.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 0, 0, 1]))
    select_file = tmp_path / "select.txt"
    reject_file = tmp_path / "reject.txt"
    select_file.write_text("3, 1:4\n# duplicate below\n1\n", encoding="utf-8")
    reject_file.write_text("2 2\n", encoding="utf-8")
    out_path = tmp_path / "select_reject.nxs"

    preprocess_nxtomo(
        raw_path,
        out_path,
        PreprocessConfig(select_views_file=select_file, reject_views_file=reject_file),
    )

    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(corrected.projections[:, 0, 0], np.array([0.2, 0.4]))
    np.testing.assert_allclose(corrected.thetas_deg, np.array([60.0, 120.0], dtype=np.float32))
    with h5py.File(out_path, "r") as handle:
        provenance = handle["/entry/processing/tomojax/preprocess"].attrs
        assert json.loads(provenance["final_sample_view_indices"]) == [1, 3]


def test_preprocess_crop_updates_output_shape_and_detector_metadata(tmp_path):
    dark = np.zeros((4, 5), dtype=np.float32)
    sample = np.arange(20, dtype=np.float32).reshape(4, 5) + 1.0
    flat = np.full((4, 5), 10.0, dtype=np.float32)
    raw_path = _raw_path(
        tmp_path,
        frames=np.stack([dark, sample, flat]),
        image_key=np.array([2, 0, 1]),
    )
    out_path = tmp_path / "crop.nxs"

    preprocess_nxtomo(raw_path, out_path, PreprocessConfig(crop="1:3,2:5"))

    corrected = load_nxtomo(str(out_path))
    assert corrected.projections.shape == (1, 2, 3)
    np.testing.assert_allclose(corrected.projections[0], sample[1:3, 2:5] / 10.0)
    assert corrected.detector["nu"] == 3
    assert corrected.detector["nv"] == 2
    assert corrected.detector["du"] == 1.25
    assert corrected.detector["dv"] == 2.5
    np.testing.assert_allclose(corrected.detector["det_center"], [1.75, -0.5])

    with h5py.File(out_path, "r") as handle:
        preprocess = handle["/entry/processing/tomojax/preprocess"]
        np.testing.assert_allclose(preprocess["flat_mean"][...], np.full((2, 3), 10.0))
        assert json.loads(preprocess.attrs["crop_bounds"]) == {
            "x0": 2,
            "x1": 5,
            "y0": 1,
            "y1": 3,
        }


def test_preprocess_filters_optional_sample_length_alignment_metadata(tmp_path):
    frames = np.array(
        [
            np.zeros((1, 1), dtype=np.float32),
            np.full((1, 1), 0.2, dtype=np.float32),
            np.full((1, 1), 0.3, dtype=np.float32),
            np.full((1, 1), 0.4, dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 0, 1]))
    align_params = np.arange(15, dtype=np.float32).reshape(3, 5)
    angle_offset = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    with h5py.File(raw_path, "r+") as handle:
        align = handle.require_group("/entry/processing/tomojax/align")
        align.create_dataset("thetas", data=align_params)
        align.create_dataset("angle_offset_deg", data=angle_offset)

    out_path = tmp_path / "optional_metadata.nxs"

    preprocess_nxtomo(raw_path, out_path, PreprocessConfig(reject_views="1"))

    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(corrected.align_params, align_params[[0, 2]])
    np.testing.assert_allclose(corrected.angle_offset_deg, angle_offset[[0, 2]])
    with h5py.File(out_path, "r") as handle:
        filtered = json.loads(
            handle["/entry/processing/tomojax/preprocess"].attrs[
                "optional_sample_metadata_filtered"
            ]
        )
        assert filtered == {"align_params": True, "angle_offset_deg": True}


def test_preprocess_auto_reject_nonfinite_drops_bad_corrected_view(tmp_path):
    frames = np.array(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.full((1, 2), 0.5, dtype=np.float32),
            np.array([[np.nan, 0.5]], dtype=np.float32),
            np.full((1, 2), 1.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 1]))
    out_path = tmp_path / "nonfinite_reject.nxs"

    result = preprocess_nxtomo(
        raw_path,
        out_path,
        PreprocessConfig(auto_reject="nonfinite"),
    )

    assert result.sample_count == 1
    corrected = load_nxtomo(str(out_path))
    np.testing.assert_allclose(corrected.projections, np.full((1, 1, 2), 0.5))
    np.testing.assert_allclose(corrected.thetas_deg, np.array([30.0], dtype=np.float32))
    with h5py.File(out_path, "r") as handle:
        auto = json.loads(handle["/entry/processing/tomojax/preprocess"].attrs["auto_reject"])
        assert auto["rejected_sample_view_indices"] == [1]
        assert auto["rejected_reasons"] == {"1": ["nonfinite"]}


def test_preprocess_auto_reject_outliers_uses_robust_view_medians(tmp_path):
    values = [1.0, 1.1, 1.2, 50.0, 1.1]
    frames = [np.zeros((1, 2), dtype=np.float32)]
    frames.extend(np.full((1, 2), value, dtype=np.float32) for value in values)
    frames.append(np.ones((1, 2), dtype=np.float32))
    raw_path = _raw_path(
        tmp_path,
        frames=np.stack(frames),
        image_key=np.array([2, 0, 0, 0, 0, 0, 1]),
    )
    out_path = tmp_path / "outlier_reject.nxs"

    preprocess_nxtomo(
        raw_path,
        out_path,
        PreprocessConfig(auto_reject="outliers", outlier_z_threshold=6.0),
    )

    corrected = load_nxtomo(str(out_path))
    assert corrected.projections.shape == (4, 1, 2)
    np.testing.assert_allclose(corrected.projections[:, 0, 0], np.array([1.0, 1.1, 1.2, 1.1]))
    with h5py.File(out_path, "r") as handle:
        auto = json.loads(handle["/entry/processing/tomojax/preprocess"].attrs["auto_reject"])
        assert auto["rejected_sample_view_indices"] == [3]
        assert auto["rejected_reasons"] == {"3": ["outlier"]}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PreprocessConfig(reject_views="abc"), "invalid view index"),
        (PreprocessConfig(reject_views="9"), "out of bounds"),
        (PreprocessConfig(select_views="1:1"), "non-empty"),
        (PreprocessConfig(crop="0:2,0:9"), "out of bounds"),
        (PreprocessConfig(crop="1:1,0:1"), "non-empty"),
        (PreprocessConfig(reject_views="0:2"), "removed all sample views"),
    ],
)
def test_preprocess_invalid_view_and_crop_ranges_fail_clearly(tmp_path, config, message):
    frames = np.array(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.full((2, 2), 0.5, dtype=np.float32),
            np.full((2, 2), 0.6, dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 1]))

    with pytest.raises(ValueError, match=message):
        preprocess_nxtomo(raw_path, tmp_path / "bad.nxs", config)


def test_preprocess_warns_when_rejection_changes_angular_coverage(tmp_path, caplog):
    frames = np.array(
        [
            np.zeros((1, 1), dtype=np.float32),
            np.full((1, 1), 0.5, dtype=np.float32),
            np.full((1, 1), 0.6, dtype=np.float32),
            np.full((1, 1), 0.7, dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 0, 1]))

    caplog.set_level("WARNING")
    preprocess_nxtomo(raw_path, tmp_path / "coverage.nxs", PreprocessConfig(reject_views="0"))

    assert any("changed angular coverage" in record.message for record in caplog.records)


def test_preprocess_cli_combines_crop_reject_and_auto_reject(tmp_path, capsys):
    frames = np.array(
        [
            np.zeros((2, 3), dtype=np.float32),
            np.full((2, 3), 0.5, dtype=np.float32),
            np.full((2, 3), 0.6, dtype=np.float32),
            np.array([[0.7, 0.7, 0.7], [np.nan, 0.7, 0.7]], dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
        ],
        dtype=np.float32,
    )
    raw_path = _raw_path(tmp_path, frames=frames, image_key=np.array([2, 0, 0, 0, 1]))
    out_path = tmp_path / "cli_combined.nxs"

    status = preprocess_cli.main(
        [
            raw_path,
            str(out_path),
            "--crop",
            "0:1,0:2",
            "--reject-views",
            "1",
            "--auto-reject",
            "both",
        ]
    )

    assert status == 0
    assert "shape=[2, 1, 2]" in capsys.readouterr().out
    corrected = load_nxtomo(str(out_path))
    assert corrected.projections.shape == (2, 1, 2)
