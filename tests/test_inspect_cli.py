from __future__ import annotations

import json

import h5py
import imageio.v3 as iio
import numpy as np

from tomojax.cli import inspect as inspect_cli
from tomojax.data.io_hdf5 import NXTomoMetadata, save_nxtomo


def _write_minimal_nxtomo(path):
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["definition"] = "NXtomo"

        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset(
            "data",
            data=np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
        )
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))

        sample = entry.create_group("sample")
        sample.create_dataset("name", data="sample")
        rotation_angle = sample.create_group("transformations").create_dataset(
            "rotation_angle",
            data=np.array([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"


def test_inspect_cli_reports_minimal_file(tmp_path, capsys):
    path = tmp_path / "minimal.nxs"
    _write_minimal_nxtomo(path)

    status = inspect_cli.main([str(path)])

    captured = capsys.readouterr()
    assert status == 0
    assert "Projection shape: [2, 4, 4]" in captured.out
    assert "Dtype: float32" in captured.out
    assert "Views: 2" in captured.out
    assert "Detector shape: {'nv': 4, 'nu': 4}" in captured.out
    assert "Angle coverage: 90 deg" in captured.out
    assert "Geometry type: not found" in captured.out
    assert "Detector metadata: not found" in captured.out
    assert "Flats/darks: not found" in captured.out
    assert "Alignment parameters: not found" in captured.out


def test_inspect_cli_writes_stable_json_for_minimal_file(tmp_path):
    path = tmp_path / "minimal.nxs"
    json_path = tmp_path / "inspect" / "minimal.json"
    _write_minimal_nxtomo(path)

    status = inspect_cli.main([str(path), "--json", str(json_path)])

    assert status == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "input_path",
        "projection",
        "angles",
        "geometry",
        "detector_metadata",
        "flats_darks",
        "alignment",
        "memory_estimates",
    }
    assert payload["schema_version"] == 1
    assert payload["projection"]["shape"] == [2, 4, 4]
    assert payload["projection"]["dtype"] == "float32"
    assert payload["projection"]["n_views"] == 2
    assert payload["projection"]["detector_shape"] == {"nv": 4, "nu": 4}
    assert payload["projection"]["nonfinite"] == {
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
        "inf_count": 0,
    }
    assert payload["angles"]["coverage_deg"] == 90.0
    assert payload["geometry"]["found"] is False
    assert payload["detector_metadata"]["found"] is False
    assert payload["alignment"]["found"] is False
    assert payload["memory_estimates"]["feasible"] is True


def test_inspect_cli_reports_optional_metadata(tmp_path):
    path = tmp_path / "metadata.nxs"
    projections = np.zeros((3, 2, 4), dtype=np.float32)
    metadata = NXTomoMetadata(
        thetas_deg=np.array([0.0, 45.0, 90.0], dtype=np.float32),
        image_key=np.array([0, 1, 2], dtype=np.int32),
        grid={"nx": 5, "ny": 6, "nz": 7, "vx": 1.0, "vy": 1.0, "vz": 1.0},
        detector={"nu": 4, "nv": 2, "du": 1.25, "dv": 2.5, "det_center": [0.5, -0.5]},
        geometry_type="lamino",
        geometry_meta={"tilt_deg": 35.0, "tilt_about": "x"},
        align_params=np.zeros((3, 5), dtype=np.float32),
    )
    save_nxtomo(path, projections=projections, metadata=metadata)
    json_path = tmp_path / "metadata.json"

    status = inspect_cli.main([str(path), "--json", str(json_path)])

    assert status == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["geometry"]["found"] is True
    assert payload["geometry"]["type"] == "lamino"
    assert payload["geometry"]["meta_found"] is True
    assert payload["geometry"]["meta_keys"] == ["tilt_about", "tilt_deg"]
    assert payload["detector_metadata"] == {
        "found": True,
        "nu": 4,
        "nv": 2,
        "du": 1.25,
        "dv": 2.5,
        "det_center": [0.5, -0.5],
    }
    assert payload["flats_darks"]["flats_present"] is True
    assert payload["flats_darks"]["darks_present"] is True
    assert payload["flats_darks"]["flat_count"] == 1
    assert payload["flats_darks"]["dark_count"] == 1
    assert payload["alignment"]["found"] is True
    assert payload["alignment"]["params_found"] is True
    assert payload["alignment"]["params_shape"] == [3, 5]
    assert payload["memory_estimates"]["reconstruction_grid_shape"] == [5, 6, 7]


def test_inspect_cli_tolerates_missing_optional_metadata(tmp_path):
    path = tmp_path / "missing_optional.nxs"
    _write_minimal_nxtomo(path)
    with h5py.File(path, "r+") as f:
        del f["/entry/instrument/detector/image_key"]
        del f["/entry/sample/transformations/rotation_angle"]
    json_path = tmp_path / "missing_optional.json"

    status = inspect_cli.main([str(path), "--json", str(json_path)])

    assert status == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["projection"]["found"] is True
    assert payload["angles"]["found"] is False
    assert payload["flats_darks"]["image_key_found"] is False
    assert payload["geometry"]["found"] is False
    assert payload["alignment"]["found"] is False


def test_inspect_cli_writes_projection_quicklook(tmp_path):
    path = tmp_path / "quicklook.nxs"
    quicklook_path = tmp_path / "quicklooks" / "projection.png"
    _write_minimal_nxtomo(path)

    status = inspect_cli.main([str(path), "--quicklook", str(quicklook_path)])

    assert status == 0
    image = iio.imread(quicklook_path)
    assert image.shape == (4, 4)
    assert image.dtype == np.uint8


def test_inspect_cli_reports_missing_file(tmp_path, capsys):
    path = tmp_path / "missing.nxs"

    status = inspect_cli.main([str(path)])

    captured = capsys.readouterr()
    assert status == 2
    assert "ERROR: file not found" in captured.err


def test_inspect_cli_rejects_file_without_projection_data(tmp_path, capsys):
    path = tmp_path / "no_projection.nxs"
    with h5py.File(path, "w") as f:
        f.create_group("entry")

    status = inspect_cli.main([str(path)])

    captured = capsys.readouterr()
    assert status == 1
    assert "Could not find projections dataset" in captured.err
