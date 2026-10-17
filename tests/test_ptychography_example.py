"""The measured-data recipe preserves axis order, phase sign and batching."""

from types import SimpleNamespace

from examples import align_ptychography as recipe
from examples.align_ptychography import make_scan, prepare_phase, read_psi_objects
import h5py
import numpy as np


def test_psi_recipe_preserves_input_metadata_and_reports_estimation_time(tmp_path, monkeypatch):
    path = tmp_path / "objects.mat"
    values = np.arange(3 * 13 * 11).reshape(3, 13, 11).astype(np.float32)
    packed = np.empty(values.shape, dtype=[("real", "f4"), ("imag", "f4")])
    packed["real"], packed["imag"] = values, -values
    with h5py.File(path, "w") as source:
        source["stack_object"] = packed
        source["rotation_angle"] = [[90, 0, 45]]
        source["lambda"], source["pixel_size"] = [[2e-10]], [[7e-9]]
    objects, angles, wavelength, pixel = read_psi_objects(
        path, rows=slice(2, 8), columns=slice(3, 10)
    )
    np.testing.assert_array_equal(objects, ((1 - 1j) * values[:, 3:10, 2:8]).transpose(0, 2, 1))
    np.testing.assert_array_equal(angles, [90, 0, 45])
    assert wavelength == 2e-10 and pixel == 7e-9
    monkeypatch.setattr(
        recipe, "read_psi_objects", lambda _path: (objects, angles, wavelength, pixel)
    )
    monkeypatch.setattr(
        recipe, "prepare_phase", lambda data, _air: np.zeros(data.shape, np.float32)
    )
    seen = {}

    def align(scan, *, mode, config):
        assert mode == "pose" and config.iterations == 20
        seen["scan"] = scan
        return SimpleNamespace(
            scan=scan,
            poses=np.zeros((3, 6)),
            info={"estimation_seconds": 0.04},
        )

    def reconstruct(scan, method):
        assert scan is seen["scan"] and method == "fourier"
        return SimpleNamespace(volume=np.ones((7, 7, 6), np.float32))

    monkeypatch.setattr(recipe.tj, "align", align)
    monkeypatch.setattr(recipe.tj, "reconstruct", reconstruct)
    delta, poses, report = recipe.run_example(path)
    np.testing.assert_allclose(delta, wavelength / (2 * np.pi * pixel))
    np.testing.assert_array_equal(poses, np.zeros((3, 6)))
    assert report["estimation_seconds"] == 0.04 and report["whole_seconds"] >= 0


def test_preparation_sign_air_weights_and_final_partial_batch():
    v, u = np.meshgrid(np.linspace(-1, 1, 31), np.linspace(-1, 1, 42), indexing="ij")
    sample = 12 * np.maximum(1 - 4 * (u * u + v * v), 0) ** 2
    truth = np.stack([sample, 0.7 * sample, 1.1 * sample]).astype(np.float32)
    plane = np.arange(3)[:, None, None] + 5 * u - 3 * v
    objects = ((1 + 0.2 * np.cos(u + v)) * np.exp(1j * (plane - truth))).astype(np.complex64)
    phase = prepare_phase(objects, abs(u) > 0.7, batch_size=2)
    np.testing.assert_allclose(phase, truth, atol=5e-5)
    scan = make_scan(phase, np.array([0, 90, 180]))
    assert scan.projections is phase
    assert (scan.grid.nx, scan.grid.ny, scan.grid.nz) == (42, 42, 31)
    assert (scan.detector.nv, scan.detector.nu) == (31, 42)
