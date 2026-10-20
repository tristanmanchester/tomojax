"""Independent full-FFT and known-subpixel checks for nano CT scoring."""

from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
from scipy import fft, ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import nanotomo_metrics as metrics
from nanotomo_metrics import (
    fourier_shell_correlation,
    interleaved_views,
    register_translation,
    shift_error,
    threshold_crossing,
)


def test_fsc_matches_full_sphere_counts_signed_power_and_half_bit():
    rng = np.random.default_rng(913)
    for shape in ((19, 22, 25), (20, 21, 24)):
        a = rng.normal(size=shape).astype(np.float32)
        b = (0.6 * a + rng.normal(size=shape)).astype(np.float32)
        spacing = np.array([0.7, 0.8, 0.9])
        result = fourier_shell_correlation(a, b, spacing=spacing, taper=0)
        fa, fb = np.fft.fftn(a - a.mean()), np.fft.fftn(b - b.mean())
        frequencies = np.meshgrid(
            *(np.fft.fftfreq(n, d) for n, d in zip(shape, spacing, strict=True)), indexing="ij"
        )
        radius = np.sqrt(sum(f * f for f in frequencies))
        df = np.max(1 / (np.array(shape) * spacing))
        shell = np.floor(radius / df + 0.5)
        for index in range(1, len(result["frequency"])):
            selected = (shell == index) & (radius <= min(0.5 / spacing))
            x, y = fa[selected], fb[selected]
            expected = np.vdot(y, x).real / np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real)
            assert result["samples"][index] == selected.sum()
            np.testing.assert_allclose(result["correlation"][index], expected, atol=2e-7)
            n = selected.sum()
            s = (np.sqrt(2) - 1) / 2
            threshold = (s * np.sqrt(n) + 2 * np.sqrt(s) + 1) / (
                (s + 1) * np.sqrt(n) + 2 * np.sqrt(s)
            )
            np.testing.assert_allclose(result["half_bit"][index], threshold)
        negative = fourier_shell_correlation(a, -a, taper=0)
        np.testing.assert_allclose(negative["correlation"][1:], -1, atol=2e-7)
        assert negative["resolution"]["status"] == "below_band"
    crossing = threshold_crossing(
        np.arange(5) * 0.1, np.array([1.0, 0.9, 0.6, 0.2, 0.9]), np.full(5, 0.5), nyquist=0.5
    )
    np.testing.assert_allclose(crossing["frequency"], 0.225)
    np.testing.assert_allclose(crossing["half_period"], 1 / 0.45)


def test_registration_and_shared_translation_do_not_hide_per_view_errors():
    rng = np.random.default_rng(627)
    image = ndimage.gaussian_filter(rng.normal(size=(97, 103)), 1)
    image[:15] = image[-15:] = image[:, :15] = image[:, -15:] = 0
    shift = np.array([3.23, -4.47])
    moving = fft.ifft2(ndimage.fourier_shift(fft.fft2(image), shift)).real
    row, col = np.indices(image.shape)
    fitted, peak = register_translation(image + 0.01 * row - 0.02 * col, moving + 2, taper=0)
    np.testing.assert_allclose(fitted, -shift, atol=0.01)
    assert peak > 0.99
    angles = np.array([0.0, 90.0, 45.0, 135.0, 22.5, 112.5, 67.5, 157.5])
    left, right = interleaved_views(angles)
    np.testing.assert_array_equal(left, [0, 1, 2, 3])
    np.testing.assert_array_equal(right, [4, 5, 6, 7])
    c, s = np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))
    rotations = np.zeros((len(angles), 3, 3))
    rotations[:, 0, 0], rotations[:, 0, 1] = c, -s
    rotations[:, 1, 0], rotations[:, 1, 1] = s, c
    rotations[:, 2, 2] = 1
    reference = rng.normal(size=(len(angles), 2))
    translation = np.array([3.0, -2.0, 7.0])
    actual = reference + rotations[:, [0, 2], :] @ translation
    result = shift_error(actual, reference, rotations)
    assert result["vector_rmse_px"] < 1e-12
    actual[0] += [0.4, -0.2]
    assert shift_error(actual, reference, rotations)["vector_rmse_px"] > 0.1


def test_registration_recenters_window_for_large_motion_of_truncated_features():
    rng = np.random.default_rng(521)
    row, col = np.indices((91, 103))
    image = np.exp(-((row - 16) ** 2 + (col - 25) ** 2) / 1000)
    image += 0.08 * ndimage.gaussian_filter(rng.normal(size=image.shape), 2)
    shift = np.array([13.23, -17.47])
    moving = fft.ifft2(ndimage.fourier_shift(fft.fft2(image), shift)).real
    fitted, peak = register_translation(image, moving)
    np.testing.assert_allclose(fitted, -shift, atol=0.005)
    assert peak > 0.999


def test_half_data_pipeline_preserves_pose_corrections_and_physical_spacing():
    from nanotomo_metrics import half_dataset_fsc

    import tomojax as tj
    from tomojax.recon import FBPConfig

    grid, detector = tj.Grid(12, 12, 8, 0.7, 0.7, 0.7), tj.Detector(12, 8, 0.7, 0.7)
    angles = np.array([0.0, 90.0, 45.0, 135.0, 22.5, 112.5, 67.5, 157.5])
    rng = np.random.default_rng(172)
    poses = np.zeros((len(angles), 6), dtype=np.float32)
    poses[:, [3, 4]] = rng.normal(size=(len(angles), 2)) * 0.2
    scan = tj.Scan(
        rng.normal(size=(len(angles), 8, 12)).astype(np.float32),
        tj.ParallelGeometry(grid, detector, angles),
    ).with_poses(poses)
    config = FBPConfig(backprojector="jax")
    region = (slice(None), slice(None), slice(1, -1))
    result = half_dataset_fsc(scan, config=config, taper=0, region=region)
    expected = [
        np.asarray(tj.reconstruct(scan.selected(ix), "fbp", config=config).volume)[region]
        for ix in ([0, 1, 2, 3], [4, 5, 6, 7])
    ]
    direct = fourier_shell_correlation(*expected, spacing=(0.7, 0.7, 0.7), taper=0)
    np.testing.assert_allclose(result["correlation"], direct["correlation"])
    np.testing.assert_array_equal(result["halves"], [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert result["grid_shape"] == [12, 12, 8]
    assert result["scored_shape"] == [12, 12, 6]
    assert result["region"] == [[None, None, None], [None, None, None], [1, -1, None]]


def test_repeated_angle_observations_never_leak_into_both_fsc_halves():
    angles = np.array([0.0, 90.0, 45.0, 135.0, 0.0, 45.0])
    first, second = interleaved_views(angles)
    np.testing.assert_array_equal(first, [0, 1, 4])
    np.testing.assert_array_equal(second, [2, 3, 5])
    assert not set(angles[first]) & set(angles[second])
    np.testing.assert_array_equal(np.sort(np.r_[first, second]), np.arange(len(angles)))


def test_half_data_releases_spatial_volumes_before_fft(monkeypatch):
    import tomojax as tj

    references = []
    rng = np.random.default_rng(25)

    def reconstruct(*args, **kwargs):
        volume = rng.normal(size=(7, 8, 9)).astype(np.float32)
        references.append(weakref.ref(volume))
        return SimpleNamespace(volume=volume)

    transform = metrics.fft.rfftn

    def checked_fft(*args, **kwargs):
        assert all(reference() is None for reference in references)
        return transform(*args, **kwargs)

    monkeypatch.setattr(tj, "reconstruct", reconstruct)
    monkeypatch.setattr(metrics.fft, "rfftn", checked_fft)
    scan = SimpleNamespace(
        angles=np.arange(4),
        grid=SimpleNamespace(nx=7, ny=8, nz=9, vx=1, vy=1, vz=1),
        selected=lambda _: None,
    )
    metrics.half_dataset_fsc(scan, region=(slice(None), slice(None), slice(1, -1)))
