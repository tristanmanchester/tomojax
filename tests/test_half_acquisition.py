"""Independent Radon integrals on an offset detector through a complete circular orbit."""

import numpy as np
import pytest

import tomojax as tj


@pytest.mark.parametrize("method", ["fbp", "fourier"])
@pytest.mark.parametrize("side", [-1, 1])
def test_full_turn_resolves_the_axis_and_outer_field(method, side):
    # Gaussian Radon integrals on irregular, unpaired directions (including a repeat).
    angles = np.linspace(3, 363, 243, endpoint=False) + 0.08 * np.sin(np.arange(243))
    angles = np.r_[angles, angles[0]]
    grid = tj.Grid(96, 96, 1, 0.7, 0.7, 1.0)
    detector = tj.Detector(54, 1, 0.7, 1.0, center=(side * 14.7, 0))
    g = tj.ParallelGeometry(grid, detector, angles)
    u = (np.arange(54) - 26.5) * detector.du + detector.center[0]
    theta = np.deg2rad(angles)[:, None]
    x = (np.arange(96) - 47.5) * grid.vx
    truth = np.zeros((96, 96))
    p = np.zeros((len(angles), 54))
    for cx, cy, sigma, peak in [(0, 0, 8, 0.8), (21, -5, 2, 1.3), (-2, 1, 1, 0.6)]:
        distance = u - cx * np.cos(theta) + cy * np.sin(theta)
        p += np.sqrt(2 * np.pi) * sigma * peak * np.exp(-0.5 * (distance / sigma) ** 2)
        truth += peak * np.exp(-0.5 * ((x - cx) ** 2 + (x[:, None] - cy) ** 2) / sigma**2)
    volume = np.asarray(tj.reconstruct(tj.Scan(p[:, None].astype("f4"), g), method).volume)[0]
    radius = np.hypot(x, x[:, None])
    for mask in (radius < 4, radius < 16, (radius > 16) & (radius < 32)):
        # Detector sampling and linear angular interpolation broaden the narrow peaks.
        assert np.linalg.norm((volume - truth)[mask]) < 0.035 * np.linalg.norm(truth[mask])


def test_fitted_field_follows_the_measured_axis_and_keeps_world_origin(tmp_path):
    import h5py

    # Analytic line integrals on a 64-column camera with the axis at column 6.25.
    # Pixel edges at -0.5 and 63.5 cover a 114.5-pixel diameter over a full turn.
    angles = np.arange(301) * 1.2
    u = np.arange(64) - 6.25
    theta = np.deg2rad(angles)[:, None]
    p = np.zeros((len(angles), 64))
    for x, y, sigma in [(3, 1, 1.7), (-8, 9, 2), (29, -4, 3)]:
        p += 0.3 * np.exp(-0.5 * ((u - x * np.cos(theta) + y * np.sin(theta)) / sigma) ** 2)
    path = tmp_path / "full.nxs"
    with h5py.File(path, "w") as f:
        f["exchange/data"] = (10000 * np.exp(-p[:, None])).astype(np.float32)
        f["exchange/data_white"] = np.full((1, 1, 64), 10000, np.float32)
        f["exchange/data_dark"] = np.zeros((1, 1, 64), np.float32)
        f["exchange/theta"] = angles
        for axis in ("x", "y"):
            d = f.create_dataset(
                f"measurement/instrument/detector/actual_pixel_size_{axis}", data=0.7
            )
            d.attrs["units"] = "mm"
    initial = tj.inspect(path)["loading"]
    assert initial["full_turn"] and "axis_to_edge_px" not in initial
    aligned = tj.align(tj.load(path), fit="center")
    assert aligned.info["diagnostics"]["pairs"] == 64
    assert aligned.info["diagnostics"]["overlap_px"] == pytest.approx(13.5, abs=0.3)
    fit = aligned.scan
    assert 31.5 - fit.detector.center[0] / 0.7 == pytest.approx(6.25, abs=0.15)
    assert fit.grid.shape == (1, 115, 115)
    assert fit.grid.vol_center == (0, 0, 0)
    saved = tmp_path / "calibrated.nxs"
    tj.save(saved, fit)
    record = tj.inspect(saved)
    assert record["loading"]["axis_to_edge_px"] == pytest.approx(6.75, abs=0.15)
    assert record["loading"]["overlap_px"] == pytest.approx(13.5, abs=0.3)
    assert record["memory_estimates"]["reconstruction_grid_shape"] == [115, 115, 1]


@pytest.mark.parametrize("method", ["fbp", "fourier"])
def test_offset_detector_continues_only_beyond_the_complete_orbit(method):
    # A homogeneous cylinder extends past the far detector edge. Its radial
    # projection is known on both sides of the axis, including the missing tail.
    d = tj.Detector(48, 1, 1, 1, center=(18, 0))
    g = tj.ParallelGeometry(tj.Grid(84, 84, 1, 1, 1, 1), d, np.arange(181) * 2.0)
    radius, mu = 52, 0.02
    u = np.arange(48) - 5.5
    chord = 2 * mu * np.sqrt(radius**2 - u**2)
    p = np.broadcast_to(chord.astype("f4"), (181, 1, 48))
    result = tj.reconstruct(tj.Scan(p, g), method)
    rec = result.info["truncation"]
    assert rec["field_of_view_radius"] == pytest.approx(42.0)  # panel edge at 47.5, axis at 5.5
    assert rec["radius"] == pytest.approx(radius, rel=0.01)
    x = np.arange(84) - 41.5
    r = np.hypot(x[:, None], x)
    assert np.mean(np.asarray(result.volume)[0, r < 32]) == pytest.approx(mu, rel=0.04)


def test_resumed_offset_slabs_keep_their_physical_field_and_repeated_exposures(tmp_path):
    from tomojax.recon import FourierConfig

    # The closing view repeats zero; the second copy has a different exposure
    # error. Their mean is the analytic profile. Distinct rows catch slab strides.
    angles = np.arange(121) * 3.0
    det = tj.Detector(39, 5, 1, 1, center=(-13.25, 0))
    grid = tj.Grid(66, 66, 5, 1, 1, 1)
    u = np.arange(det.nu) - (det.nu - 1) / 2 + det.center[0]
    xy = np.arange(grid.nx) - (grid.nx - 1) / 2
    theta = np.deg2rad(angles)
    attenuation = []
    truth = []
    for k in range(5):
        x, y, sigma = -14 + 7 * k, -4 + k, 2.0
        distance = u - (x * np.cos(theta) - y * np.sin(theta))[:, None]
        p = np.sqrt(2 * np.pi) * sigma * 0.1 * np.exp(-0.5 * (distance / sigma) ** 2)
        p[0] += 0.02 * np.cos(u)
        p[-1] -= 0.02 * np.cos(u)
        attenuation.append(p)
        truth.append(0.1 * np.exp(-0.5 * ((xy - x) ** 2 + (xy[:, None] - y) ** 2) / sigma**2))
    counts = (1000 * np.exp(-np.stack(attenuation, axis=1))).astype(np.float32)
    frames = tj.Frames(counts, tj.ParallelGeometry(grid, det, angles), white_level=1000)
    settings = {"config": FourierConfig(slices_per_batch=2), "checkpoint": tmp_path / "resume"}
    output = np.zeros(grid.shape)
    with tj.reconstruct_slabs(frames, "fourier", **settings) as slabs:
        z, first = next(slabs)
        output[z] = first.volume
        next(slabs)
    info = {}
    with tj.reconstruct_slabs(frames, "fourier", info=info, **settings) as slabs:
        for z, remaining in slabs:
            assert z.start >= 2
            output[z] = remaining.volume
    for actual, expected in zip(output, truth, strict=True):
        assert np.linalg.norm(actual - expected) < 0.035 * np.linalg.norm(expected)
    assert info["truncation"]["field_of_view_radius"] == pytest.approx(32.75)


def test_off_panel_axis_reports_the_disk_no_view_can_measure(tmp_path):
    # Axis at column -2.5, outside the panel edges -0.5 and 23.5.
    detector = tj.Detector(24, 1, 0.7, 1, center=(14 * 0.7, 0))
    geometry = tj.ParallelGeometry(tj.Grid(52, 52, 1, 0.7, 0.7, 1), detector, np.arange(360))
    path = tmp_path / "gap.nxs"
    tj.save(path, tj.Scan(np.zeros((360, 1, 24), np.float32), geometry))
    record = tj.inspect(path)["loading"]
    assert record["overlap_px"] == 0
    assert record["central_gap_px"] == pytest.approx(4)
    assert record["field_of_view_px"] == pytest.approx(52)


@pytest.mark.parametrize("method", ["fbp", "fourier"])
@pytest.mark.parametrize("exterior", ["zero", "cylinder"])
def test_rolled_offset_panel_recovers_the_join_and_outer_detail(method, exterior):
    from tomojax.recon import FBPConfig, FourierConfig

    # Every physical z line crosses the tilted panel at different horizontal ends.
    # The broad central Gaussian crosses both panels' overlap. A second peak
    # lies outside the radius of a centered detector of the same width.
    roll = 18
    detector = tj.Detector(42, 29, 0.8, 1.1, center=(12.0, 0.2))
    grid = tj.Grid(77, 77, 7, 0.7, 0.7, 1.6)
    angles = np.linspace(3, 363, 211, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles, detector_roll=roll)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (np.arange(detector.nv)[:, None] - (detector.nv - 1) / 2) * detector.dv
    c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    ray = detector.center[0] + c * u - s * v
    height = detector.center[1] + s * u + c * v
    theta = np.deg2rad(angles)[:, None, None]
    z, y, x = [
        (a - (n - 1) / 2) * h
        for a, n, h in zip(
            np.ogrid[: grid.nz, : grid.ny, : grid.nx],
            grid.shape,
            (grid.vz, grid.vy, grid.vx),
            strict=True,
        )
    ]
    projections = np.zeros((len(angles), detector.nv, detector.nu))
    truth = np.zeros(grid.shape)
    for cx, cy, cz, width, depth in [(0.8, -1.2, 1, 3.8, 4.5), (19, -4, -1, 1.9, 3)]:
        distance = ray - cx * np.cos(theta) + cy * np.sin(theta)
        projections += (
            np.sqrt(2 * np.pi)
            * width
            * np.exp(-0.5 * ((distance / width) ** 2 + ((height - cz) / depth) ** 2))
        )
        truth += np.exp(
            -0.5 * (((x - cx) ** 2 + (y - cy) ** 2) / width**2 + ((z - cz) / depth) ** 2)
        )
    config = (FBPConfig if method == "fbp" else FourierConfig)(exterior=exterior)
    data = projections.astype(np.float32)
    rec = tj.reconstruct(tj.Scan(data, geometry), method, config=config)
    image = np.asarray(rec.volume)
    frames = tj.Frames(1000 * np.exp(-data), geometry, white_level=1000)
    with tj.reconstruct_slabs(frames, method, config=config) as slabs:
        streamed = np.concatenate([np.asarray(slab.volume) for _, slab in slabs])
    np.testing.assert_allclose(streamed, image, rtol=3e-4, atol=3e-5)
    radius = np.hypot(x, y)[0]
    for region in (radius < 6, (radius > 16) & (radius < 23)):
        assert np.linalg.norm((image - truth)[:, region]) < 0.05 * np.linalg.norm(truth[:, region])


@pytest.mark.parametrize("roll", [18, 90])
def test_detector_derived_grid_contains_every_rolled_panel_corner(tmp_path, roll):
    import h5py

    detector = tj.Detector(42, 29, 0.8, 1.1, center=(12.0, -0.3))
    geometry = tj.ParallelGeometry(
        tj.Grid(9, 9, 3, 1, 1, 1), detector, np.arange(36) * 10, detector_roll=roll
    )
    path = tmp_path / "panel.nxs"
    tj.save(path, tj.Scan(np.zeros((36, 29, 42), np.float32), geometry))
    with h5py.File(path, "r+") as handle:
        del handle["entry"].attrs["grid_meta_json"]
    loaded = tj.load(path)
    report = tj.inspect(path)
    # Project the actual rectangle edges into the horizontal and vertical lab axes.
    theta = np.deg2rad(roll)
    u = np.array([-1, 1]) * 42 * 0.8 / 2
    v = np.array([[-1], [1]]) * 29 * 1.1 / 2
    x = 12.0 + np.cos(theta) * u - np.sin(theta) * v
    z = np.sin(theta) * u + np.cos(theta) * v
    for n, pitch, target in [
        (loaded.grid.nx, loaded.grid.vx, abs(x).max()),
        (loaded.grid.nz, loaded.grid.vz, abs(z).max()),
    ]:
        assert target - 1e-10 <= n * pitch / 2 < target + pitch
    assert report["loading"]["full_turn"]
    if roll == 90:
        assert report["loading"]["field_of_view_px"] == pytest.approx(2 * abs(x).max() / 1.1)
    else:
        assert "overlap_px" not in report["loading"]  # the interval varies with height
