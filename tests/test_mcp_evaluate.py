"""Held-out prediction must respect independent acquisitions and input poses."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax import _evaluation as evaluation  # check-public-imports: allow-private
from tomojax.corrections import RejectViews, Stripes

STRIDE = dict(stride=5, offset=2, guard=1)


def test_fits_ignore_every_excluded_observation():
    scan = tj.simulate("shepp", size=12, n_views=60, rotation=360).scan
    steps = (Stripes(width=3),)
    split, a, b = evaluation.partitions(scan, "interleaved", STRIDE, steps)
    x = np.array(scan.projections)
    x[~split.train.ravel()] += np.random.default_rng(39).normal(size=x[~split.train.ravel()].shape)
    _, altered, held = evaluation.partitions(
        replace(scan, projections=x), "interleaved", STRIDE, steps
    )
    np.testing.assert_array_equal(a.scan.projections, altered.scan.projections)
    np.testing.assert_allclose(
        a.scan.projections, scan.selected(a.indices).corrected(*steps).projections
    )
    assert not np.allclose(b.scan.projections, held.scan.projections)
    reference = dict(method="cgls", iterations=2)
    for kind, settings in (
        ("reconstruct", reference),
        (
            "reconstruct",
            dict(method="cgls", config=tj.recon.CGLSConfig(interpolation="cubic", iterations=2)),
        ),
        ("align", dict(fit="center", method="opposite-views")),
        (
            "align",
            dict(
                fit="translations", motion="smooth", config=None, iterations=1, image_iterations=2
            ),
        ),
    ):
        r1, h1, _, d1 = evaluation.fit_candidate(a, b, kind, settings, reference)
        r2, h2, _, d2 = evaluation.fit_candidate(altered, held, kind, settings, reference)
        np.testing.assert_allclose(r1.volume, r2.volume, atol=1e-6)
        if r1.scan.poses is not None:
            np.testing.assert_allclose(r1.scan.poses, r2.scan.poses, atol=1e-7)
            np.testing.assert_allclose(h1.poses, h2.poses, atol=1e-7)
        assert d1 == d2


def test_segments_and_noncommuting_local_rotations():
    sim = tj.simulate(size=8, n_views=12)
    scan = sim.scan
    angles = np.tile(scan.angles, 2)
    positions = np.tile(np.arange(12, dtype=float) ** 1.1, 2)
    double = tj.Scan(
        np.tile(scan.projections, (2, 1, 1)),
        replace(scan.geometry, angles=angles),
        view_positions=positions,
    )
    old = np.random.default_rng(9).normal(size=(24, 6)) * 4
    scan = double.with_poses(old)
    _, train, test = evaluation.partitions(
        scan, "interleaved", dict(stride=4, offset=2, guard=0), ()
    )
    unchanged = evaluation.transfer_alignment(
        train.scan,
        test.scan,
        tj.Alignment(train.scan, {"calibration": {}}),
        training_segments=train.segments,
        held_segments=test.segments,
        local=True,
    )
    np.testing.assert_allclose(unchanged.poses, old[test.indices], atol=1e-12)

    def expected(ids):
        # Rotate about local x by a linear function of acquisition position, with
        # a different intercept in orbit two. Translations are in the lab frame.
        out = old[ids].copy()
        update = positions[ids] * 0.5 + 10 * (ids >= 12)
        before = Rotation.from_euler("YXZ", out[:, [1, 0, 2]], degrees=True)
        out[:, :3] = (before * Rotation.from_euler("X", update[:, None], degrees=True)).as_euler(
            "YXZ", degrees=True
        )[:, [1, 0, 2]]
        out[:, 3] += update
        return out

    fit = tj.Alignment(
        train.scan.with_poses(expected(train.indices)), {"calibration": {"roll": {"change": 2}}}
    )
    result = evaluation.transfer_alignment(
        train.scan,
        test.scan,
        fit,
        training_segments=train.segments,
        held_segments=test.segments,
        local=True,
    )
    np.testing.assert_allclose(result.poses, expected(test.indices), atol=1e-6)
    assert result.geometry.detector_roll == 2
    with pytest.raises(tj.InvalidParameter, match="inside its acquisition segment"):
        evaluation.partitions(scan, "angular", dict(start=-20, width=40, guard=0, period=360), ())


def test_reject_removed_bracket_and_frames_domain():
    scan = tj.simulate(size=8, n_views=24).scan
    # Negative log of calibrated counts yields the same held-out line integrals.
    y = np.array(scan.projections) * 0.1
    frames = tj.Frames(100 * np.exp(-y), scan.geometry, white_level=100)
    direct = evaluation.partitions(replace(scan, projections=y), "interleaved", STRIDE, ())
    raw = evaluation.partitions(frames, "interleaved", STRIDE, ())
    assert direct[0].fingerprint == raw[0].fingerprint
    for x, r in zip(direct[1:], raw[1:], strict=True):
        np.testing.assert_allclose(x.scan.projections, r.scan.projections, atol=3e-7)
    y[0] = np.nan
    with pytest.raises(tj.InvalidParameter, match="bracket"):
        evaluation.partitions(replace(scan, projections=y), "interleaved", STRIDE, (RejectViews(),))
    with pytest.raises(tj.InvalidParameter, match="detector splits"):
        evaluation.fit_candidate(
            *direct[1:], "align", dict(fit="rotations", motion=None, config=None), {}
        )


def test_orbit_boundaries_follow_trajectories_not_calibration_segments():
    from tomojax.geometry import ConeSegments

    scan = tj.simulate(size=8, n_views=24, geometry="cone", rotation=360).scan
    parts = tuple(scan.selected(np.arange(start, start + 6)).geometry for start in range(0, 24, 6))
    geom = ConeSegments(parts, trajectory_lengths=(12, 12))
    segmented = replace(scan, geometry=geom)
    _, train, test = evaluation.partitions(
        segmented, "angular", dict(start=75, width=30, guard=0, period=180), ()
    )
    assert test.indices.tolist() == [5, 6, 17, 18]
    assert test.segments.tolist() == [1, 1, 2, 2]
    assert train.segments.max() == 2
    with pytest.raises(tj.InvalidParameter, match="inside its acquisition segment"):
        evaluation.partitions(
            segmented, "angular", dict(start=165, width=30, guard=0, period=360), ()
        )


def test_raw_path_and_frames_handle_use_the_same_transmission_pipeline(tmp_path):
    from tests.test_mcp_stdio import connection
    from tomojax.io import ProjectionDataset, save_dataset
    from tomojax.io.api import LoadedNXTomo, NXTomoMetadata

    scan = tj.simulate(size=8, n_views=20).scan
    counts = np.exp(-0.1 * np.asarray(scan.projections))
    counts[:, 4, 4] = 2
    raw = np.concatenate([np.ones_like(counts[:1]), counts, np.zeros_like(counts[:1])])
    metadata = NXTomoMetadata(
        angles=np.r_[0, scan.angles, 0],
        image_key=np.r_[1, np.zeros(len(counts), dtype=np.int32), 2],
        grid=scan.grid,
        detector=scan.detector,
    )
    save_dataset(tmp_path / "raw.nxs", ProjectionDataset.from_nxtomo(LoadedNXTomo(raw, metadata)))
    with connection(tmp_path) as c:
        handle = c.call("open", path="raw.nxs", kind="frames")["structuredContent"]["handle"]
        args = dict(
            candidates=[dict(type="reconstruct", method="cgls", iterations=2)],
            corrections=[dict(type="Zingers", threshold=0.5)],
        )
        scores = []
        for source in ("raw.nxs", handle):
            result = c.call("evaluate", source=source, **args)
            assert not result["isError"], result
            scores.append(result["structuredContent"]["scores"])
        (path_score,), (handle_score,) = scores
        assert path_score["count"] == handle_score["count"] == 4 * 8**2
        assert path_score["target_energy"] == pytest.approx(handle_score["target_energy"])
        assert path_score["mse"] == pytest.approx(handle_score["mse"], rel=2e-6)


def test_cgls_predicts_odd_held_out_views_on_two_devices():
    import jax

    from tomojax.recon.api import predict_projections

    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("requires two devices")
    scan = tj.simulate(size=8, n_views=16).scan
    _, train, test = evaluation.partitions(
        scan, "interleaved", dict(stride=5, offset=2, guard=0), ()
    )
    predictions, volumes = [], []
    for selected in (devices[:1], devices):
        settings = dict(method="cgls", config=tj.recon.CGLSConfig(devices=selected, iterations=2))
        fit, held, config, _ = evaluation.fit_candidate(train, test, "reconstruct", settings, {})
        predictions.append(
            np.asarray(predict_projections(held.geometry, fit.volume, grid=fit.grid, config=config))
        )
        volumes.append(np.asarray(fit.volume))
    assert predictions[1].shape == (3, 8, 8)
    np.testing.assert_allclose(predictions[1], predictions[0], atol=1e-5)
    np.testing.assert_allclose(volumes[1], volumes[0], atol=1e-5)
