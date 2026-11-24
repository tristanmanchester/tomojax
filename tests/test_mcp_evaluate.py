"""Held-out prediction must respect independent acquisitions and input poses."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.worker_adapter import Stage, admitted
import tomojax as tj
from tomojax.cli import operations as op, wire_options
from tomojax.corrections import Stripes
from tomojax.evaluation import Split
from tomojax.recon import api
from tomojax.wire.arrays import Member

STRIDE = op.Interleaved(stride=5, offset=2, guard=1)
CGLS = op.ReconstructSettings(method="cgls", iterations=2)


def evaluate(data, tmp_path, *, split=STRIDE, candidates=(CGLS,), corrections=()):
    ref = Member("input", "self")
    cmd = op.Evaluate(ref, candidates, split, corrections, CGLS)
    return admitted(op.Run(cmd), data, Stage(tmp_path), ref)


@pytest.fixture
def predictions(monkeypatch):
    records = []
    predict = api.predict_projections

    def record(geometry, volume, **kwargs):
        result = predict(geometry, volume, **kwargs)
        records.append((geometry, np.asarray(volume), np.asarray(result)))
        return result

    monkeypatch.setattr(api, "predict_projections", record)
    return records


def test_fits_ignore_every_excluded_observation(tmp_path, monkeypatch, predictions):
    scan = tj.simulate("shepp", size=12, n_views=60, rotation=360).scan
    ids = np.arange(len(scan.angles))
    train = ~np.isin(ids % 5, (1, 2, 3))
    x = np.array(scan.projections)
    x[~train] += np.random.default_rng(39).normal(size=x[~train].shape)
    corrected = []
    correct = tj.Scan.corrected

    def record(scan, *args, **kwargs):
        result = correct(scan, *args, **kwargs)
        corrected.append(result.projections)
        return result

    monkeypatch.setattr(tj.Scan, "corrected", record)
    fits = []
    align = tj.align

    def record_alignment(scan, **kwargs):
        fit = align(scan, **kwargs)
        fits.append(fit.poses)
        return fit

    monkeypatch.setattr(tj, "align", record_alignment)
    candidates = (
        CGLS,
        op.ReconstructSettings(
            method="cgls", config=wire_options.CGLSConfig(interpolation="cubic", iterations=2)
        ),
        op.AlignSettings(fit="center", method="opposite-views"),
        op.AlignSettings(fit="translations", motion="smooth", iterations=1, image_iterations=2),
    )
    args = dict(candidates=candidates, corrections=(wire_options.Stripes(width=3),))
    before = evaluate(scan, tmp_path, **args)
    after = evaluate(replace(scan, projections=x), tmp_path, **args)
    a, b, altered, held = corrected
    np.testing.assert_array_equal(a, altered)
    np.testing.assert_allclose(a, correct(scan.selected(ids[train]), Stripes(width=3)).projections)
    assert not np.allclose(b, held)
    for first, second in zip(fits[:2], fits[2:], strict=True):
        np.testing.assert_allclose(first, second, atol=1e-7)
    for i, (first, second) in enumerate(zip(predictions[:4], predictions[4:], strict=True)):
        g1, v1, y1 = first
        g2, v2, y2 = second
        np.testing.assert_allclose(v1, v2, atol=1e-6)
        if (poses := tj.Scan(y1, g1).poses) is not None:
            np.testing.assert_allclose(poses, tj.Scan(y2, g2).poses, atol=1e-7)
        assert before.scores[i]["diagnostic"] == after.scores[i]["diagnostic"]


def test_segments_and_noncommuting_local_rotations(tmp_path, monkeypatch, predictions):
    scan = tj.simulate(size=8, n_views=12).scan
    angles = np.tile(scan.angles, 2)
    positions = np.tile(np.arange(12, dtype=float) ** 1.1, 2)
    old = np.random.default_rng(9).normal(size=(24, 6)) * 4
    scan = tj.Scan(
        np.tile(scan.projections, (2, 1, 1)),
        replace(scan.geometry, angles=angles),
        view_positions=positions,
    ).with_poses(old)

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

    split = op.Interleaved(stride=4, offset=2)
    test = np.arange(2, 24, 4)
    train = np.delete(np.arange(24), test)
    for change in (False, True):

        def align(scan, *, change=change, **kwargs):
            return tj.Alignment(
                scan.with_poses(expected(train)) if change else scan,
                {"method": "joint", "calibration": {"roll": {"change": 2}} if change else {}},
            )

        monkeypatch.setattr(tj, "align", align)
        evaluate(
            scan,
            tmp_path,
            split=split,
            candidates=(op.AlignSettings(fit="translations", motion="smooth"),),
        )
        geometry, _, pixels = predictions[-1]
        np.testing.assert_allclose(
            tj.Scan(pixels, geometry).poses,
            expected(test) if change else old[test],
            atol=1e-6 if change else 1e-12,
        )
        if change:
            assert geometry.detector_roll == 2
    with pytest.raises(tj.InvalidParameter, match="inside its acquisition segment"):
        evaluate(scan, tmp_path, split=op.Angular(-20, 40))


def test_reject_removed_bracket_and_frames_domain(tmp_path, predictions, monkeypatch):
    scan = tj.simulate(size=8, n_views=24).scan
    y = np.array(scan.projections) * 0.1
    frames = tj.Frames(100 * np.exp(-y), scan.geometry, white_level=100)
    targets = []
    score = Split.score

    def record(split, prediction, target, **kwargs):
        targets.append(target)
        return score(split, prediction, target, **kwargs)

    monkeypatch.setattr(Split, "score", record)
    direct = evaluate(replace(scan, projections=y), tmp_path)
    raw = evaluate(frames, tmp_path)
    assert direct.split_fingerprint == raw.split_fingerprint
    n = len(targets) // 2
    np.testing.assert_allclose(targets[:n], targets[n:], atol=3e-7)
    np.testing.assert_allclose(predictions[0][1], predictions[1][1], atol=3e-7)
    y[0] = np.nan
    with pytest.raises(tj.InvalidParameter, match="bracket"):
        evaluate(replace(scan, projections=y), tmp_path, corrections=(wire_options.RejectViews(),))
    with pytest.raises(tj.InvalidParameter, match="detector splits"):
        evaluate(scan, tmp_path, candidates=(op.AlignSettings(fit="rotations"),))


def test_orbit_boundaries_follow_trajectories_not_calibration_segments(tmp_path, monkeypatch):
    from tomojax.geometry import ConeSegments

    scan = tj.simulate(size=8, n_views=24, geometry="cone", rotation=360).scan
    parts = tuple(scan.selected(np.arange(start, start + 6)).geometry for start in range(0, 24, 6))
    segmented = replace(scan, geometry=ConeSegments(parts, trajectory_lengths=(12, 12)))
    selected = []
    select = tj.Scan.selected

    def record(scan, ids):
        if scan is segmented:
            selected.append(ids)
        return select(scan, ids)

    monkeypatch.setattr(tj.Scan, "selected", record)
    evaluate(segmented, tmp_path, split=op.Angular(75, 30, period=180))
    assert selected[1].tolist() == [5, 6, 17, 18]
    with pytest.raises(tj.InvalidParameter, match="inside its acquisition segment"):
        evaluate(segmented, tmp_path, split=op.Angular(165, 30))


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


def test_cgls_predicts_odd_held_out_views_on_two_devices(tmp_path, predictions):
    import jax

    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("requires two devices")
    scan = tj.simulate(size=8, n_views=16).scan
    for selected in (devices[:1], devices):
        candidate = op.ReconstructSettings(
            method="cgls", iterations=2, devices=tuple(f"{d.platform}:{d.id}" for d in selected)
        )
        evaluate(scan, tmp_path, candidates=(candidate,), split=op.Interleaved())
    _, v1, y1 = predictions[0]
    _, v2, y2 = predictions[1]
    assert y2.shape == (3, 8, 8)
    np.testing.assert_allclose(y2, y1, atol=1e-5)
    np.testing.assert_allclose(v2, v1, atol=1e-5)
