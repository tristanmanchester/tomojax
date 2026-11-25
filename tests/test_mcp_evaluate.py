"""Held-out prediction must respect independent acquisitions and input poses."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import weakref

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


def test_dry_run_validates_partitions_without_reading_detector(tmp_path, monkeypatch):
    from tests.test_io_workflows import _labelled_dataset
    from tests.test_worker_adapters import trace_detector_reads
    from tomojax.io import save_dataset

    scan = tj.simulate(size=8, n_views=24).scan
    path = tmp_path / "scan.nxs"
    # The acquisition restarts before view 8, leaving view 7 unbracketed.
    tj.save(path, replace(scan, view_positions=np.r_[0:8, 0:16]))
    reads = trace_detector_reads(monkeypatch, "/entry/instrument/detector/data")
    ref = op.PathInput(str(path))

    def validate(step, candidate=CGLS, split=STRIDE):
        cmd = op.Evaluate(ref, (candidate,), split, (step,), CGLS)
        return admitted(op.Validate(operation=cmd), None, Stage(tmp_path), None)

    def forbid(*args, **kwargs):
        pytest.fail("dry run applied a correction")

    monkeypatch.setattr(tj.Scan, "corrected", forbid)
    monkeypatch.setattr(tj.Frames, "corrected", forbid)
    assert validate(wire_options.Stripes(width=3)).details["test_views"] == 3
    with pytest.raises(ValueError, match="width"):
        validate(wire_options.Stripes(width=0))
    with pytest.raises(ValueError, match="counts|transmission"):
        validate(wire_options.Zingers())
    with pytest.raises(ValueError, match="motion|poses"):
        validate(wire_options.Stripes(), op.AlignSettings(fit="rotations"))
    with pytest.raises(ValueError, match="guard"):
        validate(wire_options.Stripes(), split=op.Interleaved(stride=2, guard=1))
    save_dataset(path, _labelled_dataset([8, *([4] * 24), 2], [1, *([0] * 24), 2]))
    assert validate(wire_options.Zingers()).details["test_views"] == 4
    assert reads == []


@pytest.mark.parametrize("geometry", ["parallel", "cone"])
def test_prediction_batches_match_single_batch_and_release_before_next(
    tmp_path, monkeypatch, geometry
):
    scan = tj.simulate(size=8, n_views=124, geometry=geometry).scan
    candidate = op.ReconstructSettings(
        method="cgls",
        config=wire_options.CGLSConfig(
            interpolation="cubic" if geometry == "parallel" else "linear", iterations=2
        ),
    )
    settings = dict(candidates=(candidate,), split=op.Interleaved(stride=2, offset=1))
    reference = evaluate(scan, tmp_path, **settings)
    predict = api.predict_projections
    batches = []
    previous = None

    def measured(geometry, volume, **kwargs):
        nonlocal previous
        assert previous is None or previous() is None
        # Track the host view owned by the scorer, independent of JAX transfer caches.
        block = np.asarray(predict(geometry, volume, **kwargs)).view()
        batches.append(block.shape[0])
        previous = weakref.ref(block)
        return block

    monkeypatch.setattr(api, "predict_projections", measured)
    monkeypatch.setattr(
        "tomojax._evaluation._PREDICTION_BYTES", 7 * scan.detector.nv * scan.detector.nu * 4
    )
    (actual,) = evaluate(scan, tmp_path, **settings).scores
    (expected,) = reference.scores
    assert actual["count"] == expected["count"]
    assert actual["target_energy"] == expected["target_energy"]
    assert actual["mse"] == pytest.approx(expected["mse"], rel=3e-6)
    assert len(batches) > 2 and max(batches) == 7
    assert previous() is None


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


@pytest.mark.slow
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
        op.AlignSettings(method="opposite-views"),
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


@pytest.mark.parametrize("file_backed", [False, True])
def test_segments_and_noncommuting_local_rotations(tmp_path, monkeypatch, predictions, file_backed):
    scan = tj.simulate(size=8, n_views=12).scan
    angles = np.tile(scan.angles, 2)
    positions = np.tile(np.arange(12, dtype=float) ** 1.1, 2)
    old = np.random.default_rng(9).normal(size=(24, 6)) * 4
    scan = tj.Scan(
        np.tile(scan.projections, (2, 1, 1)),
        replace(scan.geometry, angles=angles),
        view_positions=positions,
    ).with_poses(old)

    if file_backed:
        np.save(tmp_path / "posed.npy", scan.projections)
        scan = replace(scan, projections=np.load(tmp_path / "posed.npy", mmap_mode="r"))

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
    scored = []
    score = Split.score

    def record(split, prediction, target, *, views):
        scored.append(views.start)
        return score(split, prediction, target, views=views)

    monkeypatch.setattr(Split, "score", record)
    evaluate(segmented, tmp_path, split=op.Angular(75, 30, period=180))
    assert scored == [5, 6, 17, 18]
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


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux RLIMIT_DATA accounting")
def test_evaluate_file_backed_scan_under_memory_limit(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_mcp_evaluate import run_limited_evaluation; "
            "import sys; run_limited_evaluation(sys.argv[1])",
            str(tmp_path),
        ],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "1"},
        check=True,
    )


def run_limited_evaluation(directory):
    import resource

    root = Path(directory)
    shape = (128, 4096, 32)
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 1, 1.0, 1.0, 1.0),
        tj.Detector(shape[2], shape[1], 1.0, 1.0),
        np.linspace(0, 180, shape[0], endpoint=False),
    )
    # View-dependent data on a tall detector. Fourier fits just the central row.
    signal = np.arange(1, shape[0] + 1, dtype=np.float32)[:, None, None] / shape[0]
    scan = tj.Scan(np.broadcast_to(signal, shape), geometry)
    args = dict(
        candidates=(op.ReconstructSettings(method="fourier"),),
        split=op.Interleaved(stride=shape[0], offset=64),
    )
    reference = evaluate(scan, root, **args)
    mapped = np.memmap(root / "scan.bin", mode="w+", dtype=np.float32, shape=shape)
    mapped[:] = scan.projections
    scan = replace(scan, projections=np.asarray(mapped))

    # Limit additional anonymous memory, while allowing file-backed mappings.
    vmdata = next(
        line
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmData:")
    )
    limit = int(vmdata.split()[1]) * 1024 + (24 << 20)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, resource.getrlimit(resource.RLIMIT_DATA)[1]))
    result = evaluate(scan, root, **args)
    assert result.scores == reference.scores


@pytest.mark.parametrize(
    ("geometry", "interpolation"),
    [("cone", "linear"), ("lamino", "cubic"), ("parallel", "analytic")],
)
def test_host_volume_projection_only_transfers_slabs(monkeypatch, geometry, interpolation):
    import jax.numpy as jnp

    from tomojax.recon import CGLSConfig, FBPConfig

    scan = tj.simulate(size=9, n_views=13, geometry=geometry).scan
    volume = np.random.default_rng(87).normal(size=scan.grid.shape).astype(np.float32)
    config = FBPConfig() if interpolation == "analytic" else CGLSConfig(interpolation=interpolation)
    if interpolation == "analytic":
        expected = tj.project(scan.geometry, jnp.asarray(volume), grid=scan.grid)

        def forbid_volume_copy(*args, **kwargs):
            pytest.fail("analytic scoring requested the projector's private volume layout")

        monkeypatch.setattr(tj.Projector, "__call__", forbid_volume_copy)
        resident = api.predict_projections(
            scan.geometry, jnp.asarray(volume), grid=scan.grid, config=config
        )
        np.testing.assert_allclose(resident, expected, rtol=2e-5, atol=2e-5)
    else:
        expected = api.predict_projections(
            scan.geometry, jnp.asarray(volume), grid=scan.grid, config=config
        )
    plane = volume[0].nbytes
    monkeypatch.setattr("tomojax.recon._prediction._SLAB_BYTES", 2 * plane)
    upload = jnp.asarray
    transfers = []

    def checked(array, *args, **kwargs):
        if isinstance(array, np.ndarray) and np.shares_memory(array, volume):
            transfers.append(array.nbytes)
            assert array.nbytes <= 2 * plane
        return upload(array, *args, **kwargs)

    monkeypatch.setattr(jnp, "asarray", checked)
    actual = api.predict_projections(scan.geometry, volume, grid=scan.grid, config=config)
    assert sum(transfers) == volume.nbytes
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_saved_scan_streamed_into_budgeted_partitions(tmp_path, monkeypatch):
    from tests.test_worker_adapters import trace_detector_reads

    data = tj.simulate(size=8, n_views=24)
    path = tmp_path / "saved.nxs"
    # Also exercises archives containing an unused ground-truth volume.
    tj.save(
        path,
        tj.Simulation(
            replace(data.scan, projections=np.asfortranarray(data.scan.projections)), data.phantom
        ),
    )
    corrections = (wire_options.BeamHardening(rates=(0.8, 1.2), weights=(0.5, 0.5)),)
    expected = evaluate(data.scan, tmp_path, corrections=corrections)
    monkeypatch.setattr("tomojax._evaluation._PARTITION_BYTES", data.scan.projections[0].nbytes)
    reads = trace_detector_reads(monkeypatch, "/entry/instrument/detector/data")
    monkeypatch.setattr(tj, "load", lambda *args: pytest.fail("eager loader"))
    corrected = tj.Scan.corrected

    def check_storage(scan, *args, **kwargs):
        assert isinstance(scan.projections, np.memmap)
        result = corrected(scan, *args, **kwargs)
        assert isinstance(result.projections, np.memmap)
        return result

    monkeypatch.setattr(tj.Scan, "corrected", check_storage)
    cmd = op.Evaluate(op.PathInput(str(path)), (CGLS,), STRIDE, corrections, CGLS)
    actual = admitted(op.Run(cmd), None, Stage(tmp_path), None)
    assert actual.scores == expected.scores
    assert reads and max(reads) <= data.scan.projections[0].nbytes


def test_candidate_and_alignment_image_lifetimes(tmp_path, monkeypatch):
    from tomojax import _evaluation

    scan = tj.simulate(size=8, n_views=24).scan
    volumes = []
    fit = _evaluation.reconstruct_resolved

    def reconstruct(plan):
        assert all(ref() is None for ref in volumes)
        result = fit(plan)
        volumes.append(weakref.ref(result.volume))
        return result

    def align(scan, **kwargs):
        assert all(ref() is None for ref in volumes)
        image = np.zeros(scan.grid.shape, dtype=np.float32)
        volumes.append(weakref.ref(image))
        return tj.Alignment(scan, {"method": "joint", "calibration": {}}, image, scan.grid)

    monkeypatch.setattr(_evaluation, "reconstruct_resolved", reconstruct)
    monkeypatch.setattr(tj, "align", align)
    evaluate(scan, tmp_path, candidates=(CGLS, op.AlignSettings(fit="center"), CGLS))
    assert len(volumes) == 4 and all(ref() is None for ref in volumes)


def test_raw_partitions_stage_counts_and_only_transfer_bracketing_flats(tmp_path, monkeypatch):
    import jax

    scan = tj.simulate(size=8, n_views=24).scan
    positions = np.arange(24, dtype=float)
    levels = 100 + positions[:, None, None] + np.zeros((24, 8, 8))
    data = np.asarray(scan.projections) * 0.1
    frames = tj.Frames(
        levels * np.exp(-data),
        scan.geometry,
        flats=levels,
        flat_positions=positions,
        view_positions=positions,
        linear_limit=(200, "camera"),
    )
    reference = evaluate(replace(scan, projections=data), tmp_path)
    plane = data[0].nbytes
    monkeypatch.setattr("tomojax._evaluation._PARTITION_BYTES", plane)
    monkeypatch.setattr("tomojax.corrections._engine._SIMPLE_FRAME_BYTES", plane)
    transfer = jax.device_put
    corrected = tj.Frames.corrected

    def check_transfer(array, *args, **kwargs):
        if isinstance(array, np.ndarray) and array.ndim == 3:
            assert len(array) <= 2
        return transfer(array, *args, **kwargs)

    def check_partition(frames, *steps):
        assert isinstance(frames.counts, np.memmap)
        result = corrected(frames, *steps)
        assert isinstance(result.projections, np.memmap)
        return result

    def run(frames, *steps):
        with monkeypatch.context() as transfers:
            transfers.setattr(jax, "device_put", check_transfer)
            return check_partition(frames, *steps)

    monkeypatch.setattr(tj.Frames, "corrected", run)
    actual = evaluate(frames, tmp_path)
    assert actual.scores[0]["mse"] == pytest.approx(reference.scores[0]["mse"], rel=3e-6)


def test_support_masks_are_staged_only_during_their_fit(tmp_path, monkeypatch):
    import h5py

    from tomojax.wire.arrays import Hdf5

    scan = tj.simulate(size=8, n_views=24).scan
    path = tmp_path / "masks.h5"
    with h5py.File(path, "w") as f:
        f["mask"] = np.ones(scan.grid.shape, dtype=bool)
    setting = op.ReconstructSettings(
        method="sirt", config=wire_options.SIRTConfig(support=Hdf5(str(path), "mask"), iterations=1)
    )
    monkeypatch.setattr("tomojax.cli._arguments._SUPPORT_BYTES", 64)
    masks = []
    read = h5py.Dataset.read_direct

    def track(dataset, dest, source_sel, dest_sel):
        assert isinstance(dest, np.memmap) and dest[dest_sel].nbytes <= 64
        if not masks or masks[-1]() is not dest:
            assert all(ref() is None for ref in masks)
            masks.append(weakref.ref(dest))
        return read(dataset, dest, source_sel, dest_sel)

    monkeypatch.setattr(h5py.Dataset, "read_direct", track)
    result = evaluate(scan, tmp_path, candidates=(setting, setting))
    assert len(masks) == 2 and result.scores[0] == result.scores[1]


def test_partitioning_preserves_copy_on_write_input(tmp_path):
    scan = tj.simulate(size=8, n_views=24).scan
    path = tmp_path / "source.npy"
    np.save(path, np.zeros_like(scan.projections))
    overlay = np.load(path, mmap_mode="c")
    overlay[:] = scan.projections
    expected = evaluate(scan, tmp_path)
    result = evaluate(replace(scan, projections=np.asarray(overlay)), tmp_path)
    assert result.scores == expected.scores
    np.testing.assert_array_equal(overlay, scan.projections)


def test_saved_matlab_evaluation_reads_only_selected_phase_views(tmp_path, monkeypatch):
    from tests.test_psi_reader import _write_v73
    from tests.test_worker_adapters import trace_detector_reads

    path = tmp_path / "phase.mat"
    pixels = np.random.default_rng(74).normal(size=(24, 5, 8))
    _write_v73(path, pixels, np.arange(24) * 7.5, 1e-3)
    expected = evaluate(tj.load(path), tmp_path)
    reads = trace_detector_reads(monkeypatch, "/aligned_phase_sinogram")
    cmd = op.Evaluate(op.PathInput(str(path)), (CGLS,), STRIDE, (), CGLS)
    actual = admitted(op.Run(cmd), None, Stage(tmp_path), None)
    assert actual.scores == expected.scores
    assert max(reads) == pixels[0].nbytes
    assert sum(reads) < pixels.nbytes
