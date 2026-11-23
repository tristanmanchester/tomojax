"""Previews register completed state and leave bulk volumes on their owners."""

from pathlib import Path

import msgspec
import numpy as np
import pytest

import tomojax as tj
from tomojax import progress
from tomojax.recon import CGLSConfig, FISTAConfig, MLEMConfig, SIRTConfig, SPDHGConfig
from tomojax.wire.progress import Event


@pytest.mark.parametrize(
    ("method", "config"),
    [
        ("sirt", SIRTConfig(iterations=5)),
        ("sirt", SIRTConfig(iterations=5, acceleration=True)),
        ("mlem", MLEMConfig(iterations=5)),
        ("cgls", CGLSConfig(iterations=5)),
        ("fista", FISTAConfig(iterations=5)),
        ("spdhg", SPDHGConfig(iterations=5)),
    ],
)
def test_final_preview_matches_all_three_planes_and_physical_coordinates(method, config, tmp_path):
    grid = tj.Grid(7, 4, 3, 0.87, 1.21, 1.1, vol_origin=(-2.3, -1.04, -1.6))
    detector = tj.Detector(9, 5, 1.2, 1.31, (0.03, 0.1))
    geom = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 5, endpoint=False))
    data = np.random.default_rng(12).uniform(0.1, 2, (5, 5, 9)).astype(np.float32)
    events = []
    with progress.watch(events.append, previews=True):
        result = tj.reconstruct(tj.Scan(data, geom), method, config=config)
    (first, *rest) = [e for e in events if isinstance(e, progress.Preview)]
    preview = rest[-1] if rest else first
    assert set(preview.arrays) == set("xyz")
    record = progress.to_json(preview, directory=tmp_path)
    msgspec.convert(record, type=Event)
    for axis, index, coordinate, expected in [
        ("x", 3, -2.3 + 3 * 0.87, result.volume[:, :, 3]),
        ("y", 2, -1.04 + 2 * 1.21, result.volume[:, 2, :]),
        ("z", 1, -1.6 + 1.1, result.volume[1, :, :]),
    ]:
        descriptor = record["arrays"][axis]
        (fixed,) = descriptor["fixed"]
        assert (fixed["name"], fixed["index"], fixed["unit"]) == (axis, index, "scan_length")
        assert fixed["coordinate"] == pytest.approx(coordinate)
        path = Path(descriptor["storage"]["path"])
        assert path.parent == tmp_path
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)
    assert len(list(tmp_path.iterdir())) == 3


def test_preview_rate_final_cancellation_and_disabled_cost(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(progress, "monotonic", lambda: now[0])
    built = []
    control = progress.Control()
    events = []

    def sample():
        built.append(now[0])
        return {}

    for enabled in (False, True):
        with (
            progress.watch(events.append, previews=enabled, control=control),
            progress.task("reconstruct", "counter", 5, "iterations"),
        ):
            task = progress.current_task.get()
            for i, t in enumerate([0.0, 0.1, 1.01, 1.1], 1):
                now[0] = t
                task.advance(i, {})
                progress.preview(sample, final=i == 4)
            progress.preview(sample, final=True)
            if enabled:
                control.cancel()
                task.advance(5, {})
                with pytest.raises(progress.Cancelled):
                    progress.preview(sample, final=True)
        if not enabled:
            assert built == []
    assert built == [0, 1.01, 1.1]
    assert len([e for e in events if isinstance(e, progress.Preview)]) == 3


def test_zero_updates_have_no_preview():
    grid = tj.Grid(4, 3, 2, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(5, 2, 1, 1), [0.0, 90.0])
    events = []
    with progress.watch(events.append, previews=True):
        tj.reconstruct(
            tj.Scan(np.ones((2, 2, 5), np.float32), geom), "sirt", config=SIRTConfig(iterations=0)
        )
    assert not any(isinstance(e, progress.Preview) for e in events)


def test_slice_before_transfer_and_use_private_storage_order():
    from tomojax.core.registration import planes  # check-public-imports: allow-private

    grid = tj.Grid(5, 4, 3, 0.5, 1.2, 2)
    data = np.arange(60).reshape(3, 4, 5)
    selections = []

    class Device:
        def __array__(self, *args, **kwargs):
            pytest.fail("downloaded the complete volume")

        def __getitem__(self, key):
            selections.append(key)
            return data.T[key]

    images = planes(Device(), grid, reversed_axes=True)
    assert len(selections) == 3
    for i, name in enumerate("zyx"):
        np.testing.assert_array_equal(images[name].data, data.take(data.shape[i] // 2, axis=i))


def test_correction_sinograms_carry_native_and_binned_domains():
    counts = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9) + 21
    grid = tj.Grid(7, 5, 3, 1, 1, 1)
    geom = tj.ParallelGeometry(grid, tj.Detector(9, 7, 1.2, 1.4, (0.2, -0.3)), [0, 15, 64, 80, 131])
    frames = tj.Frames(counts, geom, white_level=400)
    events = []
    with progress.watch(events.append, previews=True):
        corrected = frames.corrected(binning=2)
    (preview,) = [e for e in events if isinstance(e, progress.Preview)]
    before, after = preview.arrays["before"], preview.arrays["after"]
    assert (before.domain, after.domain) == ("counts", "line_integrals")
    assert before.fixed[0].coordinate == pytest.approx(-0.3)
    assert after.fixed[0].coordinate == pytest.approx(-0.3 - 0.7)
    np.testing.assert_array_equal(before.data, counts[:, 3, :])
    np.testing.assert_array_equal(after.data, corrected.projections[:, 1, :])
    np.testing.assert_array_equal(after.coordinates[0][1], geom.angles)


def test_raw_iterative_slab_preview_contains_whole_sweep():
    grid = tj.Grid(7, 6, 5, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(9, 5, 1, 1), np.arange(7) * 24.0)
    counts = np.random.default_rng(3432).uniform(10, 500, (7, 5, 9)).astype(np.float32)
    frames = tj.Frames(counts, geometry, white_level=601)
    events = []
    actual = np.empty(grid.shape, np.float32)
    with (
        progress.watch(events.append, previews=True),
        tj.reconstruct_slabs(frames, "sirt", config=SIRTConfig(iterations=2)) as slabs,
    ):
        for section, slab in slabs:
            actual[section] = slab.volume
    (preview,) = [e for e in events if isinstance(e, progress.Preview)]
    for i, axis in enumerate("zyx"):
        np.testing.assert_array_equal(
            preview.arrays[axis].data, np.take(actual, actual.shape[i] // 2, axis=i)
        )
