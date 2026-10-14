"""Raw HDF5 calibrations follow detector windows without eager pixel reads."""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest

import tomojax as tj

pytestmark = pytest.mark.surface


def _raw_file(path, layout):
    rows, cols = 5, 7
    pattern = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    flats = np.stack([1000 + pattern, 1200 + pattern, 1600 + pattern])
    darks = np.stack([10 + pattern, 20 + pattern])
    counts = np.stack([300 + pattern, 700 + pattern])
    with h5py.File(path, "w") as file:
        if layout == "marked":
            stack = np.stack(
                [flats[0], darks[0], counts[0], flats[1], counts[1], darks[1], flats[2]]
            )
            file.create_dataset("entry/instrument/detector/data", data=stack)
            file.create_dataset("entry/instrument/detector/image_key", data=[1, 2, 0, 1, 0, 2, 1])
            file.create_dataset(
                "entry/sample/transformations/rotation_angle", data=[0, 0, 0, 0, 90, 0, 0]
            )
        else:
            file.create_dataset("exchange/data", data=counts)
            file.create_dataset("exchange/data_white", data=flats)
            file.create_dataset("exchange/data_dark", data=darks)
            file.create_dataset("exchange/theta", data=[0, 90])
    return counts, flats, darks


@pytest.mark.parametrize("layout", ["marked", "exchange"])
def test_loading_selection_and_nested_crops_read_no_calibration_pixels(
    tmp_path, monkeypatch, layout
):
    path = tmp_path / "raw.h5"
    counts, flats, darks = _raw_file(path, layout)
    reads = []
    original_get = h5py.Dataset.__getitem__
    original_read = h5py.Dataset.read_direct

    def get(dataset, selection):
        if dataset.ndim == 3:
            reads.append((dataset.name, selection))
        return original_get(dataset, selection)

    def read(dataset, dest, source_sel=None, dest_sel=None):
        if dataset.ndim == 3:
            reads.append((dataset.name, source_sel))
        return original_read(dataset, dest, source_sel=source_sel, dest_sel=dest_sel)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", get)
    monkeypatch.setattr(h5py.Dataset, "read_direct", read)
    frames = tj.load_frames(path)
    selected = frames.selected([1]).cropped(slice(1, 5), slice(1, 7))
    selected = selected.cropped(slice(1, 3), slice(2, 5))
    assert reads == []
    assert selected.flats.shape == (3, 2, 3)
    assert selected.darks.shape == (2, 2, 3)
    np.testing.assert_array_equal(np.asarray(selected.flats), flats[:, 2:4, 3:6])
    np.testing.assert_array_equal(np.asarray(selected.darks), darks[:, 2:4, 3:6])
    with pytest.raises(ValueError, match="without copying"):
        np.array(selected.flats, copy=False)
    assert reads
    assert all(selection[1:] == (slice(2, 4), slice(3, 6)) for _, selection in reads)
    reference = replace(frames, counts=counts, flats=flats, darks=darks).corrected()
    actual = selected.corrected(batch_views=1)
    np.testing.assert_array_equal(actual.projections, reference.projections[1:2, 2:4, 3:6])
    assert actual.corrections == reference.corrections


class _LimitedCalibration:
    """Array-like calibration that refuses eager conversion and logs batches."""

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.requests = []

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selected):
        indices = np.arange(len(self))[selected]
        self.requests.append(indices)
        return self.values[selected]

    def __array__(self, *args, **kwargs):
        raise AssertionError("calibration must be read in bounded batches")


@pytest.mark.parametrize("several_sets", [False, True])
def test_calibration_means_bound_reads_without_losing_fp64_accumulation(several_sets):
    rows, cols = 512, 512
    rng = np.random.default_rng(6)
    flat_values = np.broadcast_to(rng.uniform(950, 1100, 257)[:, None, None], (257, rows, cols))
    dark_values = np.broadcast_to(rng.uniform(10, 15, 131)[:, None, None], (131, rows, cols))
    flat = _LimitedCalibration(flat_values)
    dark = _LimitedCalibration(dark_values)
    counts = np.full((3, rows, cols), 500.0, np.float32)
    positions = np.resize([0, 3], len(flat)) if several_sets else np.zeros(len(flat))
    geometry = tj.ParallelGeometry(
        tj.Grid(4, 4, 2, 1.0, 1.0, 1.0), tj.Detector(cols, rows, 1.0, 1.0), [0, 60, 120]
    )
    lazy = tj.Frames(counts, geometry, flats=flat, darks=dark, flat_positions=positions).corrected()
    means = [flat_values[positions == p, 0, 0].mean(dtype=np.float64) for p in np.unique(positions)]
    interpolated = np.interp(np.arange(3) + 0.5, np.unique(positions), means)
    expected = -np.log(
        (500.0 - dark_values[:, 0, 0].mean(dtype=np.float64))
        / (interpolated - dark_values[:, 0, 0].mean(dtype=np.float64))
    )
    np.testing.assert_allclose(lazy.projections[:, 0, 0], expected, rtol=2e-6)
    for stack in [flat, dark]:
        requested = np.concatenate(stack.requests)
        np.testing.assert_array_equal(np.sort(requested), np.arange(len(stack)))
        assert (
            max(len(indices) * rows * cols * stack.dtype.itemsize for indices in stack.requests)
            <= 256 * 2**20
        )
