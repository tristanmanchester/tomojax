"""Large host arrays follow one rule: in memory, or a temporary file with room checked first."""

from collections import namedtuple
import json
import logging
import tempfile

import h5py
import numpy as np
import pytest

import tomojax as tj
import tomojax.backends.api
from tomojax.core import host_volume


def _raw_nxtomo(path, views=7, rows=6, cols=9):
    """Raw counts with flats and a dark, every pixel different."""
    rng = np.random.default_rng(31)
    dark = rng.uniform(90, 110, (1, rows, cols))
    flats = rng.uniform(900, 1100, (2, rows, cols))
    samples = rng.uniform(200, 800, (views, rows, cols))
    frames = np.concatenate([dark, flats, samples]).astype(np.float32)
    key = np.r_[2, 1, 1, np.zeros(views)].astype(np.int32)
    angles = np.r_[0, 0, 0, np.linspace(0, 180, views, endpoint=False)].astype(np.float32)
    detector = tj.Detector(cols, rows, 1.0, 1.0)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        entry.create_group("geometry").attrs["type"] = "parallel"
        group = entry.create_group("instrument").create_group("detector")
        group.create_dataset("data", data=frames)
        group.create_dataset("image_key", data=key)
        group.attrs["detector_meta_json"] = json.dumps(detector.to_dict())
        sample = entry.create_group("sample").create_group("transformations")
        sample.create_dataset("rotation_angle", data=angles).attrs["units"] = "degree"


def test_a_corrected_stack_too_large_for_memory_is_held_in_a_temporary_file(
    tmp_path, monkeypatch, caplog
):
    # I13 159815 (3301 frames of 2560 x 2160, 73 GB corrected) on a 30 GB machine: tj.load
    # filled an in-memory stack and the kernel killed the process with no message.
    path = tmp_path / "raw.nxs"
    _raw_nxtomo(path)
    expected = tj.load(path)
    assert host_volume.scratch_directory(expected.projections) is None  # small: in memory
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(tomojax.backends.api, "host_available_memory_bytes", lambda: 1024)
    with caplog.at_level(logging.INFO, logger="tomojax"):
        actual = tj.load(path)
    assert host_volume.scratch_directory(actual.projections) == scratch
    assert not list(scratch.iterdir())  # unlinked: removed when released
    np.testing.assert_array_equal(np.asarray(actual.projections), expected.projections)
    assert f"held in a temporary file in {scratch}" in caplog.text


def test_a_temporary_file_is_refused_with_a_plain_error_when_its_directory_has_no_room(
    tmp_path, monkeypatch
):
    # A mapped page written past a full disk (a small RAM disk as TMPDIR) ended the process
    # with a bus error: the corrected stack and the solvers' work volumes alike.
    path = tmp_path / "raw.nxs"
    _raw_nxtomo(path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tomojax.backends.api, "host_available_memory_bytes", lambda: 1024)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(host_volume.shutil, "disk_usage", lambda _: usage(1000, 900, 100))
    with pytest.raises(MemoryError, match=r"needs 0\.0 GiB.*TMPDIR.*binning=.*selected"):
        tj.load(path)
    with pytest.raises(MemoryError, match=str(tmp_path)):
        host_volume.host_volume((4, 4, 4), tmp_path)
