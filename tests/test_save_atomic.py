"""A save replaces its file only when it is complete: a failure leaves the old file or none."""

import errno

import h5py
import numpy as np
import pytest

import tomojax as tj


def _reconstruction(value=1.0):
    grid = tj.Grid(6, 5, 4, 1, 1, 1)
    geometry = tj.ParallelGeometry(
        grid, tj.Detector(8, 4, 1, 1), np.linspace(0, 180, 6, endpoint=False)
    )
    scan = tj.Scan(np.full((6, 4, 8), value, np.float32), geometry)
    return tj.Reconstruction(
        volume=np.full((4, 5, 6), value, np.float32),
        geometry=scan.geometry,
        method="fbp",
        info={},
        scan=scan,
    )


def _fail_part_way(monkeypatch, suffix):
    """The disk fills once the volume is written (NXtomo) or part-way through the archive (NPZ)."""
    if suffix == ".nxs":
        written = h5py.Group.create_dataset

        def full(group, name, *args, **kwargs):
            dataset = written(group, name, *args, **kwargs)
            if group.name == "/entry" and name == "record":
                raise OSError(errno.ENOSPC, "No space left on device")
            return dataset

        monkeypatch.setattr(h5py.Group, "create_dataset", full)
    else:

        def full(file, **payload):
            file.write(b"\0" * 4096)
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(np, "savez_compressed", full)


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_a_save_that_fails_part_way_leaves_no_file(tmp_path, monkeypatch, suffix):
    # tj.save wrote straight to the final name: a full disk left a truncated file there.
    path = tmp_path / f"recon{suffix}"
    _fail_part_way(monkeypatch, suffix)
    with pytest.raises(
        OSError, match=r"stopped after .* written: .* free.*volume .* GiB, projections"
    ):
        tj.save(path, _reconstruction(), include_scan=True)
    assert list(tmp_path.iterdir()) == []  # neither the file nor a partial one


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_a_save_that_fails_over_a_good_file_keeps_it(tmp_path, monkeypatch, suffix):
    # Opening the final name for writing emptied a good file before the new one was complete.
    path = tmp_path / f"recon{suffix}"
    tj.save(path, _reconstruction(1.0), include_scan=True)
    _fail_part_way(monkeypatch, suffix)
    with pytest.raises(OSError, match="stopped after"):
        tj.save(path, _reconstruction(2.0), include_scan=True)
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(tj.load_reconstruction(path).volume, 1.0)


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_a_save_replaces_the_file_when_complete(tmp_path, suffix):
    path = tmp_path / f"recon{suffix}"
    tj.save(path, _reconstruction(1.0), include_scan=True)
    tj.save(path, _reconstruction(2.0), include_scan=True)
    assert list(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(tj.load_reconstruction(path).volume, 2.0)


def test_interrupted_slab_sequence_preserves_the_previous_generation(tmp_path):
    from tomojax.io.api import NXTomoMetadata, slab_volume

    path = tmp_path / "slabs.nxs"
    tj.save(path, _reconstruction(3))

    def interrupted_solve():
        with slab_volume(path, NXTomoMetadata(), (4, 5, 6)) as volume:
            volume[0] = -1
            raise InterruptedError()

    with pytest.raises(InterruptedError):
        interrupted_solve()
    assert list(tmp_path.iterdir()) == [path]
    with h5py.File(path) as saved:
        np.testing.assert_array_equal(saved["entry/volume/data"], 3)


def test_save_without_overwrite_preserves_a_file_created_during_the_write(tmp_path, monkeypatch):
    from tomojax.io.api import save_nxtomo

    path = tmp_path / "first.nxs"
    create = h5py.Group.create_dataset

    def competing_writer(group, name, *args, **kwargs):
        if group.name == "/entry" and name == "record":
            path.write_bytes(b"first writer")
        return create(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", competing_writer)
    with pytest.raises(FileExistsError):
        save_nxtomo(str(path), np.zeros((2, 1, 4), np.float32), overwrite=False)
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == b"first writer"
