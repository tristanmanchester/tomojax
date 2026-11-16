"""A save replaces its file only when it is complete: a failure leaves the old file or none."""

import errno

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
        volume=np.full((4, 5, 6), value, np.float32), scan=scan, method="fbp", info={}
    )


def _fail_part_way(monkeypatch, suffix):
    """The disk fills while the volume is written (NXtomo) or while the archive is (NPZ)."""
    if suffix == ".nxs":
        import tomojax._data._io_nxtomo as nxtomo

        written = nxtomo._write_volume_section

        def full(entry, meta, **kwargs):
            written(entry, meta, **kwargs)
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(nxtomo, "_write_volume_section", full)
    else:
        import tomojax._data._io_npz as npz

        def full(file, **payload):
            file.write(b"\0" * 4096)
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(npz.np, "savez_compressed", full)


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_a_save_that_fails_part_way_leaves_no_file(tmp_path, monkeypatch, suffix):
    # tj.save wrote straight to the final name: a full disk left a truncated file there.
    path = tmp_path / f"recon{suffix}"
    _fail_part_way(monkeypatch, suffix)
    with pytest.raises(
        OSError, match=r"stopped after .* written: .* free.*volume .* GiB, projections"
    ):
        tj.save(path, _reconstruction())
    assert list(tmp_path.iterdir()) == []  # neither the file nor a partial one


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_a_save_that_fails_over_a_good_file_keeps_it(tmp_path, monkeypatch, suffix):
    # Opening the final name for writing emptied a good file before the new one was complete.
    path = tmp_path / f"recon{suffix}"
    tj.save(path, _reconstruction(1.0))
    _fail_part_way(monkeypatch, suffix)
    with pytest.raises(OSError, match="stopped after"):
        tj.save(path, _reconstruction(2.0))
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(tj.load_reconstruction(path).volume, 1.0)


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_a_save_replaces_the_file_when_complete(tmp_path, suffix):
    path = tmp_path / f"recon{suffix}"
    tj.save(path, _reconstruction(1.0))
    tj.save(path, _reconstruction(2.0))
    assert list(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(tj.load_reconstruction(path).volume, 2.0)
