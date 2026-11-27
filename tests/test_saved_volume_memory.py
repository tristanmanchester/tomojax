"""Memmap a saved 2 GiB result and copy it between native containers by bounded reads."""

from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pytest

import tomojax as tj
from tomojax.core.host_volume import mapped_storage


def measure(mode: str, folder: str) -> None:
    root = Path(folder)
    shape = (512, 1024, 1024)
    grid = tj.Grid(1024, 1024, 512, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(4, 2, 1, 1), [0.0, 90.0])
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if mode == "write":
        source = np.memmap(root / "pixels", "<f4", "r", shape=shape)
        tj.save(root / "volume.nxs", tj.Reconstruction(source, geometry, "fbp", {}))
    else:
        result = tj.load_reconstruction(root / "volume.nxs")
        assert mapped_storage(result.volume) is not None
        assert not result.volume.flags.writeable
        np.testing.assert_array_equal(result.volume[511], 511.25)
    change = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert change < 256 * 1024, f"{mode} increased resident memory by {change / 1024:.1f} MiB"


@pytest.mark.slow
def test_native_volume_stays_bounded_on_save_and_load(tmp_path):
    with (tmp_path / "pixels").open("wb") as stream:
        for z in range(512):
            np.full((1024, 1024), z + 0.25, "<f4").tofile(stream)
    for mode in ("write", "read"):
        subprocess.run([sys.executable, __file__, mode, str(tmp_path)], check=True)
    (tmp_path / "pixels").unlink()
    (tmp_path / "volume.nxs").unlink()


if __name__ == "__main__":
    measure(*sys.argv[1:])
