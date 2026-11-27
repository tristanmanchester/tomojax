"""Write and reopen a 2 GiB lazy detector as views and sinograms."""

from pathlib import Path
import resource
import subprocess
import sys

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io.api import locate_frames


def measure(mode: str, folder: str, axis: str) -> None:
    root = Path(folder)
    output = root / "measurements.raw"
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if mode == "write":
        stack = locate_frames(str(root / "source.h5"), data_path="/detector").stack
        angles = np.linspace(0, 180, 512, endpoint=False)
        geometry = tj.ParallelGeometry(
            tj.Grid(4, 4, 2, 1, 1, 1), tj.Detector(1024, 1024, 1, 1), angles
        )
        tj.export(output, tj.Scan(stack, geometry), axis=axis)
    else:
        shape = (512, 1024, 1024) if axis == "view" else (1024, 512, 1024)
        with output.open("rb") as stream:
            for view, row, col in [(0, 0, 0), (19, 83, 991), (511, 1023, 1023)]:
                index = (view, row, col) if axis == "view" else (row, view, col)
                stream.seek(4 * int(np.ravel_multi_index(index, shape)))
                assert np.frombuffer(stream.read(4), "<f4")[0] == view * 10000 + row * 4 + col
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 512 * 1024


@pytest.mark.slow
def test_two_gibibytes_of_lazy_projections_stay_bounded_both_ways(tmp_path):
    source = tmp_path / "source.h5"
    plane = np.arange(1024, dtype=np.float32)[:, None] * 4 + np.arange(1024, dtype=np.float32)
    with h5py.File(source, "w") as file:
        stack = file.create_dataset("detector", (512, 1024, 1024), dtype="<f4")
        for i in range(512):
            stack[i] = 10000 * i + plane
    for axis in ("view", "v"):
        for mode in ("write", "read"):
            subprocess.run([sys.executable, __file__, mode, str(tmp_path), axis], check=True)
        (tmp_path / "measurements.raw").unlink()
        (tmp_path / "measurements.raw.json").unlink()
    source.unlink()


if __name__ == "__main__":
    measure(*sys.argv[1:])
