"""Bound both the writer and an independent reader of a larger-than-memory export."""

from __future__ import annotations

from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pytest
import tifffile

import tomojax as tj

MIB = 1 << 20


def mapped_result(path: Path, depth: int) -> tj.Reconstruction:
    data = np.memmap(path, "<f4", mode="r", shape=(depth, 1024, 1024))
    grid = tj.Grid(1024, 1024, depth, 1.0, 1.0, 1.0)
    return tj.Reconstruction(
        data, tj.ParallelGeometry(grid, tj.Detector(4, 4, 1, 1), np.arange(3.0)), "fbp", {}
    )


def check_last(path: Path, recon: tj.Reconstruction, axis: str) -> None:
    # The last output plane is axis x or z of the same random input.
    last = recon.volume[-1] if axis == "z" else recon.volume[:, :, -1]
    # Read the reference by offsets; touching a strided memmap here would measure the oracle.
    expected = []
    with Path(recon.volume.filename).open("rb") as stream:
        for y, x in [(0, 0), (1, 17), (last.shape[0] - 1, last.shape[1] - 1)]:
            i = (recon.volume.shape[0] - 1, y, x) if axis == "z" else (y, x, 1023)
            stream.seek(4 * int(np.ravel_multi_index(i, recon.volume.shape)))
            expected.append(np.frombuffer(stream.read(4), "<f4")[0])
    if path.is_dir():
        actual = tifffile.imread(sorted(path.glob("*.tif"))[-1])
    elif path.suffix == ".raw":
        with path.open("rb") as stream:
            stream.seek(-4 * last.size, 2)
            actual = np.fromfile(stream, "<f4", count=last.size).reshape(last.shape)
    else:
        with tifffile.TiffFile(path) as t:
            actual = t.pages[-1].asarray()
    np.testing.assert_array_equal([actual[0, 0], actual[1, 17], actual[-1, -1]], expected)


def run(mode: str, folder: str, name: str, axis: str) -> None:
    base = Path(folder)
    recon = mapped_result(base / "source", 512)
    path = base / name
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if mode == "write":
        tj.export(path, recon, axis=axis, compress=True if name == "lzw.tif" else None)
    else:
        check_last(path, recon, axis)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak - before <= 512 * 1024, (
        f"{mode} {name} {axis}: RSS rose by {(peak - before) / 1024:.1f} MiB"
    )


@pytest.mark.slow
def test_two_gibibytes_stay_bounded_across_formats_and_stack_axes(tmp_path):
    import shutil

    rng = np.random.default_rng(42)
    with (tmp_path / "source").open("wb") as stream:
        for _ in range(512):
            rng.random((1024, 1024), np.float32).tofile(stream)
    for axis in ("z", "x"):
        for name in ("volume.tif", "slices", "volume.raw", "lzw.tif"):
            for mode in ("write", "read"):
                subprocess.run(
                    [sys.executable, __file__, mode, str(tmp_path), name, axis], check=True
                )
            path = tmp_path / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            (tmp_path / "volume.raw.json").unlink(missing_ok=True)
    (tmp_path / "source").unlink()


@pytest.mark.slow
def test_bigtiff_offsets_cross_four_gibibytes(tmp_path):
    from tools.check_big_tiff import big_stacks

    big_stacks(tmp_path)


if __name__ == "__main__":
    run(*sys.argv[1:])
