"""Exercise TIFF directory offsets with incompressible data beyond four GiB."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import tifffile

import tomojax as tj

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def big_stacks(
    root: Path, read: Callable[[Path, list[int], np.ndarray], None] | None = None
) -> None:
    """Verify independent voxel sentinels before and beyond the classic TIFF offset limit."""
    depth = 1152
    grid = tj.Grid(1024, 1024, depth, 0.0015, 0.002, 0.003)
    geometry = tj.ParallelGeometry(grid, tj.Detector(3, 3, 1, 1), np.arange(3.0))
    # Reuse a random plane. TIFF compression starts again at each strip.
    rng = np.random.default_rng(41)
    values = rng.random((1024, 1024), np.float32)
    source = tj.Reconstruction(
        np.broadcast_to(values, grid.shape), geometry, "fbp", {}, length_unit="mm"
    )
    for compressed in (False, True):
        for suffix in (".tif", ".ome.tif") if read else (".tif",):
            path = root / ("big" + suffix)
            # Vary the sentinel across pages, so detecting an offset wrap needs only one voxel.
            with (root / "sentinels.bin").open("wb") as file:
                for z in range(depth):
                    values[2, 3] = z
                    values.tofile(file)
            pixels = np.memmap(root / "sentinels.bin", "<f4", "r", shape=grid.shape)
            try:
                tj.export(path, replace(source, volume=pixels), compress=compressed)
                with tifffile.TiffFile(path) as t:
                    assert t.is_bigtiff
                    crossing = next(i for i, p in enumerate(t.pages) if p.dataoffsets[0] >= 2**32)
                    indices = [0, crossing, depth - 1]
                    for i in indices:
                        assert t.pages[i].asarray()[2, 3] == i
                    if read:
                        read(path, indices, np.asarray(indices, np.float32))
            finally:
                del pixels
                (root / "sentinels.bin").unlink()
                path.unlink(missing_ok=True)
