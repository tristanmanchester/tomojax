"""tj.align fits at a bounded working scale; the scan it returns keeps the user's grid."""

from pathlib import Path

import numpy as np
import pytest

import tomojax as tj

# A detector wide enough that the fit bins it (to at most 320 pixels a side).
NU, NV, VIEWS = 336, 4, 16
ANGLES = np.linspace(0, 360, VIEWS, endpoint=False)
GRID = tj.Grid(NU, NU, NV, 1, 1, 1)
GEOMETRIES = {
    "parallel": lambda: tj.ParallelGeometry(GRID, tj.Detector(NU, NV, 1, 1), ANGLES),
    "cone": lambda: tj.ConeGeometry(
        GRID, tj.Detector(NU, NV, 1.5, 1.5), ANGLES, tj.ConeBeam(4 * NU, 6 * NU)
    ),
    "laminography": lambda: tj.LaminographyGeometry(
        GRID, tj.Detector(NU, NV, 1, 1), ANGLES, tilt=30
    ),
}


def _scan(kind: str) -> tj.Scan:
    geometry = GEOMETRIES[kind]()
    z, y, x = np.mgrid[:NV, :NU, :NU].astype(np.float32)
    center = (NU - 1) / 2
    volume = np.exp(-((x - center - 20) ** 2 + (y - center) ** 2) / 5e3).astype(np.float32)
    return tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)


def _shape(grid: tj.Grid) -> tuple[int, int, int]:
    return grid.nz, grid.ny, grid.nx


@pytest.mark.parametrize("kind", GEOMETRIES)
def test_reconstructing_an_aligned_scan_keeps_the_scan_grid(kind):
    scan = _scan(kind)
    result = tj.align(scan)
    assert result.grid != scan.grid  # the fit worked at a binned scale
    assert np.shape(result.image) == _shape(result.grid)
    assert result.scan.grid == scan.grid
    assert np.shape(tj.reconstruct(result.scan).volume) == _shape(scan.grid)


def test_the_align_docstring_example_reconstructs_at_the_scan_grid():
    scan = _scan("parallel")
    result = tj.align(scan, fit=("translations", "roll"))
    recon = tj.reconstruct(result.scan)
    assert np.shape(recon.volume) == _shape(scan.grid)


def test_a_saved_alignment_reloads_on_the_scan_grid(tmp_path: Path):
    scan = _scan("parallel")
    result = tj.align(scan)
    tj.save(tmp_path / "aligned.nxs", result)
    loaded = tj.load(tmp_path / "aligned.nxs")
    assert loaded.grid == scan.grid
    np.testing.assert_array_equal(loaded.poses, result.poses)
    assert np.shape(tj.reconstruct(loaded).volume) == _shape(scan.grid)
    # The saved alignment image comes back with the grid it is on.
    image = tj.load_reconstruction(tmp_path / "aligned.nxs")
    assert image.grid == result.grid
    np.testing.assert_array_equal(image.volume, result.image)
