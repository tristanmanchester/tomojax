"""The K11 gallery planes follow their stated physical grid and share one gray range."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("h5py")

GALLERY = Path(__file__).parents[1] / "examples" / "gallery"


def _encoded_reconstruct(calls):
    """A stand-in for tj.reconstruct whose voxel (z, y, x) holds 10000 z + 100 y + x."""

    def reconstruct(scan, method, *, grid, **options):
        calls.append((method, grid, options))
        z, y, x = np.meshgrid(
            np.arange(grid.nz), np.arange(grid.ny), np.arange(grid.nx), indexing="ij"
        )
        return SimpleNamespace(volume=(10000 * z + 100 * y + x).astype(np.float32))

    return reconstruct


def test_plane_orients_each_cut_and_centers_its_grid(monkeypatch):
    monkeypatch.syspath_prepend(str(GALLERY))
    import _k11

    calls = []
    monkeypatch.setattr(_k11.tj, "reconstruct", _encoded_reconstruct(calls))
    center = (-345.0, -60.0, 12.0)

    xy = _k11.plane(None, center, (5, 4, 1))
    xz = _k11.plane(None, center, (5, 1, 3))
    yz = _k11.plane(None, center, (1, 4, 3), "fista", tv_weight=0.1)

    assert xy.shape == (4, 5) and xy[2, 3] == 100 * 2 + 3  # rows along y, columns along x
    assert xz.shape == (3, 5) and xz[2, 3] == 10000 * 2 + 3  # rows along z, columns along x
    assert yz.shape == (3, 4) and yz[2, 3] == 10000 * 2 + 100 * 3  # rows along z, columns y
    for _, grid, _ in calls:
        assert (grid.vx, grid.vy, grid.vz) == (_k11.PITCH_UM,) * 3
        np.testing.assert_allclose(grid.vol_center, center)
    assert calls[2][0] == "fista" and calls[2][2] == {"tv_weight": 0.1}


@pytest.mark.parametrize("module", ["k11_ruby_sphere", "k11_glass_layer"])
def test_figure_draws_200_um_bars_on_one_gray_range(monkeypatch, tmp_path, module):
    monkeypatch.syspath_prepend(str(GALLERY))
    gallery = __import__(module)
    figures = []
    monkeypatch.setattr(gallery, "save", lambda fig, path: figures.append(fig))
    rng = np.random.default_rng(0)
    cuts = [
        ("xy", rng.random((60, 60))),
        ("xz", 2 + rng.random((20, 90))),
        ("yz", rng.random((20, 90))),
    ]

    gallery.figure(cuts, tmp_path / "out.png")

    fig = figures[0]
    limits = {ax.images[0].get_clim() for ax in fig.axes}
    assert len(limits) == 1  # one gray range across the planes
    low, high = limits.pop()
    values = np.concatenate([cut.ravel() for _, cut in cuts])
    assert (low, high) == tuple(np.percentile(values, (1, 99.5)))
    for ax, (title, _) in zip(fig.axes, cuts, strict=True):
        assert ax.get_title() == title
        bar = ax.lines[0].get_xdata()
        assert bar[1] - bar[0] == pytest.approx(200 / gallery.PITCH_UM)
        assert any(text.get_text() == "200 µm" for text in ax.texts)
    plt.close(fig)
