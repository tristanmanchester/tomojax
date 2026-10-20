"""The gallery scale and slice coordinates follow its stated physical grid."""

from pathlib import Path

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")


def test_ruby_gallery_preserves_region_and_labels_scale(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples" / "gallery"))
    import k11_ruby_sphere as gallery

    initial = gallery.sampling_grid(4)
    assert (initial.nx, initial.ny, initial.nz) == (160, 160, 135)
    assert initial.vx == 8.64
    grid = gallery.tj.Grid(552, 552, 192, 8.64, 8.64, 8.64)
    volume = np.broadcast_to(np.arange(grid.nx)[:, None, None], (grid.nx, grid.ny, grid.nz))
    figures = []
    monkeypatch.setattr(gallery, "save", lambda fig, path: figures.append(fig))
    gallery.slice_figure(volume, grid, tmp_path / "ruby.png")
    fig = figures[0]
    assert any("2.16 µm pixels · 4x4 binning · 8.64 µm voxels" in t.get_text() for t in fig.texts)
    yz = fig.axes[2].images[0].get_array()
    np.testing.assert_array_equal(yz, yz[0, 0])
    x = gallery.grid_volume_origin(grid)[0] + float(yz[0, 0]) * grid.vx
    assert abs(x - (-332.64)) <= grid.vx / 2 + 1e-9
    assert f"x = {x:.1f} µm" in fig.axes[2].get_title()
    for ax in fig.axes:
        assert np.diff(ax.lines[0].get_xdata())[0] == 200
        assert any(text.get_text() == "200 µm" for text in ax.texts)
    plt.close(fig)


def test_layer_gallery_uses_world_slice_coordinate_and_enlarges_cross_sections(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples" / "gallery"))
    import k11_glass_layer as gallery

    grid = gallery.tj.Grid(384, 384, 64, 8.64, 8.64, 8.64, vol_center=(0, 0, 17.28))
    volume = np.broadcast_to(np.arange(grid.nz)[None, None, :], (grid.nx, grid.ny, grid.nz))
    figures = []
    monkeypatch.setattr(gallery, "save", lambda fig, path: figures.append(fig))
    gallery.slice_figure(volume, grid, tmp_path / "layer.png")
    fig = figures[0]
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), 31)
    assert "z = 13.0 µm" in fig.axes[0].get_title()
    for ax in fig.axes[1:]:
        extent = ax.images[0].get_extent()
        assert extent[1] - extent[0] < grid.nx * grid.vx
        assert extent[3] - extent[2] < grid.nz * grid.vz
    for ax in fig.axes:
        assert np.diff(ax.lines[0].get_xdata())[0] == 200
        assert any(text.get_text() == "200 µm" for text in ax.texts)
    plt.close(fig)
