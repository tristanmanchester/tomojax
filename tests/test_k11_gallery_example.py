"""The gallery scale and slice coordinates follow its stated physical grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
