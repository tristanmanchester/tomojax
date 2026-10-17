"""The public walnut example keeps TIFF orientation and acquisition order explicit."""

import importlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def test_walnut_reader_matches_the_authors_view_and_image_order(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    reader = importlib.import_module("_walnut")
    for name, value in (("ROWS", 2), ("COLUMNS", 4), ("VIEWS", 4)):
        monkeypatch.setattr(reader, name, value)
    folder = tmp_path / "Projections/tubeV2"
    folder.mkdir(parents=True)
    vectors = np.arange(48).reshape(4, 12)
    np.savetxt(folder / "scan_geom_original.geom", vectors)
    for name, value in (("di000000", 10), ("io000000", 100), ("io000001", 120)):
        iio.imwrite(folder / f"{name}.tif", np.full((4, 2), value, np.uint16))
    values = np.arange(8).reshape(4, 2)
    for view in range(1, 5):
        iio.imwrite(folder / f"scan_{view:06d}.tif", (values + view * 10).astype(np.uint16))
    data, actual = reader.load_orbit(tmp_path, 2, 2)
    np.testing.assert_array_equal(actual, vectors[::2])
    expected = np.stack([np.flipud(values + view * 10).T for view in (4, 2)], axis=1)
    np.testing.assert_allclose(data, -np.log((expected - 10) / 100), rtol=1e-6)
