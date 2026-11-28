"""Storage-order planes and the two wire encodings, checked with independent decoders."""

from io import BytesIO

import h5py
import imagecodecs
import numpy as np
from PIL import Image
import pytest

from tomojax.cli import envelope as e
from tomojax.cli._pixels import fetch, gray_levels  # check-public-imports: allow-private
from tomojax.wire.arrays import Hdf5, Npy

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("stored", ["npy", "contiguous", "gzip"])
@pytest.mark.parametrize("line_detector", [False, True])
def test_lossless_plane(tmp_path, stored, line_detector):
    rng = np.random.default_rng(211)
    shape = (15, 1 if line_detector else 7, 23)
    samples = rng.standard_normal(shape, dtype=np.float32)
    y = 0 if line_detector else 4
    samples[1:4, y, 2] = [np.nan, -np.inf, np.inf]
    samples[4, y, 2] = -0.0
    samples.view("u4")[5, y, 2] = 0x7FC00019
    disk = samples[:, 0] if line_detector else samples
    path = tmp_path / ("data.npy" if stored == "npy" else "data.h5")
    if stored == "npy":
        np.save(path, disk)
        ref = Npy(str(path))
    else:
        with h5py.File(path, "w") as f:
            options = (
                {"chunks": (3, *disk.shape[1:]), "compression": "gzip"} if stored == "gzip" else {}
            )
            f.create_dataset("signal", data=disk, **options)
        ref = Hdf5(str(path), "signal")
    request = e.Fetch(
        storage=ref,
        shape=shape,
        roi=((1, 15), (y, y + 1), (2, 20)),
        stride=(1, 1, 3),
        encoding="float32",
    )
    frame, payload = fetch(request)
    expected = samples[1:15, y : y + 1, 2:20:3]
    byte_planes = np.frombuffer(imagecodecs.zstd_decode(payload), dtype="u1").reshape(4, -1)
    assert byte_planes.T.copy().tobytes() == expected.astype("<f4").tobytes()
    assert frame.shape == expected.shape and frame.bytes == len(payload)


@pytest.mark.parametrize("case", ["ramp", "step", "constant", "zero-window", "noise"])
def test_jpeg(tmp_path, case):
    source = np.tile(np.arange(255, dtype="f4"), (97, 1))
    limits = (0, 254)
    if case == "step":
        source = 254 * (source >= 53).astype("f4")
    elif case == "constant":
        source[:] = 4
        limits = None
    elif case == "zero-window":
        limits = (37, 37)
    elif case == "noise":
        source = np.random.default_rng(77).standard_normal((257, 255)).astype("f4")
        limits = (-2, 2)
    path = tmp_path / "input.npy"
    np.save(path, source)
    frame, payload = fetch(
        e.Fetch(storage=Npy(str(path)), shape=source.shape, encoding="jpeg", window=limits)
    )
    with Image.open(BytesIO(payload)) as image:
        decoded = np.array(image, dtype=float)
    assert frame.range == (limits or (4, 4))
    if case in ("constant", "zero-window"):
        assert np.unique(decoded).size == 1
    elif case == "noise":
        expected = np.clip((source + 2) * 63.75, 0, 255)
        assert abs(decoded.mean() - expected.mean()) <= 2
    else:
        assert np.max(np.abs(decoded - np.round(source * (255 / 254)))) <= 8
        if case == "step":
            assert np.max(np.abs(np.argmax(decoded >= 128, axis=1) - 53)) <= 1


def test_levels_before_lossy_encoding():
    values = np.array([[3, 4, 5, np.nan, -np.inf, np.inf]])
    levels, limits = gray_levels(values, None)
    assert limits == (3, 5)
    np.testing.assert_array_equal(levels, [[0, 128, 255, 0, 0, 0]])
