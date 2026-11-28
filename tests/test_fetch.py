"""Storage-order planes and the two wire encodings, checked with independent decoders."""

from io import BytesIO
import json
import subprocess

import h5py
import imagecodecs
import msgspec
import numpy as np
from PIL import Image
import pytest

from tests.worker_process import command
from tomojax.cli import envelope as e
from tomojax.wire.arrays import Hdf5, Npy

pytestmark = pytest.mark.surface


def fetch(request):
    with subprocess.Popen(command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
        assert json.loads(worker.stdout.readline())["type"] == "ready"
        worker.stdin.write(msgspec.json.encode({"id": "frame", "command": request}) + b"\n")
        worker.stdin.flush()
        payload = bytearray()
        while True:
            row = json.loads(worker.stdout.readline())
            if row["type"] == "executor":
                continue
            if row["type"] == "block":
                payload.extend(worker.stdout.read(row["bytes"]))
                worker.stdin.write(msgspec.json.encode({"credit": row["bytes"]}) + b"\n")
                worker.stdin.flush()
                continue
            assert row["type"] == "result", row
            worker.stdin.close()
            assert worker.wait(timeout=10) == 0
            return msgspec.convert(row["result"], type=e.Pixels), payload


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


@pytest.mark.parametrize("encoding", ["float64", "uint16"])
def test_strided_volume(tmp_path, encoding):
    values = np.linspace(-2, 7, 11 * 13 * 17).reshape(11, 13, 17)
    values[1, 2, 0] = 1 + 2**-25
    path = tmp_path / "volume.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("signal", data=values, chunks=(2, 3, 2), compression="gzip")
    frame, payload = fetch(
        e.Fetch(
            storage=Hdf5(str(path), "signal"),
            shape=values.shape,
            stride=(3, 4, 3),
            roi=((1, 10), (2, 12), (0, 17)),
            encoding=encoding,
        )
    )
    expected = values[1:10:3, 2:12:4, ::3]
    dtype = "<f8" if encoding == "float64" else "<u2"
    planes = np.frombuffer(imagecodecs.zstd_decode(payload), dtype="u1").reshape(
        np.dtype(dtype).itemsize, -1
    )
    actual = planes.T.copy().view(dtype).reshape(frame.shape)
    if encoding == "float64":
        np.testing.assert_array_equal(actual, expected)
    else:
        low, high = frame.range
        step = (high - low) / 65535
        np.testing.assert_allclose(low + actual * step, expected, atol=step / 2 + 1e-15, rtol=0)
