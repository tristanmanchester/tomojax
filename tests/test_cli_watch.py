"""``--watch-dir`` images and the renderer they share with the other previews."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from tomojax import progress
from tomojax.cli.main import main
from tomojax.wire.arrays import Axis, Uniform

pytestmark = pytest.mark.surface


@pytest.fixture(name="watching")
def watch_module():
    """The CLI module under test (a white-box test of its internals)."""
    from tomojax.cli import _watch as watching  # check-public-imports: allow-private

    return watching


@pytest.fixture(name="images")
def images_module():
    """The CLI module under test (a white-box test of its internals)."""
    from tomojax.cli import _images as images  # check-public-imports: allow-private

    return images


def _axis(name, frame, pitch, unit="mm"):
    return Axis(name, frame, unit, Uniform(0.0, pitch))


def test_planes_get_square_physical_pixels_and_sinograms_keep_their_samples(images):
    plane = (_axis("z", "volume", 2.0), _axis("x", "volume", 1.0))
    assert images.display_shape((300, 400), plane, 2048) == (600, 400)
    sinogram = (_axis("view", "acquisition", 1.0, "degree"), _axis("u", "detector", 0.5))
    assert images.display_shape((300, 400), sinogram, 2048) == (300, 400)
    mixed_units = (_axis("z", "volume", 2.0), _axis("x", "volume", 1.0, "um"))
    assert images.display_shape((300, 400), mixed_units, 2048) == (300, 400)
    # Capped by a whole factor before anything is allocated; small ones enlarged.
    assert images.display_shape((5000, 100), sinogram, 2048) == (1666, 33)
    assert images.display_shape((64, 32), plane, 2048) == (256, 64)


def test_an_image_creates_its_folder(tmp_path, images):
    path = tmp_path / "previews" / "band.png"
    images.save_png(images.Image.new("L", (4, 4)), path)
    assert path.exists()


def test_shared_limits_and_non_finite_values(images):
    bright, dim = np.full((4, 4), 10.0), np.full((4, 4), 1.0)
    low, high = images.limits(bright, dim)
    assert low < 1.5 and high > 9.5
    image = images.to_uint8(np.array([[np.nan, 1.0, 10.0]]), (1.0, 10.0))
    assert image.tolist() == [[0, 0, 255]]


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    path = tmp_path_factory.mktemp("scan") / "scan.nxs"
    assert main(["simulate", "-o", str(path), "--size", "16", "--progress", "none"]) == 0
    return path


@pytest.mark.parametrize("method", ["cgls", "fista"])
def test_a_reconstruction_writes_its_central_planes(tmp_path, scan, method):
    watch = tmp_path / "watch"
    argv = ["reconstruct", str(scan), "-o", str(tmp_path / "r.nxs"), "--method", method]
    assert main([*argv, "--iterations", "6", "--watch-dir", str(watch)]) == 0
    names = sorted(p.name for p in watch.iterdir())
    assert names == ["slice_x.png", "slice_y.png", "slice_z.png"]


def test_the_watched_volume_is_bit_identical(tmp_path, scan):
    import tomojax as tj

    volumes = []
    for watched in (False, True):
        out = tmp_path / f"r{watched}.nxs"
        argv = ["reconstruct", str(scan), "-o", str(out), "--method", "cgls", "--iterations", "5"]
        assert main(argv + (["--watch-dir", str(tmp_path / "w")] if watched else [])) == 0
        volumes.append(np.asarray(tj.load_reconstruction(out).volume))
    np.testing.assert_array_equal(volumes[0], volumes[1])


@pytest.mark.slow
def test_an_alignment_writes_its_working_image_and_poses(tmp_path, scan):
    watch = tmp_path / "watch"
    argv = ["align", str(scan), "-o", str(tmp_path / "a.nxs"), "--watch-dir", str(watch)]
    assert main([*argv, "--iterations", "2", "--image-iterations", "2"]) == 0
    assert {"slice_z.png", "poses.png"} <= {p.name for p in watch.iterdir()}


def _sample():
    plane = (_axis("y", "volume", 1.0), _axis("x", "volume", 1.0))
    return progress.Sample(np.ones((8, 8), np.float32), plane, "1/mm", "attenuation")


def test_canceling_after_a_preview_keeps_its_image(tmp_path, watching):
    from tomojax.cli._progress import run_command  # check-public-imports: allow-private

    watch = watching.WatchDirectory(tmp_path / "watch", working=False)

    def operation():
        with progress.previews(), progress.task("solve", "cgls", 3, "iterations"):
            progress.current_task.get().advance(1, {})
            progress.preview(lambda: {"z": _sample()}, final=True)
            raise progress.Canceled

    assert run_command("reconstruct", "none", operation, watch=watch) == 130
    assert (tmp_path / "watch" / "slice_z.png").exists()


def _records(capsys):
    lines = capsys.readouterr().out.splitlines()
    return [
        (r["event"], r.get("name"), r.get("label"))
        for r in map(json.loads, lines)
        if r["event"] in ("started", "finished", "note", "preview")
    ]


def test_json_with_a_watch_directory_writes_images_and_the_same_stream(tmp_path, capsys):
    raw = tmp_path / "raw.nxs"
    from ._helpers import write_raw_nxtomo

    write_raw_nxtomo(raw)
    streams = []
    for watched in (False, True):
        out = tmp_path / ("watched" if watched else "plain") / "r.nxs"
        out.parent.mkdir()
        argv = ["reconstruct", str(raw), "-o", str(out)]
        argv += ["--method", "cgls", "--iterations", "3", "--progress", "json"]
        if watched:
            argv += ["--watch-dir", str(tmp_path / "watch")]
        assert main(argv) == 0
        streams.append(_records(capsys))
    assert streams[0] == streams[1]
    assert not any(kind == "preview" for kind, _, _ in streams[1])
    assert list((tmp_path / "watch").glob("slice_*.png"))
    assert h5py.is_hdf5(tmp_path / "watched" / "r.nxs")
