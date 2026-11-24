"""``--watch-dir`` images and the renderer they share with the other previews."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from tomojax import progress
from tomojax.cli import _images
from tomojax.cli._watch import LOCK, STATUS, WatchDirectory
from tomojax.cli.main import main
from tomojax.wire.arrays import Axis, Uniform

pytestmark = pytest.mark.surface


def _axis(name, frame, pitch, unit="mm"):
    return Axis(name, frame, unit, Uniform(0.0, pitch))


def test_planes_get_square_physical_pixels_and_sinograms_keep_their_samples():
    plane = (_axis("z", "volume", 2.0), _axis("x", "volume", 1.0))
    assert _images.display_shape((300, 400), plane, 2048) == (600, 400)
    sinogram = (_axis("view", "acquisition", 1.0, "degree"), _axis("u", "detector", 0.5))
    assert _images.display_shape((300, 400), sinogram, 2048) == (300, 400)
    mixed_units = (_axis("z", "volume", 2.0), _axis("x", "volume", 1.0, "um"))
    assert _images.display_shape((300, 400), mixed_units, 2048) == (300, 400)
    # Capped by a whole factor before anything is allocated; small ones enlarged.
    assert _images.display_shape((5000, 100), sinogram, 2048) == (1666, 33)
    assert _images.display_shape((64, 32), plane, 2048) == (256, 64)


def test_shared_limits_and_non_finite_values():
    bright, dim = np.full((4, 4), 10.0), np.full((4, 4), 1.0)
    low, high = _images.limits(bright, dim)
    assert low < 1.5 and high > 9.5
    image = _images.to_uint8(np.array([[np.nan, 1.0, 10.0]]), (1.0, 10.0))
    assert image.tolist() == [[0, 0, 255]]


def test_a_failed_write_leaves_the_old_image_and_no_temporary(tmp_path, monkeypatch):
    target = tmp_path / "slice_z.png"
    target.write_bytes(b"old")
    picture = _images.labelled(np.zeros((4, 4), np.uint8), "x")

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(picture), "save", broken)
    with pytest.raises(OSError, match="disk full"):
        _images.save_png(picture, target)
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["slice_z.png"]


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    path = tmp_path_factory.mktemp("scan") / "scan.nxs"
    assert main(["simulate", "-o", str(path), "--size", "16", "--progress", "none"]) == 0
    return path


def _status(directory: Path) -> dict:
    return json.loads((directory / STATUS).read_text())


@pytest.mark.parametrize("method", ["cgls", "fista"])
def test_a_reconstruction_rewrites_its_planes_and_ends_done(tmp_path, scan, method):
    watch = tmp_path / "watch"
    argv = ["reconstruct", str(scan), "-o", str(tmp_path / "r.nxs"), "--method", method]
    assert main([*argv, "--iterations", "6", "--watch-dir", str(watch)]) == 0
    names = sorted(p.name for p in watch.iterdir())
    assert names == sorted([LOCK, STATUS, "slice_x.png", "slice_y.png", "slice_z.png"])
    status = _status(watch)
    assert status["status"] == "done" and status["checkpoints_written"] >= 1
    assert "checkpoint" in status["last_checkpoint"]


def test_the_watched_volume_is_bit_identical(tmp_path, scan):
    import tomojax as tj

    volumes = []
    for watched in (False, True):
        out = tmp_path / f"r{watched}.nxs"
        argv = ["reconstruct", str(scan), "-o", str(out), "--method", "cgls", "--iterations", "5"]
        assert main(argv + (["--watch-dir", str(tmp_path / "w")] if watched else [])) == 0
        volumes.append(np.asarray(tj.load_reconstruction(out).volume))
    np.testing.assert_array_equal(volumes[0], volumes[1])


def test_an_alignment_writes_its_working_image_and_poses(tmp_path, scan):
    watch = tmp_path / "watch"
    argv = ["align", str(scan), "-o", str(tmp_path / "a.nxs"), "--watch-dir", str(watch)]
    assert main([*argv, "--iterations", "2", "--image-iterations", "2"]) == 0
    assert {"slice_z.png", "poses.png"} <= {p.name for p in watch.iterdir()}
    assert _status(watch)["status"] == "done"


def test_no_update_means_no_image_and_a_status_that_says_so(tmp_path, scan):
    watch = tmp_path / "watch"
    argv = ["reconstruct", str(scan), "-o", str(tmp_path / "r.nxs"), "--method", "fbp"]
    assert main([*argv, "--watch-dir", str(watch)]) == 0
    assert not list(watch.glob("*.png"))
    assert _status(watch)["status"] == "done, no preview"


def test_reuse_removes_the_earlier_images(tmp_path, scan):
    watch = tmp_path / "watch"
    base = ["reconstruct", str(scan), "-o", str(tmp_path / "r.nxs"), "--watch-dir", str(watch)]
    assert main([*base, "--method", "cgls", "--iterations", "3"]) == 0
    assert list(watch.glob("*.png"))
    with pytest.raises(SystemExit):  # earlier images need --force
        main([*base, "--method", "fbp"])
    assert main([*base, "--method", "fbp", "--force"]) == 0
    assert not list(watch.glob("*.png"))


def _sample():
    plane = (_axis("y", "volume", 1.0), _axis("x", "volume", 1.0))
    return progress.Sample(np.ones((8, 8), np.float32), plane, "1/mm", "attenuation")


def test_cancelling_after_an_image_leaves_it_labelled_and_the_run_cancelled(tmp_path):
    from tomojax.cli._progress import run_command

    watch = WatchDirectory(tmp_path / "watch", "reconstruct", working=False)

    def operation():
        watch.admit(force=False, related=())
        with progress.previews(), progress.task("solve", "cgls", 3, "iterations"):
            progress.current_task.get().advance(1, {})
            progress.preview(lambda: {"z": _sample()}, final=True)
            raise progress.Cancelled

    assert run_command("reconstruct", "none", operation, watch=watch) == 130
    assert (tmp_path / "watch" / "slice_z.png").exists()
    status = _status(tmp_path / "watch")
    assert status["status"] == "cancelled"
    assert status["last_checkpoint"] == "cgls · checkpoint 1"


def _admit(directory, *, force=False, related=()):
    WatchDirectory(directory, "reconstruct", working=False).admit(force=force, related=related)


def test_a_directory_with_other_files_is_refused_and_left_alone(tmp_path):
    directory = tmp_path / "watch"
    directory.mkdir()
    for name, text in ((STATUS, "{}"), ("slice_z.png", "old"), ("notes.txt", "mine")):
        (directory / name).write_text(text)
    with pytest.raises(ValueError, match=r"holds other files \(notes.txt\)"):
        _admit(directory, force=True)
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [STATUS, "slice_z.png", "notes.txt", LOCK]
    )


def test_unrecorded_images_links_and_contained_inputs_are_refused(tmp_path):
    unrecorded = tmp_path / "a"
    unrecorded.mkdir()
    (unrecorded / "slice_z.png").write_text("someone's")
    with pytest.raises(ValueError, match="no TomoJAX run recorded"):
        _admit(unrecorded, force=True)
    linked = tmp_path / "b"
    linked.mkdir()
    (linked / STATUS).write_text("{}")
    (linked / "slice_z.png").symlink_to(tmp_path / "elsewhere.png")
    with pytest.raises(ValueError, match="not a regular file"):
        _admit(linked, force=True)
    holding = tmp_path / "c"
    with pytest.raises(ValueError, match="contains the command's file"):
        _admit(holding, related=[holding / "scan.nxs"])


def test_a_live_writer_keeps_out_a_forced_second_one(tmp_path):
    directory = tmp_path / "watch"
    directory.mkdir()
    handle = os.open(directory / LOCK, os.O_RDWR | os.O_CREAT)
    fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        with pytest.raises(FileExistsError, match="another run"):
            _admit(directory, force=True)
    finally:
        os.close(handle)
    _admit(directory)  # a dead writer's lock needs no recovery


@pytest.mark.parametrize(
    "left", [[STATUS, ".slice_z.png.tmp", "slice_y.png"], [".tomojax-watch.json.tmp"]]
)
def test_a_killed_run_leaves_a_directory_the_next_one_accepts(tmp_path, left):
    directory = tmp_path / "watch"
    directory.mkdir()
    for name in left:
        (directory / name).write_text("partial")
    _admit(directory, force=True)
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        {LOCK, STATUS, *left} & {LOCK, STATUS}
    )


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
