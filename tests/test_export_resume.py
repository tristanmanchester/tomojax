"""Kill a writer at durable boundaries and count its reads on the subsequent request."""

from __future__ import annotations

from dataclasses import replace
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

import tomojax as tj


def pixels():
    return (np.arange(35 * 4 * 5).reshape(35, 4, 5) - 21).astype(np.float32)


def source(root):
    grid = tj.Grid(5, 4, 35, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(5, 4, 1, 1), [0.0, 90.0])
    tj.save(root / "source.nxs", tj.Reconstruction(pixels(), geometry, "fbp", {}))


def writer(root, name, axis, stop, change):
    from tomojax.io import (  # check-public-imports: allow-private
        _export_formats as formats,
        _export_resume as resume,
    )
    from tomojax.io._export_data import Selection  # check-public-imports: allow-private

    path = root / name
    journal = root / f".{name}.partial" / "journal.json"
    item = tj.load_reconstruction(root / "source.nxs")
    if change == "memory":
        item = replace(item, volume=item.volume.copy())
    reads = []
    original_read = Selection.read
    original_replace = Path.replace
    original_write = formats.write_at
    original_page = tifffile.imwrite

    def read(selection, start, end):
        reads.append((int(start), int(end)))
        return original_read(selection, start, end)

    def kill():
        os.kill(os.getpid(), signal.SIGKILL)

    def publish(before, after):
        target = Path(after)
        if target == journal and json.loads(before.read_bytes())["planes"] == 3:
            if stop == "before-journal":
                kill()
            result = original_replace(before, after)
            if stop == "after-journal":
                kill()
            return result
        return original_replace(before, after)

    def midway():
        return (
            stop == "mid-plane"
            and journal.is_file()
            and json.loads(journal.read_bytes())["planes"] == 2
        )

    def write(fd, block, offset):
        if midway():
            os.pwrite(fd, b"partial", offset)
            kill()
        return original_write(fd, block, offset)

    def page(filename, *args, **kwargs):
        if midway():
            Path(filename).write_bytes(b"unfinished page")
            kill()
        return original_page(filename, *args, **kwargs)

    link, rename = os.link, os.rename

    def install(real):
        def call(old, new, *args, **kwargs):
            result = real(old, new, *args, **kwargs)
            if stop == "publication" and Path(new).parent == root:
                kill()
            return result

        return call

    with (
        patch.object(Selection, "block_depth", property(lambda _: 1)),
        patch.object(Selection, "read", read),
        patch.object(resume, "INTERVAL", 0.0),
        patch.object(formats, "write_at", write),
        patch.object(tifffile, "imwrite", page),
        patch.object(Path, "replace", publish),
        patch.object(os, "link", install(link)),
        patch.object(os, "rename", install(rename)),
    ):
        receipt = tj.export(
            path, item, axis=axis, dtype="uint8", range=(1, 500) if change == "options" else None
        )
    (root / "reads.json").write_text(json.dumps(reads))
    (root / "receipt.json").write_text(json.dumps(receipt))


def run(root, name, axis, stop="none", change="none"):
    process = subprocess.run(
        [sys.executable, __file__, str(root), name, axis, stop, change],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == (0 if stop == "none" else -signal.SIGKILL), process.stderr


@pytest.mark.parametrize(
    ("name", "axis"),
    [("slices", "z"), ("sinograms", "x"), ("voxels.raw", "x"), ("volume.mhd", "z")],
)
@pytest.mark.parametrize(
    ("stop", "committed"),
    [("mid-plane", 2), ("before-journal", 2), ("after-journal", 3), ("publication", 35)],
)
def test_killed_writer_resumes_the_committed_bytes(tmp_path, name, axis, stop, committed):
    source(tmp_path)
    run(tmp_path, name, axis, stop)
    journal = tmp_path / f".{name}.partial" / "journal.json"
    assert json.loads(journal.read_bytes())["planes"] == committed
    run(tmp_path, name, axis)
    assert not journal.parent.exists()
    reads = json.loads((tmp_path / "reads.json").read_bytes())
    assert reads == [[i, i + 1] for i in range(committed, 35)]
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    coordinates = np.rint(np.linspace(0, 34, 16)).astype(int)
    values = pixels()
    low, high = np.percentile(values[coordinates], [0.1, 99.9])
    np.testing.assert_allclose(receipt["values"]["range"], [low, high])
    assert receipt["clipped"] == {
        "below": int((values < low).sum()),
        "above": int((values > high).sum()),
    }
    expected = np.rint(np.clip((values.astype(float) - low) * 255 / (high - low), 0, 255)).astype(
        np.uint8
    )
    if axis == "x":
        expected = expected.transpose(2, 0, 1)
    path = tmp_path / name
    if path.is_dir():
        output = np.stack([tifffile.imread(p) for p in sorted(path.glob("*.tif"))])
    else:
        output = np.fromfile(path.with_suffix(".raw"), np.uint8).reshape(expected.shape)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("change", ["options", "edited", "replaced", "memory"])
def test_stale_or_unidentified_source_starts_again(tmp_path, change):
    import h5py

    source(tmp_path)
    name = "volume.raw"
    run(tmp_path, name, "z", "after-journal", "memory" if change == "memory" else "none")
    if change == "edited":
        with h5py.File(tmp_path / "source.nxs", "r+") as file:
            file["entry/volume/data"][1] = 77  # outside the 16 range samples
    elif change == "replaced":
        source(tmp_path)
    run(tmp_path, name, "z", change=change)
    reads = json.loads((tmp_path / "reads.json").read_bytes())
    assert reads[-35:] == [[i, i + 1] for i in range(35)]


def test_failed_export_removes_partial_and_preserves_published_files(tmp_path):
    name = tmp_path / "volume.mhd"
    source(tmp_path)
    item = tj.load_reconstruction(tmp_path / "source.nxs")
    tj.export(name, item)
    files = [name, name.with_suffix(".raw"), name.with_suffix(".mhd.json")]
    previous = [p.read_bytes() for p in files]
    invalid = pixels()
    invalid[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tj.export(name, replace(item, volume=invalid), dtype="uint16", range=(0, 100))
    assert not (tmp_path / ".volume.mhd.partial").exists()
    assert [p.read_bytes() for p in files] == previous


def test_disk_full_reports_partial_payload_before_cleanup(tmp_path, monkeypatch):
    source(tmp_path)
    item = tj.load_reconstruction(tmp_path / "source.nxs")
    path = tmp_path / "voxels.raw"
    tj.export(path, item)
    previous = [p.read_bytes() for p in (path, path.with_suffix(".raw.json"))]

    def disk_full(fd, data, offset):
        os.write(fd, b"seven!!")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "pwrite", disk_full)
    with pytest.raises(OSError, match=r"stopped after 7 bytes; \d+ bytes free") as error:
        tj.export(path, item)
    assert error.value.errno == errno.ENOSPC
    assert not (tmp_path / ".voxels.raw.partial").exists()
    assert [p.read_bytes() for p in (path, path.with_suffix(".raw.json"))] == previous


def gated_worker(root):
    from tomojax.cli.main import main
    from tomojax.io._export_data import Selection  # check-public-imports: allow-private
    from tomojax.io._export_resume import Journal  # check-public-imports: allow-private

    save = Journal.advanced
    read = Selection.read

    def checkpoint(journal, count, below, above):
        save(journal, count, below, above)
        if count == 2 and not (root / "release").exists():
            (root / "at-boundary").touch()
            deadline = time.monotonic() + 30
            while not (root / "release").exists() and time.monotonic() < deadline:
                time.sleep(0.02)

    def measured(selection, start, stop):
        if (root / "release").exists():
            with (root / "worker-reads").open("a") as log:
                log.write(f"{start} {stop}\n")
        return read(selection, start, stop)

    with (
        patch.object(Selection, "block_depth", property(lambda _: 1)),
        patch.object(Selection, "read", measured),
        patch.object(Journal, "advanced", checkpoint),
    ):
        return main(["serve"])


def test_worker_cancel_retains_the_partial_and_resumes_from_the_same_handle(tmp_path):
    source(tmp_path)
    path = tmp_path / "worker.mhd"
    journal = tmp_path / ".worker.mhd.partial" / "journal.json"
    with subprocess.Popen(
        [sys.executable, __file__, "serve", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:

        def send(rid, command):
            process.stdin.write(json.dumps(dict(id=rid, command=command)) + "\n")
            process.stdin.flush()

        def result(rid):
            for line in process.stdout:
                row = json.loads(line)
                if row["request"] == rid and row["type"] != "progress":
                    return row
            pytest.fail("worker stopped before replying")

        assert json.loads(process.stdout.readline())["type"] == "ready"
        send("open", dict(type="open", kind="reconstruction", path=str(tmp_path / "source.nxs")))
        handle = result("open")["result"]["value"]["handle"]
        operation = dict(
            type="export",
            source=dict(type="member", member="volume", handle=handle),
            path=str(path),
        )
        try:
            send("first", dict(type="run", operation=operation))
            deadline = time.monotonic() + 30
            while not (tmp_path / "at-boundary").exists():
                assert time.monotonic() < deadline, "export did not reach its second slab"
                time.sleep(0.02)
            competitor = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tomojax.cli.main",
                    "export",
                    str(tmp_path / "source.nxs"),
                    "-o",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert competitor.returncode and "another export is writing" in competitor.stderr
            send("cancel", dict(type="control", target="first", action="cancel"))
            assert result("cancel")["type"] == "result"
            (tmp_path / "release").touch()
            error = result("first")["error"]
            assert error["kind"] == "canceled" and str(journal.parent) in error["message"]
            assert json.loads(journal.read_bytes())["planes"] == 2 and not path.exists()
            send("second", dict(type="run", operation=operation))
            assert result("second")["type"] == "result"
        finally:
            (tmp_path / "release").touch()
            send("close", {"type": "shutdown"})
            process.stdin.close()
        assert process.wait(timeout=30) == 0
    assert not journal.parent.exists()
    reads = (tmp_path / "worker-reads").read_text().splitlines()
    assert reads == [f"{i} {i + 1}" for i in range(2, 35)]
    np.testing.assert_array_equal(np.fromfile(path.with_suffix(".raw"), "<f4"), pixels().ravel())


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        sys.exit(gated_worker(Path(sys.argv[2])))
    writer(Path(sys.argv[1]), *sys.argv[2:])
