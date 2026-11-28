"""Drive the worker's filesystem and transfer protocol without a numerical executor."""

from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import time

import h5py
import numpy as np
import pytest

from .worker_process import command

pytestmark = pytest.mark.surface


class Client:
    def __init__(self, process):
        self.process = process
        self.rows = []
        self.payload = {}

    def write(self, row, data=b""):
        self.process.stdin.write(json.dumps(row).encode() + b"\n" + data)
        self.process.stdin.flush()

    def send(self, rid, kind, **fields):
        self.write({"id": rid, "command": {"type": kind, **fields}})

    def read(self):
        row = json.loads(self.process.stdout.readline())
        if row["type"] == "block":
            n = row["bytes"]
            assert 0 < n <= 65536
            self.payload.setdefault(row["request"], bytearray()).extend(self.process.stdout.read(n))
            self.write({"credit": n})
        self.rows.append(row)
        return row

    def terminal(self, rid):
        while True:
            for row in self.rows:
                if row.get("request") == rid and row["type"] in {
                    "protocol_error",
                    "error",
                    "result",
                }:
                    return row
            self.read()


@contextmanager
def worker(path, *options):
    with subprocess.Popen(
        command("--results", str(path), "--python", "/nonexistent/tomojax", *options),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as proc:
        c = Client(proc)
        assert c.read()["type"] == "ready"
        try:
            yield c
        finally:
            proc.stdin.close()
            assert proc.wait(10) == 0


def test_copy_directory_and_documents(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "scan.npy", np.arange(24).reshape(2, 3, 4))
    dest = tmp_path / "result"
    sources = [{"path": str(source), "name": "input"}]
    with worker(tmp_path) as c:
        c.send(
            "one",
            "copy",
            sources=sources,
            documents={"meta/scan.json": {"grid": [4, 3, 2]}},
            destination=str(dest),
        )
        assert c.terminal("one")["result"]["bytes"] == (source / "scan.npy").stat().st_size
        c.send("dup", "copy", sources=sources, destination=str(dest))
        assert c.terminal("dup")["type"] == "error"
        c.send("nested", "copy", sources=sources, destination=str(source / "nested"))
        assert c.terminal("nested")["type"] == "error"
        source.rename(tmp_path / "moved")
        c.send("ls", "list", path=str(dest))
        assert [entry["name"] for entry in c.terminal("ls")["result"]["entries"]] == [
            "input",
            "meta",
        ]
    np.testing.assert_array_equal(np.load(dest / "input/scan.npy"), np.arange(24).reshape(2, 3, 4))
    assert json.loads((dest / "meta/scan.json").read_text()) == {"grid": [4, 3, 2]}


@pytest.mark.parametrize("layout", ["link", "vds", "external"])
def test_copy_refuses_non_self_contained_hdf5(tmp_path, layout):
    path = tmp_path / "master.nxs"
    with h5py.File(path, "w", libver="latest") as f:
        if layout == "link":
            f["outside"] = h5py.ExternalLink("slave.h5", "/data")
        elif layout == "vds":
            mapping = h5py.VirtualLayout((2,), dtype="i4")
            mapping[:] = h5py.VirtualSource("slave.h5", "data", shape=(2,))
            f.create_virtual_dataset("data", mapping)
        else:
            f.create_dataset("data", shape=(2,), dtype="i4", external=[("external.bin", 0, 8)])
    with worker(tmp_path) as c:
        c.send(
            "copy",
            "copy",
            sources=[{"path": str(path), "name": path.name}],
            destination=str(tmp_path / "out"),
        )
        assert "reads data from other files" in c.terminal("copy")["error"]["message"]
    assert not (tmp_path / "out").exists()


def test_two_uploads_one_claim_and_download(tmp_path):
    data = os.urandom(600000)
    digest = hashlib.sha256(data).hexdigest()
    path = tmp_path / "sample.bin"
    old = tmp_path / ".tomojax-part-old"
    mine = tmp_path / "sample.part"
    for p in (old, mine):
        p.write_bytes(b"old")
        os.utime(p, (time.time() - 7200,) * 2)
    with worker(tmp_path) as c:
        for rid in ("left", "right"):
            c.send(rid, "upload", path=str(path), bytes=len(data), sha256=digest)
        for start in range(0, len(data), 65536):
            block = data[start : start + 65536]
            for rid in ("left", "right"):
                c.write({"type": "block", "request": rid, "bytes": len(block)}, block)
        assert {c.terminal(rid)["type"] for rid in ("left", "right")} == {"error", "result"}
        c.write({"window": 65536})
        c.send("get", "download", path=str(path))
        assert c.terminal("get")["result"] == {
            "type": "download",
            "bytes": len(data),
            "sha256": digest,
        }
        assert c.payload["get"] == data
    assert not old.exists() and mine.read_bytes() == b"old" and path.read_bytes() == data


def test_transfer_limits_digest_and_interrupted_upload(tmp_path):
    big = tmp_path / "big"
    with big.open("wb") as f:
        f.truncate(100 * 1024**2 + 1)
    path = tmp_path / "dest"
    digest = hashlib.sha256(b"12").hexdigest()
    with worker(tmp_path) as c:
        c.send("big", "download", path=str(big))
        assert c.terminal("big")["type"] == "error" and "big" not in c.payload
        c.send("too_big", "upload", path=str(path), bytes=big.stat().st_size, sha256=digest)
        assert c.terminal("too_big")["type"] == "error"
        for rid, block in (("over", b"123"), ("wrong", b"ab")):
            c.send(rid, "upload", path=str(path), bytes=2, sha256=digest)
            c.write({"type": "block", "request": rid, "bytes": len(block)}, block)
            assert c.terminal(rid)["type"] == "error" and not path.exists()
        c.send("cut", "upload", path=str(path), bytes=2, sha256=digest)
        c.write({"type": "block", "request": "cut", "bytes": 1}, b"1")
    assert not path.exists() and not list(tmp_path.glob(".tomojax-part-*"))
