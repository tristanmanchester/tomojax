"""Drive the worker's filesystem and transfer protocol without a numerical executor."""

from contextlib import contextmanager
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
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
    with h5py.File(path, "w", libver="latest", userblock_size=1024) as f:
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


def test_download_rejects_growth_after_its_first_block(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(bytes(3 * 65536))
    with worker(tmp_path) as client:
        client.write({"window": 65536})
        client.send("get", "download", path=str(path))
        while True:
            row = json.loads(client.process.stdout.readline())
            if row["type"] == "block":
                break
        first = client.process.stdout.read(row["bytes"])
        assert len(first) == 65536
        with path.open("ab") as out:
            out.write(b"new")
        client.write({"credit": len(first)})
        assert "grew" in client.terminal("get")["error"]["message"]


@pytest.mark.parametrize("interrupt", ["cancel", "kill"])
def test_copy_interrupted_after_progress_never_publishes(tmp_path, interrupt):
    source = tmp_path / "source"
    with source.open("wb") as f:
        f.truncate(64 << 20)
    destination = tmp_path / "copy"
    with subprocess.Popen(
        command("--results", str(tmp_path), "--python", "/bin/false"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        client = Client(process)
        client.send(
            "copy",
            "copy",
            sources=[{"name": "data", "path": str(source)}],
            destination=str(destination),
        )
        while client.read()["type"] != "progress":
            pass
        if interrupt == "kill":
            process.kill()
            process.wait(10)
        else:
            client.send("stop", "control", target="copy", action="cancel")
            assert client.terminal("copy")["error"]["kind"] == "canceled"
            assert client.terminal("stop")["result"]["type"] == "ack"
            client.process.stdin.close()
            assert process.wait(10) == 0
    assert not destination.exists()


def check_files(client, path, label):
    client.send(
        f"get-{label}",
        "fetch",
        encoding="float32",
        shape=[16],
        storage={"type": "npy", "path": str(path)},
    )
    assert client.terminal(f"get-{label}")["result"]["shape"] == [16]
    client.send(f"list-{label}", "list", path=str(path))
    assert client.terminal(f"list-{label}")["result"]["entries"][0]["name"] == path.name
    client.send(
        f"copy-{label}",
        "copy",
        destination=str(path.parent / f"out-{label}"),
        sources=[{"name": path.name, "path": str(path)}],
    )
    assert client.terminal(f"copy-{label}")["result"]["bytes"] == path.stat().st_size


def test_session_root_and_failed_start_retry_leave_files_available(tmp_path):
    launcher = tmp_path / "launch"
    launcher.write_text(
        "#!/bin/sh\n"
        'echo $$ >> "$0.pids"\n'
        'echo \'{"type":"ready","protocol":1,"root":"/invisible/tomojax/session"}\'\n'
        "exec sleep 90\n"
    )
    launcher.chmod(0o700)
    signal_path = tmp_path / "signal.npy"
    np.save(signal_path, np.arange(16))
    with worker(tmp_path, "--session", str(launcher)) as client:
        for generation in (1, 2):
            while True:
                row = client.read()
                if row.get("state") == "gone" and row["generation"] == generation:
                    break
            assert "session root is not visible" in row["detail"]
            check_files(client, signal_path, f"gone-{generation}")
        pids = [int(pid) for pid in launcher.with_suffix(".pids").read_text().splitlines()]
    assert all(not os.path.exists(f"/proc/{pid}") for pid in pids)


def test_queued_launcher_serves_files_and_is_reaped_on_eof(tmp_path):
    launcher = tmp_path / "launch"
    launcher.write_text(
        '#!/bin/sh\necho $$ > "$0.pid"\necho "waiting for resources" >&2\nexec sleep 90\n'
    )
    launcher.chmod(0o700)
    path = tmp_path / "signal.npy"
    np.save(path, np.arange(16, dtype="<f4"))
    with worker(tmp_path, "--session", str(launcher)) as client:
        while client.read().get("detail") != "waiting for resources":
            pass
        client.send("status", "status")
        assert client.terminal("status")["result"]["executor"]["state"] == "starting"
        check_files(client, path, "starting")
        pid = int(launcher.with_suffix(".pid").read_text())
    assert not os.path.exists(f"/proc/{pid}")


def test_file_requests_run_while_an_executor_request_is_waiting(tmp_path):
    launcher = tmp_path / "executor"
    launcher.write_text(
        "#!/bin/sh\n"
        'mkdir "$TMPDIR/root"\n'
        """printf '{"type":"ready","protocol":1,"root":"%s/root"}\\n' "$TMPDIR"\n"""
        'read -r request\necho "received" > "$0.received"\ncat >/dev/null\n'
    )
    launcher.chmod(0o700)
    path = tmp_path / "signal.npy"
    np.save(path, np.arange(16))
    with worker(tmp_path, "--session", str(launcher)) as client:
        while client.read().get("state") != "ready":
            pass
        client.send("run", "inspect", path=str(path))
        deadline = time.monotonic() + 5
        while not launcher.with_suffix(".received").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        check_files(client, path, "busy")


@pytest.mark.parametrize("phase", ["starting", "idle", "running", "shutdown"])
def test_direct_executor_dies_with_worker(tmp_path, phase):
    # The direct interpreter deliberately ignores TERM, including after stdin closes.
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, tempfile\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "def mark(name): Path(sys.argv[0]).with_suffix('.' + name).write_text(str(os.getpid()))\n"
        "mark('starting')\n"
        + ("signal.pause()\n" if phase == "starting" else "")
        + 'print(json.dumps({"type":"ready","protocol":1,"root":tempfile.mkdtemp()}), flush=True)\n'
        "for line in sys.stdin: mark('running')\n"
        "mark('shutdown')\n"
        "signal.pause()\n"
    )
    executable.chmod(0o700)
    with subprocess.Popen(
        command("--python", str(executable), "--results", str(tmp_path)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as process:
        try:
            client = Client(process)
            if phase != "starting":
                while client.read().get("state") != "ready":
                    pass
            if phase == "running":
                client.send("work", "inspect", path="/data")
            if phase == "shutdown":
                client.send("q", "shutdown")
            marker = executable.with_suffix("." + ("starting" if phase == "idle" else phase))
            deadline = time.monotonic() + 5
            while not (marker.exists() and marker.read_text()):
                assert time.monotonic() < deadline, phase
                time.sleep(0.01)
            pid = int(marker.read_text())
            pidfd = os.pidfd_open(pid)
            try:
                poller = select.poll()
                poller.register(pidfd, select.POLLIN)
                process.kill()
                assert process.wait(5) == -signal.SIGKILL
                assert poller.poll(3000), f"{phase} executor {pid} survived worker death"
            finally:
                if not poller.poll(0):
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                os.close(pidfd)
        finally:
            process.kill()
