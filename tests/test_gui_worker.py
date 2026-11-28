"""A JSON Lines client using committed schemas and real process pipes."""

from array import array
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import socket
import subprocess
import sys
import termios

from jsonschema import Draft202012Validator
import numpy as np
import pytest
from referencing import Registry, Resource

from .worker_process import command

DOCS = Path(__file__).resolve().parents[1] / "docs/protocol"
pytestmark = pytest.mark.surface


@pytest.fixture(scope="module")
def schema():
    docs = [json.loads(p.read_text()) for p in DOCS.glob("*.schema.json")]
    registry = Registry().with_resources((d["$id"], Resource.from_contents(d)) for d in docs)
    return Draft202012Validator(
        next(d for d in docs if d["$id"].endswith("/worker.schema.json")), registry=registry
    )


class Client:
    def __init__(self, proc, port, schema):
        self.proc = proc
        self.port = port
        self.schema = schema
        self.buffer = b""
        self.blocks = {}
        self.rows = []
        self.terminal = {}

    def start(self):
        assert self.read()["type"] == "ready"
        while (row := self.read())["state"] == "starting":
            pass
        assert row["state"] == "ready", row
        return row["detail"]

    def send(self, rid, tag, *, channel=None, **fields):
        request = {"id": rid, "command": {"type": tag, **fields}, "channel": channel}
        self.schema.validate(request)
        self.proc.stdin.write(json.dumps(request).encode() + b"\n")
        self.proc.stdin.flush()

    def read(self):
        while b"\n" not in self.buffer:
            assert select.select([self.proc.stdout], [], [], 8)[0], "worker output stalled"
            data = os.read(self.proc.stdout.fileno(), 65536)
            assert data, "unexpected worker EOF"
            self.buffer += data
        data, self.buffer = self.buffer.split(b"\n", 1)
        row = json.loads(data)
        self.schema.validate(row)
        if row["type"] == "block":
            size = row["bytes"]
            assert 0 < size <= 65536
            while len(self.buffer) < size:
                assert select.select([self.proc.stdout], [], [], 8)[0], "pixel block stalled"
                part = os.read(self.proc.stdout.fileno(), size - len(self.buffer))
                assert part
                self.buffer += part
            self.blocks.setdefault(row["request"], bytearray()).extend(self.buffer[:size])
            self.buffer = self.buffer[size:]
        self.rows.append(row)
        if row["type"] in ("error", "result", "progress"):
            assert row["request"] not in self.terminal, "record after terminal"
            if row["type"] != "progress":
                self.terminal[row["request"]] = row
        return row

    def end(self, rid):
        while rid not in self.terminal:
            self.read()
        return self.terminal[rid]

    def mark(self, signal):
        assert self.port.recv(1) == signal

    def proceed(self):
        self.port.sendall(b"C")

    def shutdown(self):
        self.send("q", "shutdown")
        self.end("q")
        assert self.proc.wait(8) == 0
        assert self.buffer + self.proc.stdout.read() == b""


@pytest.fixture
def client(schema, tmp_path):
    parent, child = socket.socketpair()
    with parent, child:
        parent.settimeout(8)
        with subprocess.Popen(
            command(
                "--session",
                shlex.join(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("worker_fixture.py")),
                        str(child.fileno()),
                    ]
                ),
            ),
            env={**os.environ, "TMPDIR": str(tmp_path)},
            pass_fds=(child.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        ) as proc:
            child.close()
            try:
                c = Client(proc, parent, schema)
                c.start()
                yield c
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(8)


def test_cli_handshake_and_terminal_pairing(schema):
    requests = [
        dict(id="d", command=dict(type="devices")),
        dict(id="q", command=dict(type="shutdown")),
    ]
    raw = subprocess.run(
        [sys.executable, "-m", "tomojax.cli.main", "serve"],
        input=b'{"id":"\xff","command":{"type":"status"}}\n'
        + b"".join(json.dumps(r).encode() + b"\n" for r in requests),
        capture_output=True,
        timeout=20,
        check=True,
    )
    rows = [json.loads(line) for line in raw.stdout.splitlines()]
    for row in rows:
        schema.validate(row)
    ready, bad, devices, bye = rows
    assert ready["protocol"] == 1
    assert ready["progress"] == 2
    assert bad["type"] == "protocol_error"
    assert devices["request"] == "d"
    assert devices["result"]["devices"] == ready["devices"]
    assert bye == dict(type="result", request="q", channel=None, result=dict(type="ack"))


def pause(c):
    c.send("old", "inspect", channel="view", path="wait")
    c.mark(b"G")
    c.send("p", "control", action="pause", target="old")
    c.end("p")
    c.send("pending", "status")
    assert c.end("pending")["result"]["requests"]["old"] == "pause-requested"
    c.proceed()
    while c.read().get("record", {}).get("event") != "paused":
        pass


def test_pause_pins_inputs_and_resume_releases_queue(client):
    c = client
    c.send("open", "open", path="volume", kind="reconstruction")
    source = {
        "type": "member",
        "handle": c.end("open")["result"]["value"]["handle"],
        "member": "volume",
    }
    pause(c)
    c.send("plane", "plane", source=source, axis="x", index=1)
    c.send("plane", "status")
    assert c.read()["type"] == "protocol_error"
    c.send("snapshot", "status")
    report = c.end("snapshot")["result"]
    assert report["requests"] == {"old": "paused", "plane": "queued"}
    assert report["handles"] == {"open": 48}
    assert report["allocators"] == [
        dict(time=4.25, device="cpu:0", scope="host", used=1234, peak=None, available=None)
    ]
    c.send("close", "close", handle=source["handle"])
    c.end("close")
    c.send("late", "plane", source=source, axis="z", index=0, channel="view")
    assert c.read()["type"] == "protocol_error"
    c.send("resume", "control", target="old", action="resume")
    c.mark(b"R")
    ref = c.end("plane")["result"]["value"]["plane"]["storage"]
    assert c.end("old")["type"] == "result"
    assert [r["record"]["event"] for r in c.rows if r["type"] == "progress"] == [
        "started",
        "paused",
        "resumed",
        "finished",
    ]
    np.testing.assert_array_equal(np.load(ref["path"]), [[1, 5, 9], [13, 17, 21]])
    c.send("release", "release", artifact=ref["artifact"])
    c.end("release")
    assert not Path(ref["path"]).exists()
    c.shutdown()


@pytest.mark.parametrize("finish", ["shutdown", "eof", "cancel"])
def test_stop_wakes_paused(client, finish):
    c = client
    pause(c)
    if finish == "eof":
        c.proc.stdin.close()
    elif finish == "shutdown":
        c.send("bye", "shutdown")
    else:
        c.send("stop", "control", target="old", action="cancel")
        c.end("stop")
    terminal = c.end("old")
    assert terminal["error"]["kind"] == "canceled"
    assert c.rows[-2]["record"]["event"] == "finished"
    if finish == "shutdown":
        c.end("bye")
    elif finish == "cancel":
        c.shutdown()
    assert c.proc.wait(8) == 0


@pytest.mark.skipif(sys.platform != "linux", reason="sets the pipe capacity with F_SETPIPE_SZ")
@pytest.mark.parametrize("drain", [True, False])
def test_full_pipe_supersedes_paused_worker_before_client_reads(client, drain):
    c = client
    pause(c)
    fd = c.proc.stdout.fileno()
    capacity = fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 4096)
    c.send("fill", "describe")
    assert select.select([fd], [], [], 8)[0]
    assert (
        fcntl.ioctl(fd, termios.FIONREAD, array("i", [0]), False)
        == array("i", [capacity]).tobytes()
    )
    c.send("new", "inspect", path="notify", channel="view")
    c.mark(b"N")  # Executor reached revision 2 with a still-full stdout pipe.
    c.proc.stdin.close()
    if drain:
        assert c.end("old")["error"]["kind"] == "superseded"
        assert c.end("new")["channel"] == {"name": "view", "revision": 2}
        assert len(c.end("fill")["result"]["parameters"]["large"]) == 32768
    assert c.proc.wait(8) == (0 if drain else 1)


@pytest.mark.parametrize("outcome", ["supersede", "cancel", "commit", "missing", "no-force"])
def test_durable_group_at_commit(client, tmp_path, outcome):
    c = client
    raw = tmp_path / ("partial.raw" if outcome == "missing" else "data.raw")
    files = [raw, raw.with_suffix(".json"), raw.with_suffix(".manifest")]
    for file in files:
        file.write_text("original")
    c.send("open", "open", path="source", kind="reconstruction")
    ref = {"type": "member", "handle": c.end("open")["result"]["value"]["handle"], "member": "self"}
    c.send(
        "export",
        "run",
        channel="save",
        operation=dict(type="export", source=ref, path=str(raw), force=outcome != "no-force"),
    )
    c.mark(b"G")  # Success has been constructed, and all intended writes have finished.
    if outcome == "supersede":
        c.send("new", "status", channel="save")
        assert c.end("new")["channel"]["revision"] == 2
    elif outcome == "cancel":
        c.send("stop", "control", target="export", action="cancel")
        c.end("stop")
    assert [f.read_text() for f in files] == ["original"] * 3
    c.proceed()
    result = c.end("export")
    if outcome == "commit":
        assert result["result"]["value"]["paths"] == list(map(str, files))
    else:
        assert result["error"]["kind"] == {"cancel": "canceled", "supersede": "superseded"}.get(
            outcome, "input"
        )
    c.shutdown()
    assert set(tmp_path.iterdir()) == set(files)
    for file in files:
        assert file.read_text() == (
            "replacement " + file.suffix if outcome == "commit" else "original"
        )


@pytest.mark.parametrize(
    ("case", "kind"),
    [
        ("invalid", "invalid_parameter"),
        ("jax-oom", "out_of_memory"),
        ("jax-unknown", "internal"),
        pytest.param(
            "alloc",
            "out_of_memory",
            marks=pytest.mark.skipif(
                sys.platform != "linux",
                reason="forces a real allocation failure with Linux RLIMIT_AS and /proc",
            ),
        ),
    ],
)
def test_executor_failure_is_a_typed_terminal(client, case, kind):
    client.send("failure", "inspect", path=case)
    error = client.end("failure")["error"]
    assert error["kind"] == kind
    if case == "invalid":
        assert error["path"] == "config.filter"
    client.shutdown()


def test_preview_release_keeps_queued_artifact_lease(client, tmp_path):
    c = client
    mine = tmp_path / "caller.npy"
    mine.write_bytes(b"caller data")
    c.send("preview", "inspect", path="preview")
    c.mark(b"G")
    row = c.read()
    ref = row["record"]["arrays"]["frame"]["storage"]
    path = Path(ref["path"])
    assert int(np.load(path)) == 99
    operation = {
        "type": "backproject",
        "source": {"type": "path", "path": "scan"},
        "projections": ref,
    }
    c.send("pinned", "run", operation=operation)
    c.send("forged", "run", operation={**operation, "projections": {**ref, "path": str(mine)}})
    assert c.read()["type"] == "protocol_error"
    c.send("path", "release", artifact=str(mine))
    assert c.read()["type"] == "protocol_error"
    c.send("drop", "release", artifact=ref["artifact"])
    c.end("drop")
    assert path.exists()
    c.send("late", "run", operation=operation)
    assert c.read()["type"] == "protocol_error"
    c.proceed()
    pinned = c.end("pinned")["result"]
    assert pinned["value"]["report"]["sum"] == 99
    assert pinned["manifest"]["inputs"] == [
        {"operation": "inspect", "parameters": {"path": "preview"}, "inputs": []}
    ]
    assert not path.exists()
    c.shutdown()
    assert mine.read_bytes() == b"caller data"


def test_queue_supersedes_before_execution_and_shutdown_removes_published_files(client):
    c = client
    c.send("preview", "inspect", path="preview")
    c.mark(b"G")
    ref = c.read()["record"]["arrays"]["frame"]["storage"]
    c.send("obsolete", "inspect", channel="view", path="wait")
    c.send("next", "inspect", channel="view", path="notify")
    result = c.end("obsolete")
    assert result["channel"] == {"name": "view", "revision": 1}
    assert result["error"]["kind"] == "superseded"
    c.proceed()
    c.mark(b"N")
    assert c.end("next")["channel"]["revision"] == 2
    c.shutdown()
    assert not Path(ref["path"]).parent.exists()


def test_dispatches_every_other_execution_tag_through_adapter(client):
    c = client
    source = {"type": "path", "path": "source"}
    reconstruct = {"type": "reconstruct", "source": source}
    inputs = {
        "validate": {"operation": reconstruct},
        "plan": {"operation": reconstruct},
        "slice": {"source": source, "axis": "z", "index": 2},
        "center_sweep": {"source": source, "axis": "z", "index": 2, "centers": [2, -1, 2]},
        "correction_preview": {"source": source, "rows": [1, 3]},
        "region_preview": {"source": source},
    }
    for tag, fields in inputs.items():
        c.send(tag, tag, **fields)
    for tag in inputs:
        result = c.end(tag)["result"]
        assert result["manifest"]["operation"].lower() == tag.replace("_", "")
    c.shutdown()


def test_replays_fixed_session_transcript(client):
    c = client
    lines = (DOCS / "transcripts/session.jsonl").read_text().splitlines()
    ready = c.rows[-1]["detail"].copy()
    assert Path(ready.pop("root")).is_dir()
    assert len(ready.pop("schema")) == 64
    assert ready == json.loads(lines[0])
    for line in lines[1:]:
        expected = json.loads(line)
        if "id" in expected:
            c.proc.stdin.write(line.encode() + b"\n")
        else:
            actual = c.read()
            if "record" in expected:
                assert actual["record"]["time"] >= 0
                actual = {
                    **actual,
                    "record": {**actual["record"], "time": expected["record"]["time"]},
                }
            if expected["type"] == "error":
                expected["error"].setdefault("path", None)
                expected["error"].setdefault("checkpoint", None)
            assert actual == expected
            if actual.get("record", {}).get("event") == "started":
                c.mark(b"G")
            if actual.get("request") == "stop":
                c.proceed()
    assert c.proc.wait(8) == 0


@pytest.mark.skipif(sys.platform != "linux", reason="sets the pipe capacity with F_SETPIPE_SZ")
def test_only_adjacent_advances_coalesce_behind_blocked_writer(client):
    c = client
    fd = c.proc.stdout.fileno()
    fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 4096)
    c.send("fill", "describe")
    assert select.select([fd], [], [], 8)[0]
    c.send("ticks", "inspect", path="ticks")
    c.mark(b"D")
    c.end("ticks")
    events = [row["record"] for row in c.rows if row["type"] == "progress"]
    assert [event["event"] for event in events] == ["advanced", "note"]
    assert events[0]["done"] == 99
    c.shutdown()


def test_broken_stdout_wakes_paused_executor_and_cleans_session(client, tmp_path):
    pause(client)
    client.proc.stdout.close()
    assert client.proc.wait(8) == 1
    assert not list(tmp_path.iterdir())


def test_oversized_line_is_drained_through_one_newline(client):
    c = client
    c.proc.stdin.write(b" " * (1024 * 1024 + 1) + b'{"id":"ignore","command":{"type":"status"}}\n')
    c.proc.stdin.flush()
    assert c.read()["type"] == "protocol_error"
    c.shutdown()


@pytest.fixture
def worker(schema):
    with subprocess.Popen(
        command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    ) as proc:
        try:
            client = Client(proc, None, schema)
            client.start()
            yield client
            client.shutdown()
        finally:
            if proc.poll() is None:
                proc.kill()


def test_simulation_transcript_artifact_lineage(worker):
    lines = [
        json.loads(line)
        for line in (DOCS / "transcripts/simulation.jsonl").read_text().splitlines()
    ]
    outputs = {}
    for request, expected in zip(lines[::2], lines[1::2], strict=True):
        operation = request["command"]["operation"]
        if operation["type"] == "backproject":
            operation["projections"] = outputs["p"]["value"]["projections"]["storage"]
        worker.send(request["id"], "run", operation=operation)
        reply = worker.end(request["id"])
        assert reply["type"] == "result", reply
        output = reply["result"]
        assert output["value"]["type"] == expected["result"]["value"]["type"]
        assert output["manifest"]["operation"] == expected["result"]["manifest"]["operation"]
        outputs[request["id"]] = output
    sim, project, back = (outputs[rid]["manifest"] for rid in ("s", "p", "b"))
    assert sim["inputs"] == []
    assert project["inputs"] == [sim, sim]
    assert back["inputs"] == [sim, project]


def test_raw_slabs_transcript_matches_worker(worker, tmp_path):
    from tests._helpers import write_raw_nxtomo

    write_raw_nxtomo(tmp_path / "raw.nxs")
    lines = (DOCS / "transcripts/raw-slabs.jsonl").read_text().replace("/data/", f"{tmp_path}/")
    rows = [json.loads(line) for line in lines.splitlines()]
    for request, expected in zip(rows[::2], rows[1::2], strict=True):
        command = request["command"]
        worker.send(request["id"], command.pop("type"), **command)
        actual = worker.end(request["id"])
        assert actual["type"] == "result", actual
        if request["id"] == "v":
            saved = expected["result"]["info"]["summary"]
            saved["truncation"] = pytest.approx(saved["truncation"])
        assert actual == expected


def test_invalid_command_echoes_readable_id(client):
    invalid, expected = map(
        json.loads, (DOCS / "transcripts/invalid-command.jsonl").read_text().splitlines()
    )
    assert not client.schema.is_valid(invalid)
    for raw_id in (invalid["id"], "", 31, None):
        request = {**invalid, "id": raw_id}
        client.proc.stdin.write(json.dumps(request).encode() + b"\n")
        client.proc.stdin.flush()
        error = client.read()
        if raw_id == invalid["id"]:
            assert error == expected
        assert error["type"] == "protocol_error"
        assert error["request"] == (raw_id if isinstance(raw_id, str) else None)
    client.send("form-17", "status")
    assert client.end("form-17")["type"] == "result"


def test_adapter_import_cannot_read_or_write_protocol():
    program = """
import os, sys
class Probe:
    def find_spec(self, fullname, *args):
        if fullname == 'tomojax.cli._adapters':
            assert os.read(0, 1) == b''
            os.write(1, b'library output')
sys.meta_path.insert(0, Probe())
from tomojax.cli.main import main
sys.exit(main(['serve']))
"""
    raw = subprocess.run(
        [sys.executable, "-c", program],
        input=b'{"id":"q","command":{"type":"shutdown"}}\n',
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert b"library output" in raw.stderr
    assert [json.loads(line)["type"] for line in raw.stdout.splitlines()] == ["ready", "result"]


def test_fetch_overtakes_paused_work_and_retires_queued_bytes(client, tmp_path):
    rng = np.random.default_rng(19)
    path = tmp_path / "plane.npy"
    data = rng.standard_normal((1024, 1024), dtype=np.float32)
    np.save(path, data)
    fields = dict(
        storage=dict(type="npy", path=str(path)), shape=list(data.shape), encoding="float32"
    )
    pause(client)
    client.send("large", "fetch", channel="pane", **fields)
    while client.read()["type"] != "block":
        pass
    client.send("small", "fetch", channel="pane", roi=[[0, 4], [0, 4]], **fields)
    assert client.end("large")["error"]["kind"] == "superseded"
    old_bytes = len(client.blocks["large"])
    assert 0 < old_bytes <= 262144
    client.proc.stdin.write(json.dumps({"credit": old_bytes}).encode() + b"\n")
    result = client.end("small")
    assert result["channel"] == {"name": "pane", "revision": 2}
    assert result["result"]["shape"] == [4, 4]
    assert len(client.blocks["small"]) == result["result"]["bytes"]
    assert len(client.blocks["large"]) == old_bytes
    client.send("cancel", "control", target="old", action="cancel")
    assert client.end("old")["error"]["kind"] == "canceled"
    client.end("cancel")
    client.shutdown()


def test_resize_with_unacknowledged_blocks(client, tmp_path):
    block = 65536
    data = np.random.default_rng(79).standard_normal((768, 768), dtype=np.float32)
    path = tmp_path / "large.npy"
    np.save(path, data)
    client.send(
        "large",
        "fetch",
        storage=dict(type="npy", path=str(path)),
        shape=list(data.shape),
        encoding="float32",
    )

    def control(**message):
        client.proc.stdin.write(json.dumps(message).encode() + b"\n")
        client.proc.stdin.flush()

    def outstanding(size):
        while len(client.blocks.get("large", b"")) < size:
            client.read()

    outstanding(4 * block)
    control(window=block)
    control(credit=3 * block)
    client.send("barrier", "status")
    client.end("barrier")
    assert len(client.blocks["large"]) == 4 * block
    control(credit=block + 1)
    assert "exceeds outstanding" in client.read()["message"]
    control(credit=block)
    outstanding(5 * block)
    control(window=3 * block)
    outstanding(7 * block)
    client.shutdown()


def test_fetch_admission_spans_files_and_executor(client, tmp_path):
    import imagecodecs

    path = tmp_path / "values.npy"
    np.save(path, np.arange(4).reshape(2, 2))
    fields = {"storage": {"type": "npy", "path": str(path)}, "shape": [2, 2], "encoding": "float32"}
    client.send("running", "inspect", channel="execution", path="wait")
    client.mark(b"G")
    assert client.read()["record"]["event"] == "started"
    client.send("running", "fetch", **fields)
    assert client.read()["type"] == "protocol_error"
    client.send("other", "fetch", channel="execution", **fields)
    assert client.read()["type"] == "protocol_error"
    client.send("f", "fetch", channel="pixels", **fields)
    assert client.end("f")["type"] == "result"
    client.send("reverse", "inspect", channel="pixels", path="notify")
    assert client.read()["type"] == "protocol_error"
    client.proceed()
    assert client.end("running")["type"] == "result"
    client.send("open", "open", path="volume", kind="reconstruction")
    handle = client.end("open")["result"]["value"]["handle"]
    client.send(
        "plane",
        "plane",
        source={"type": "member", "handle": handle, "member": "volume"},
        axis="x",
        index=1,
    )
    ref = client.end("plane")["result"]["value"]["plane"]["storage"]
    fields = {"storage": ref, "shape": [2, 3], "encoding": "float32"}
    client.send("wrong", "fetch", **{**fields, "storage": {**ref, "artifact": "wrong"}})
    assert client.read()["type"] == "protocol_error"
    # Occupy all four file readers before admitting the leased read.
    pipes = [tmp_path / f"gate-{i}" for i in range(4)]
    for i, gate in enumerate(pipes):
        os.mkfifo(gate)
        client.send(
            f"gate-{i}",
            "fetch",
            storage={"type": "npy", "path": str(gate)},
            shape=[1],
            encoding="float32",
        )
    client.send("leased", "fetch", **fields)
    client.send("release", "release", artifact=ref["artifact"])
    client.send("close", "close", handle=handle)
    client.send("heartbeat", "status")
    client.end("heartbeat")
    assert Path(ref["path"]).exists() and "release" not in client.terminal
    client.send("retired", "fetch", **fields)
    assert client.read()["type"] == "protocol_error"
    for gate in pipes:
        with gate.open("wb"):
            pass
    assert client.end("leased")["type"] == "result"
    data = imagecodecs.zstd_decode(client.blocks["leased"])
    np.testing.assert_array_equal(
        np.frombuffer(data, dtype="u1").reshape(4, -1).T.copy().view("<f4").ravel(),
        [1, 5, 9, 13, 17, 21],
    )
    client.end("release")
    client.end("close")
    assert not Path(ref["path"]).exists()
    client.shutdown()


def test_executor_generation_ends_waiters_once(client):
    client.send("bad", "inspect", path="executor-exit")
    client.send("waiting", "inspect", path="wait")
    assert client.end("bad")["error"]["kind"] == "input"
    assert client.end("waiting")["error"]["kind"] == "input"
    while (row := client.read()).get("state") != "ready":
        pass
    assert row["generation"] == 2
    client.send("again", "status")
    assert client.end("again")["result"]["handles"] == {}
    client.shutdown()
