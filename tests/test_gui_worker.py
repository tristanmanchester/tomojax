"""A JSON Lines client using committed schemas and real process pipes."""

from array import array
import fcntl
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import termios

from jsonschema import Draft202012Validator
import numpy as np
import pytest
from referencing import Registry, Resource

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
        self.rows = []
        self.terminal = {}

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
            [
                sys.executable,
                str(Path(__file__).with_name("worker_fixture.py")),
                str(child.fileno()),
            ],
            env={**os.environ, "TMPDIR": str(tmp_path)},
            pass_fds=(child.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        ) as proc:
            child.close()
            try:
                c = Client(proc, parent, schema)
                assert c.read()["type"] == "ready"
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
    assert c.rows[0] == json.loads(lines[0])
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


def test_simulation_transcript_artifact_lineage(schema):
    lines = [
        json.loads(line)
        for line in (DOCS / "transcripts/simulation.jsonl").read_text().splitlines()
    ]
    with subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    ) as proc:
        try:
            client = Client(proc, None, schema)
            assert client.read()["type"] == "ready"
            outputs = {}
            for request, expected in zip(lines[::2], lines[1::2], strict=True):
                operation = request["command"]["operation"]
                if operation["type"] == "backproject":
                    operation["projections"] = outputs["p"]["value"]["projections"]["storage"]
                client.send(request["id"], "run", operation=operation)
                reply = client.end(request["id"])
                assert reply["type"] == "result", reply
                output = reply["result"]
                assert output["value"]["type"] == expected["result"]["value"]["type"]
                assert (
                    output["manifest"]["operation"] == expected["result"]["manifest"]["operation"]
                )
                outputs[request["id"]] = output
            sim, project, back = (outputs[rid]["manifest"] for rid in ("s", "p", "b"))
            assert sim["inputs"] == []
            assert project["inputs"] == [sim, sim]
            assert back["inputs"] == [sim, project]
            client.shutdown()
        finally:
            if proc.poll() is None:
                proc.kill()


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
