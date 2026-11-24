"""Use real stdio for both MCP eras, including concurrent subscriptions and tasks."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import sys
from threading import Thread
import time

import h5py
import msgspec
import pytest

import tomojax as tj
from tomojax.cli.mcp import tools
from tomojax.cli.mcp.tools import Open

pytestmark = pytest.mark.surface
TASKS = "io.modelcontextprotocol/tasks"


class Client:
    def __init__(self, process, *, legacy):
        self.process = process
        self.legacy = legacy
        self.messages = Queue()
        self.inbox = []
        self.serial = 0

        def read_pipe():
            for line in process.stdout:
                self.messages.put(json.loads(line))
            self.messages.put(None)

        self.reader = Thread(target=read_pipe, daemon=True)
        self.reader.start()

    def send(self, method, params, *, tasks=False, notify=False):
        self.serial += 1
        if not self.legacy:
            params = {
                **params,
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientInfo": {"name": "pytest", "version": "1"},
                    "io.modelcontextprotocol/clientCapabilities": {"extensions": {TASKS: {}}}
                    if tasks
                    else {},
                },
            }
        row = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notify:
            row["id"] = self.serial
        self.process.stdin.write(json.dumps(row) + "\n")
        self.process.stdin.flush()
        return self.serial

    def receive(self, key, value):
        while True:
            for i, row in enumerate(self.inbox):
                if row.get(key) == value:
                    return self.inbox.pop(i)
            row = self.messages.get(timeout=15)
            assert row is not None, "server closed stdout"
            self.inbox.append(row)

    def rpc(self, method, params, *, tasks=False):
        return self.receive("id", self.send(method, params, tasks=tasks))

    def call(self, name, *, tasks=False, **args):
        row = self.rpc("tools/call", {"name": name, "arguments": args}, tasks=tasks)
        assert "error" not in row, row
        return row["result"]

    def task(self, action, task_id, **params):
        return self.rpc("tasks/" + action, {"taskId": task_id, **params}, tasks=True)["result"]

    def terminal(self, task_id):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            row = self.task("get", task_id)
            if row["status"] != "working":
                return row
            time.sleep(0.01)
        pytest.fail("Task did not settle")


@contextmanager
def connection(directory, *, legacy=False, overlay=False, exit_code=0):
    env = {
        **os.environ,
        "XDG_CACHE_HOME": str(directory / "cache"),
        "XDG_STATE_HOME": str(directory / "state"),
        "CLAUDE_PROJECT_DIR": str(directory),
    }
    cmd = [sys.executable, "-m", "tomojax.cli.main", "mcp"]
    if overlay:
        code = """
import sys
from tomojax.cli import mcp
mcp.__path__.insert(0, sys.argv.pop())
from tomojax.cli.mcp import tasks
tasks.TTL = 2
sys.exit(mcp.main())
"""
        cmd = [sys.executable, "-c", code, str(directory)]
    with subprocess.Popen(cmd, env=env, stdin=-1, stdout=-1, text=True) as proc:
        client = Client(proc, legacy=legacy)
        if legacy:
            assert "result" in client.rpc(
                "initialize",
                dict(
                    protocolVersion="2025-11-25",
                    capabilities={},
                    clientInfo={"name": "test", "version": "1"},
                ),
            )
            client.send("notifications/initialized", {}, notify=True)
        try:
            yield client
        finally:
            proc.stdin.close()
            try:
                assert proc.wait(timeout=10) == exit_code
            finally:
                if proc.poll() is None:
                    proc.kill()


# A copy of the tool package gains handlers during the test. Every request,
# including calls still running during a registry swap, uses the SDK dispatcher.
PROBE = """
from tomojax.cli.mcp.supervisor import activity
from tomojax.cli.mcp.tools import registry as previous_registry
class Probe(Record):
    mode: str = "success"

def registry(core):
    records = previous_registry(core)
    async def probe(args):
        base = core.supervisor.base
        if args.mode == "wait":
            (base / "entered").touch()
            while not (base / "finish").exists():
                if activity.get() and activity.get().requested:
                    return result({"kind":"cancelled"},error=True)
                await asyncio.sleep(.005)
        if args.mode == "error":
            raise ValueError("bad specimen")
        if args.mode == "loss":
            import os, signal
            g = await core.supervisor.worker()
            os.kill(g.process.pid,signal.SIGSTOP)
            _, job = g.submit({"type":"status"})
            os.kill(g.process.pid,signal.SIGKILL)
            await job.result
        return result({"generation": GENERATION})
    record = binding(Probe,probe,{})
    for number in range(GENERATION+1):
        records.append({**record,"name":"probe"+str(number),"long":True})
    return records
"""


def install_overlay(tmp_path):
    target = tmp_path / "tools"
    shutil.copytree(
        Path(tools.__file__).parent, target, ignore=shutil.ignore_patterns("__pycache__")
    )
    package = target / "__init__.py"
    initial = package.read_text() + PROBE

    def version(n):
        package.write_text(initial + f"\nGENERATION={n}\n")

    version(0)
    return version


@pytest.mark.parametrize("legacy", [True, False])
def test_adapter_schema_validation_and_capability_scope(tmp_path, legacy):
    install_overlay(tmp_path)
    with connection(tmp_path, legacy=legacy, overlay=True) as c:
        listing = c.rpc("tools/list", {})["result"]["tools"]
        opened = next(t for t in listing if t["name"] == "open")
        expected = msgspec.json.schema(Open)["$defs"]["Open"]
        assert (
            opened["inputSchema"]["properties"]["detector"]["anyOf"]
            == expected["properties"]["detector"]["anyOf"]
        )
        assert opened["inputSchema"]["type"] == opened["outputSchema"]["type"] == "object"
        probe = next(t for t in listing if t["name"] == "probe0")

        class Probe(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
            mode: str = "success"

        assert probe["inputSchema"] == msgspec.json.schema(Probe)["$defs"]["Probe"]
        invalid = c.call("open", path="unused", detector={"nu": "bad", "nv": 1, "du": 1, "dv": 1})
        assert invalid["isError"]
        assert "$.detector.nu" in invalid["content"][0]["text"]
        assert c.call("probe0")["structuredContent"] == {"generation": 0}
        if legacy:
            assert c.call("probe0", tasks=True)["structuredContent"] == {"generation": 0}
        else:
            task = c.call("probe0", tasks=True)
            assert task["resultType"] == "task"
            assert c.terminal(task["taskId"])["result"]["structuredContent"] == {"generation": 0}
            assert c.call("probe0")["resultType"] == "complete"


def test_open_inspect_reload_subscription_and_worker_loss(tmp_path):
    install = install_overlay(tmp_path)
    tj.save(tmp_path / "scan.nxs", tj.simulate("sphere", size=8, n_views=12).scan)
    with connection(tmp_path, overlay=True) as c:
        listened = c.send("subscriptions/listen", {"notifications": {"toolsListChanged": True}})
        c.receive("method", "notifications/subscriptions/acknowledged")
        report = c.call("inspect", path="scan.nxs", preview=True)
        assert report["structuredContent"]["valid"]
        assert report["content"][1]["mimeType"] == "image/png"
        opened = c.call("open", path="scan.nxs")["structuredContent"]
        assert opened["shape"] == [12, 8, 8]
        ref = opened["handle"]
        pending = c.send("tools/call", {"name": "probe0", "arguments": {"mode": "wait"}})
        end = time.monotonic() + 10
        while not (tmp_path / "entered").exists():
            assert time.monotonic() < end
            assert c.process.poll() is None
            time.sleep(0.01)
        for n in (1, 2):
            install(n)
            assert (
                "probe" + str(n) in c.call("control", action="reload")["structuredContent"]["tools"]
            )
            event = c.receive("method", "notifications/tools/list_changed")
            assert event["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == listened
            assert c.call("probe" + str(n))["structuredContent"]["generation"] == n
            assert c.call("status")["structuredContent"]["handles"][ref]["bytes"] > 0
        (tmp_path / "finish").touch()
        assert c.receive("id", pending)["result"]["structuredContent"]["generation"] == 0
        assert (
            "filter"
            in c.call("describe", operation="reconstruct", method="fbp")["structuredContent"][
                "parameters"
            ]
        )
        failure = c.call("probe2", tasks=True, mode="loss")
        failed = c.terminal(failure["taskId"])
        assert failed["status"] == "failed"
        assert ref in failed["error"]["message"]
        state = c.call("status")["structuredContent"]
        assert ref in state["recovery"]["lost"]
        assert not state["handles"]
        assert "lost" in c.call("control", action="close", target=ref)["content"][0]["text"]
        new = c.call("open", path="scan.nxs")["structuredContent"]["handle"]
        assert new != ref
        assert new in c.call("control", action="restart")["structuredContent"]["lost"]
    assert not list((tmp_path / "cache/tomojax/mcp").glob("worker-*"))


def test_task_methods_cancel_race_and_expiry(tmp_path):
    install_overlay(tmp_path)
    with connection(tmp_path, overlay=True) as c:
        task = c.call("probe0", tasks=True, mode="wait")
        tid = task["taskId"]
        working = c.task("get", tid)
        assert working["resultType"] == "complete"
        assert working["status"] == "working"
        assert c.rpc("tasks/get", {"taskId": tid})["error"]["code"] == -32021
        assert c.rpc("tasks/get", {"taskId": "unknown"}, tasks=True)["error"]["code"] == -32602
        assert (
            c.task("update", tid, inputResponses={"stale": {"action": "accept", "content": {}}})[
                "resultType"
            ]
            == "complete"
        )
        assert c.task("cancel", tid)["resultType"] == "complete"
        assert c.terminal(tid)["status"] == "cancelled"
        failure = c.call("probe0", tasks=True, mode="error")["taskId"]
        assert c.terminal(failure)["result"]["isError"]
        assert c.task("cancel", failure)["resultType"] == "complete"
        assert c.terminal(failure)["status"] == "completed"
        time.sleep(task["ttlMs"] / 1000)
        assert c.rpc("tasks/get", {"taskId": tid}, tasks=True)["error"]["code"] == -32602


def test_directory_inspection_and_unrecognised_hdf5(tmp_path):
    with h5py.File(tmp_path / "unknown.h5", "w") as file:
        file["a/b::c"] = [1, 2, 3]
    with connection(tmp_path) as c:
        tree = c.call("inspect", path="unknown.h5")["structuredContent"]
        assert tree["datasets"]["/a/b::c"]["shape"] == [3]
        root = tmp_path / "files"
        root.mkdir()
        tj.save(root / "scan.nxs", tj.simulate("sphere", size=8, n_views=12).scan)
        (root / "text").write_text("unrelated")
        scanned = c.call("inspect", path=str(root))["structuredContent"]
        assert list(scanned["datasets"]) == [str(root / "scan.nxs")]


def test_sigterm_reaps_a_stopped_worker_with_stdin_open(tmp_path):
    import signal

    install_overlay(tmp_path)
    with connection(tmp_path, overlay=True, exit_code=-signal.SIGTERM) as c:
        assert c.call("status")["structuredContent"]["devices"]
        children = Path(f"/proc/{c.process.pid}/task/{c.process.pid}/children")
        (worker,) = map(int, children.read_text().split())
        os.kill(worker, signal.SIGSTOP)
        c.process.terminate()
        assert c.process.wait(timeout=10) == -signal.SIGTERM
        assert not Path(f"/proc/{worker}").exists()
        assert not list((tmp_path / "cache/tomojax/mcp").glob("worker-*"))
