"""JSON-RPC client for conversations with an installed TomoJAX server."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Thread
import time

import pytest

from tests.worker_process import binary

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
                row = json.loads(line)
                if row.get("type") in ("result", "error", "protocol_error"):
                    row["request_terminal"] = row.get("request")
                self.messages.put(row)
            self.messages.put(None)

        self.reader = Thread(target=read_pipe, daemon=True)
        self.reader.start()

    def send(self, method, params, *, tasks=False, notify=False):
        self.serial += 1
        if not self.legacy:
            params = {
                **params,
                "_meta": {
                    **params.get("_meta", {}),
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

    def send_worker(self, name, args):
        self.serial += 1
        rid = str(self.serial)
        self.process.stdin.write(
            json.dumps({"id": rid, "command": {"type": "run", "operation": {"type": name, **args}}})
            + "\n"
        )
        self.process.stdin.flush()
        return rid

    def receive(self, key, value):
        while True:
            for i, row in enumerate(self.inbox):
                if row.get(key) == value:
                    return self.inbox.pop(i)
            row = self.messages.get(timeout=120)  # a hang, not a slow runner
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
def connection(directory, *, legacy=False, exit_code=0, executable=None, profile=None):
    scratch = directory / "cache"
    scratch.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(scratch), "CLAUDE_PROJECT_DIR": str(directory)}
    if executable is None:
        env["PATH"] = os.pathsep.join((str(Path(binary()).parent), env["PATH"]))
    cmd = [str(executable or Path(sys.executable).with_name("tomojax")), "mcp"]
    if profile is not None:
        cmd.extend(["--connection", profile])
    with subprocess.Popen(cmd, env=env, cwd=directory, stdin=-1, stdout=-1, text=True) as proc:
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
