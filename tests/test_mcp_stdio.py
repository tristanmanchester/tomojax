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
ACK = {
    "resultType": "complete",
    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "tomojax", "version": tj.__version__}},
}


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
tasks.HEARTBEAT = .15
tasks.PROGRESS_INTERVAL = .01
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
        if args.mode in ("wait", "finishing"):
            activity.get().record(dict(event="started", task=1, name="fit", label="centre"))
            if args.mode == "finishing":
                activity.get().record(dict(event="advanced", done=1, total=100, metrics={}, eta=90))
            (base / "entered").touch()
            while not (base / "finish").exists():
                if activity.get().requested and args.mode == "wait":
                    return result({"kind":"cancelled"},error=True)
                await asyncio.sleep(.005)
        if args.mode == "error":
            raise ValueError("bad specimen")
        if args.mode == "protocol":
            g = await core.supervisor.worker()
            await g.request({"type": "unknown_command"})
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
        for method, params in (("get", {}), ("cancel", {}), ("update", {"inputResponses": {}})):
            missing = c.rpc("tasks/" + method, {"taskId": tid, **params})["error"]
            assert missing["code"] == -32021
            assert missing["data"] == {"requiredCapabilities": {"extensions": {TASKS: {}}}}
        assert c.rpc("tasks/get", {"taskId": "unknown"}, tasks=True)["error"]["code"] == -32602
        assert (
            c.task("update", tid, inputResponses={"stale": {"action": "accept", "content": {}}})
            == ACK
        )
        assert c.task("cancel", tid) == ACK
        assert c.terminal(tid)["status"] == "cancelled"
        failure = c.call("probe0", tasks=True, mode="error")["taskId"]
        assert c.terminal(failure)["result"]["isError"]
        assert c.task("cancel", failure)["resultType"] == "complete"
        assert c.terminal(failure)["status"] == "completed"
        c.call("status")
        paged = c.call("python", code="print('x' * 18000)", tasks=True)
        page = c.terminal(paged["taskId"])["result"]["structuredContent"]["continuation"]
        assert "result" in c.rpc("resources/read", {"uri": page})
        time.sleep(task["ttlMs"] / 1000)
        assert c.rpc("tasks/get", {"taskId": tid}, tasks=True)["error"]["code"] == -32602
        assert "error" in c.rpc("resources/read", {"uri": page})


def test_task_progress_outlives_handoff_and_completion_wins_cancel(tmp_path):
    install_overlay(tmp_path)
    with connection(tmp_path, overlay=True) as c:
        c.call("status")  # Warm the worker before starting the shortened expiry clock.
        task = c.rpc(
            "tools/call",
            {
                "name": "probe0",
                "arguments": {"mode": "finishing"},
                "_meta": {"progressToken": "fit"},
            },
            tasks=True,
        )["result"]
        tid = task["taskId"]
        # Wait through two heartbeat notifications on a concurrent blocking call.
        other = c.send(
            "tools/call",
            {
                "name": "probe0",
                "arguments": {"mode": "wait"},
                "_meta": {"progressToken": "blocking"},
            },
        )
        for _ in range(3):
            event = c.receive("method", "notifications/progress")
            assert event["params"]["progressToken"] == "blocking"
        working = c.task("get", tid)
        assert working["statusMessage"] == "fit centre 1/100 · eta 90 s"
        assert working["pollIntervalMs"] == 10_000
        assert working["lastUpdatedAt"] > task["createdAt"]
        assert "result" not in working
        assert c.task("cancel", tid) == ACK
        (tmp_path / "finish").touch()
        success = c.terminal(tid)
        assert success["status"] == "completed" and success["resultType"] == "complete"
        assert success["result"]["structuredContent"] == {"generation": 0}
        assert c.receive("id", other)["result"]["resultType"] == "complete"
        failed = c.terminal(c.call("probe0", tasks=True, mode="protocol")["taskId"])
        assert failed["status"] == "failed" and failed["error"]["code"] == -32603
        assert "unknown_command" in failed["error"]["message"]


def test_worker_tasks_use_production_tools_and_ttl(tmp_path):
    with connection(tmp_path) as c:
        task = c.call("simulate", tasks=True, size=16, n_views=32, rotation=360)
        assert task["resultType"] == "task" and task["ttlMs"] == 86_400_000
        assert task["pollIntervalMs"] == 2_000
        simulation = c.terminal(task["taskId"])["result"]["structuredContent"]
        args = dict(source=simulation["members"]["scan"], method="opposite-views", fit="center")
        assert c.call("align", tasks=True, dry_run=True, **args)["resultType"] == "complete"
        aligned = c.terminal(c.call("align", tasks=True, **args)["taskId"])
        assert not aligned["result"]["isError"], aligned["result"]["content"][0]
        assert (
            aligned["status"] == "completed"
            and aligned["result"]["structuredContent"]["members"]["scan"]
        )
        bad = c.terminal(c.call("python", tasks=True, code="1 / 0")["taskId"])
        assert bad["status"] == "completed" and bad["result"]["isError"]
        job = c.call(
            "reconstruct", tasks=True, source=args["source"], method="sirt", iterations=10000
        )
        assert c.task("cancel", job["taskId"]) == ACK
        assert c.terminal(job["taskId"])["status"] == "cancelled"


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


@pytest.mark.parametrize("size", [32, pytest.param(128, marks=pytest.mark.gpu)])
def test_reconstruction_chain_matches_direct_api(tmp_path, size):
    import numpy as np

    def data(name, **params):
        reply = c.call(name, **params)
        assert not reply["isError"], reply["content"][0]["text"]
        if name == "align" and not params.get("dry_run"):
            assert [block["type"] for block in reply["content"]] == ["text"]
            assert "pose RMS" in reply["content"][0]["text"]
        if name == "reconstruct" and not params.get("dry_run"):
            assert any(block["type"] == "image" for block in reply["content"])
        return reply["structuredContent"]

    with connection(tmp_path) as c:
        acquisition = dict(phantom="shepp", size=size, n_views=size * 2, rotation=360)
        simulated = data("simulate", **acquisition, dry_run=True)
        assert simulated["views"] == size * 2 and simulated["phantom_shape"] == [size] * 3
        sim = data("simulate", **acquisition)
        scan = sim["members"]["scan"]
        settings = dict(source=scan, fit="center", method="opposite-views")
        assert data("align", **settings, dry_run=True)["pairs"] > 0
        fitted = data("align", **settings)
        recon = dict(source=fitted["members"]["scan"], method="fbp")
        assert data("reconstruct", **recon, dry_run=True)["grid"]["nx"] == size
        volume = data("reconstruct", **recon)
        candidates, table = check_chain_panels_and_evaluation(data, volume, recon, scan, size)
        reply = c.call("save", source=volume["handle"], path="result.nxs")
        assert not reply["isError"], reply["content"][0]["text"]
        assert [x["name"] for x in reply["content"] if x["type"] == "resource_link"] == [
            "result.nxs",
            "result.nxs.manifest.json",
        ]
        if size == 32:
            check_sweep_and_difference(c, data, scan, volume)
            corrected = data("correct", source=scan)
            edited = data(
                "edit_scan", operation=dict(type="select", source=corrected["handle"], views=[1, 3])
            )
            assert edited["shape"][0] == 2
            for source in (volume["handle"], "result.nxs"):
                array = data("project", source=scan, volume=source)
                back = data("backproject", source=scan, projections=array["handle"])
                assert back["shape"] == [size] * 3
                data("control", action="close", target=array["handle"])
            plan = data("export", source=volume["handle"], path="stack", dry_run=True)
            assert plan["slice_count"] == size and plan["destination_exists"] is False
            written = data("export", source=volume["handle"], path="stack")
            assert Path(written["first_file"]).is_file() and Path(written["last_file"]).is_file()
            sidecar = json.loads((tmp_path / "stack/slice.json").read_text())
            assert (tmp_path / "stack" / sidecar["source"]["manifest"]).is_file()
            assert data("export", source=volume["handle"], path="export.raw")["paths"]
            assert (
                data("open", path="result.nxs", kind="reconstruction", dry_run=True)["type"]
                == "validated"
            )

    phantom = tj.simulate(**acquisition)
    check_scores(phantom, candidates, table, size)
    aligned = tj.align(phantom.scan, fit="center", method="opposite-views")
    assert fitted["calibration"]["center"]["value"] == pytest.approx(
        aligned.scan.detector.center[0], abs=1e-5
    )
    actual = tj.load_reconstruction(tmp_path / "result.nxs")
    expected = tj.reconstruct(aligned.scan, "fbp")
    np.testing.assert_allclose(actual.volume, expected.volume, rtol=1e-4, atol=3e-5)
    assert np.linalg.norm(actual.volume - phantom.phantom) < np.linalg.norm(phantom.phantom)


def check_chain_panels_and_evaluation(data, volume, recon, scan, size):
    index = size // 2
    window = dict(axis="y", index=index, roi=[[1, 7], [index, index + 1], [5, 10]])
    view = data("view", source=volume["handle"], **window)["panels"][0]
    preview = data("preview", operation=dict(type="slice", **window, **recon))["panels"][0]
    assert preview["labels"] == view["labels"]
    assert preview["intensity_limits"] == pytest.approx(view["intensity_limits"], abs=1e-5)
    handle = volume["handle"]
    compact = data("status")["handles"][handle]
    assert compact == {
        **{k: volume[k] for k in ("kind", "shape", "unit")},
        "bytes": compact["bytes"],
    }
    assert compact["bytes"] > 0
    assert "manifest" in data("status", detail="full")["handles"][handle]
    candidates = [
        dict(type="reconstruct", method="cgls", iterations=2),
        dict(type="reconstruct", method="cgls", iterations=3, config=dict(interpolation="cubic")),
    ]
    candidates.append(dict(type="align", fit="center", method="opposite-views"))
    table = data(
        "evaluate",
        source=scan,
        candidates=candidates,
        reference=dict(type="reconstruct", method="cgls", iterations=3),
    )
    assert len(table["scores"]) == 3
    assert data("python", code=f"float(handles[{volume['handle']!r}].volume.max())")["value"] > 0
    return candidates, table


def check_scores(phantom, candidates, table, size):
    import numpy as np

    from tomojax.evaluation import Split
    from tomojax.forward import project_joseph
    from tomojax.geometry import stack_view_poses
    from tomojax.recon import CGLSConfig

    split = Split.interleaved(phantom.scan.projections.shape, stride=5, offset=2)
    assert table["split_fingerprint"] == split.fingerprint
    training = phantom.scan.selected(split.train.ravel())
    for row, setting in zip(table["scores"], candidates, strict=True):
        assert row["count"] == int(split.test.sum()) * size**2 and np.isfinite(row["mse"])
        if size > 32:
            continue
        fit_scan = training
        detector = phantom.scan.detector
        if setting["type"] == "align":
            alignment = tj.align(training, method="opposite-views", fit="center")
            fit_scan = alignment.scan
            detector = fit_scan.detector
            assert alignment.image is None
            assert row["diagnostic"]["calibration"]["center"]["value"] == pytest.approx(
                detector.center[0], abs=1e-5
            )
        cfg = CGLSConfig(
            iterations=setting.get("iterations", 3),
            interpolation=setting.get("config", {}).get("interpolation", "linear"),
        )
        fitted_volume = tj.reconstruct(fit_scan, "cgls", config=cfg)
        predicted = project_joseph(
            fitted_volume.volume,
            stack_view_poses(phantom.scan.geometry, size * 2),
            phantom.scan.grid,
            detector,
            interpolation=cfg.interpolation,
        )
        expected_score = split.score(predicted, phantom.scan.projections)
        assert row["count"] == expected_score.count
        assert row["relative_l2"] == pytest.approx(expected_score.relative_l2, rel=1e-6)
        assert row["mse"] == pytest.approx(expected_score.mse, rel=2e-6)
        assert np.linalg.norm(fitted_volume.volume - phantom.phantom) < np.linalg.norm(
            phantom.phantom
        )


def test_directory_discovery_counts_frames_without_worker_calls(tmp_path):
    import asyncio

    for i in range(1572):
        (tmp_path / f"clay_{i}.tif").touch()
    for name in ("clay.xtekct", "clay_recon.xtekct"):
        (tmp_path / name).touch()

    class Supervisor:
        base = tmp_path
        ledger = None

        async def worker(self):
            pytest.fail("Directory discovery opened a worker request")

    from types import SimpleNamespace

    with_tool = tools.Tools(SimpleNamespace(supervisor=Supervisor()))
    reply = asyncio.run(with_tool.inspect(tools.Inspect(str(tmp_path))))
    data = reply.structured_content
    assert data["tiff_files"] == 1572
    assert set(data["datasets"]) == {
        str(tmp_path / n) for n in ("clay.xtekct", "clay_recon.xtekct")
    }
    assert len(reply.content[0].text) < 1024


def test_read_result_parts_after_reload(tmp_path):
    with h5py.File(tmp_path / "custom.h5", "w") as f:
        for n in range(600):
            f[f"diagnostics/{n}"] = n
    with connection(tmp_path) as c:
        clipped = c.call("inspect", path="custom.h5")
        uri = clipped["structuredContent"]["continuation"]
        c.call("control", action="reload")
        chunks = []
        while uri:
            reply = c.rpc("resources/read", {"uri": uri})
            content = reply["result"]["contents"][0]["text"]
            assert len(content.encode()) < 20_000
            part = json.loads(content)
            chunks.append(part["data"])
            uri = part["next"]
        full = json.loads("".join(chunks))
        assert len(full["structuredContent"]["datasets"]) == 600


def test_client_cancellation_and_task_progress_scope(tmp_path):
    install_overlay(tmp_path)
    with connection(tmp_path, overlay=True) as c:
        ref = c.call("simulate", size=16)["structuredContent"]["members"]["scan"]
        params = {
            "name": "reconstruct",
            "arguments": {"source": ref, "method": "sirt", "iterations": 10000},
            "_meta": {"progressToken": "reconstructing"},
        }
        rid = c.send("tools/call", params)
        progress = c.receive("method", "notifications/progress")
        assert progress["params"]["progressToken"] == "reconstructing"
        deadline = time.monotonic() + 10
        while not c.call("status")["structuredContent"]["jobs"]:
            assert time.monotonic() < deadline
        c.send("notifications/cancelled", {"requestId": rid}, notify=True)
        while (state := c.call("status")["structuredContent"])["jobs"]:
            assert time.monotonic() < deadline
        assert len(state["handles"]) == 1
        task = c.rpc(
            "tools/call",
            {
                "name": "probe0",
                "arguments": {"mode": "wait"},
                "_meta": {"progressToken": "handed-off"},
            },
            tasks=True,
        )["result"]
        assert c.task("get", task["taskId"])["status"] == "working"
        c.task("cancel", task["taskId"])
        assert c.terminal(task["taskId"])["status"] == "cancelled"
        assert all(row.get("params", {}).get("progressToken") != "handed-off" for row in c.inbox)


def start_writer(server_kind, env, dest, processes, *, paused):
    server = subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", server_kind],
        stdin=-1,
        stdout=-1,
        stderr=-1,
        text=True,
        env={**env, "PAUSE_INSTALL": "1" if paused else ""},
    )
    processes.append(server)
    c = Client(server, legacy=False)
    if server_kind == "serve":
        ready = c.receive("type", "ready")
        if not paused:
            assert str(dest) in ready["recovery"]["restored"]

    def send(name, args):
        if server_kind == "mcp":
            return c.send("tools/call", dict(name=name, arguments=args))
        return c.send_worker(name, args)

    def end(rid):
        if server_kind == "mcp":
            reply = c.receive("id", rid)["result"]
            assert not reply["isError"], reply
            return reply["structuredContent"]
        return c.receive("request_terminal", rid)["result"]["value"]

    if not paused and server_kind == "mcp":
        assert str(dest) in c.call("status")["structuredContent"]["recovery"]["restored"]
    size = 6 if paused else 8
    handle = end(send("simulate", dict(size=size)))["handle"]
    ref = handle if server_kind == "mcp" else dict(type="member", handle=handle, member="self")
    return server, end, send("save", dict(source=ref, path=str(dest), force=True))


@pytest.mark.parametrize("server_kind", ["serve", "mcp"])
def test_native_worker_child_cannot_inherit_commit_lock(tmp_path, server_kind):
    """Kill A mid-install while its worker's native child lives. B must recover and save."""
    import signal

    import numpy as np

    # Replace only the uid home for the test processes. Pause the actual
    # committer after its first install, and fork in the numerical worker while
    # the lock is held. The native child keeps *all* inherited descriptors.
    (tmp_path / "sitecustomize.py").write_text("""
import os, pwd, sys, time
from pathlib import Path
root = Path(os.environ["COMMIT_TEST"])
getpwuid = pwd.getpwuid
def lookup(uid):
    record = list(getpwuid(uid))
    record[5] = str(root)
    return pwd.struct_passwd(record)
pwd.getpwuid = lookup
if os.environ.get("PAUSE_INSTALL"):
    rename = Path.replace
    def replace(path, target):
        result = rename(path, target)
        if Path(target) == root / 'volume.nxs':
            (root/'committer').write_text(str(os.getpid()))
            import signal
            signal.raise_signal(signal.SIGSTOP)
        return result
    Path.replace = replace
    if 'serve' in sys.orig_argv:
        import ctypes, threading
        def fork_in_worker():
            libc = ctypes.CDLL(None)
            fork, sleep, exit_now = libc.fork, libc.sleep, libc._exit
            while not (root/'committer').exists():
                time.sleep(.005)
            child = fork()
            if child == 0:
                sleep(45)
                exit_now(0)
            (root/'child').write_text(str(child))
        threading.Thread(target=fork_in_worker, daemon=True).start()
""")
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "COMMIT_TEST": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "CLAUDE_PROJECT_DIR": str(tmp_path),
        "JAX_PLATFORMS": "cpu",
    }
    (dest := tmp_path / "volume.nxs").write_bytes(b"original")
    processes = []

    child = None
    try:
        a, _, _ = start_writer(server_kind, env, dest, processes, paused=True)
        deadline = time.monotonic() + 20
        while not (tmp_path / "child").exists():
            assert a.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        committer = int((tmp_path / "committer").read_text())
        child = int((tmp_path / "child").read_text())
        if committer != a.pid:
            a.kill()  # The GUI worker cannot recover its failed helper ahead of B.
        os.kill(committer, signal.SIGKILL)
        a.wait(timeout=5)
        b, end, rid = start_writer(server_kind, env, dest, processes, paused=False)
        assert str(dest) in end(rid)["paths"]
        os.kill(child, 0)
        b.stdin.close()
        assert b.wait(timeout=10) == 0
        with h5py.File(dest) as f:
            np.testing.assert_array_equal(
                f["/entry/processing/tomojax/volume"][:], tj.simulate(size=8).phantom
            )
        assert not (tmp_path / ".local/state/tomojax/commits/journal.json").exists()
    finally:
        if child:
            os.kill(child, signal.SIGKILL)
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def test_slab_destination_over_stdio(tmp_path):
    import numpy as np

    with h5py.File(tmp_path / "raw.h5", "w") as f:
        f["exchange/data"] = np.random.default_rng(7).integers(350, 850, size=(12, 5, 8))
        f["exchange/theta"] = np.arange(12) * 15
    with connection(tmp_path) as c:
        raw = c.call(
            "open",
            path="raw.h5",
            kind="frames",
            flats=1000,
            darks=0,
            detector=dict(nu=8, nv=5, du=1, dv=1),
            geometry=dict(kind="parallel"),
        )
        ref = raw["structuredContent"]["handle"]
        args = dict(source=ref, path="slabs.nxs", method="fbp", config={"backend": "jax"})
        valid = c.call("reconstruct_slabs", dry_run=True, detail="full", **args)
        assert valid["structuredContent"]["type"] == "validated"
        assert not (tmp_path / "slabs.nxs").exists()
        saved = c.call("reconstruct_slabs", **args)
        assert not saved["isError"], saved
        assert saved["structuredContent"]["paths"] == [
            str(tmp_path / "slabs.nxs"),
            str(tmp_path / "slabs.nxs.manifest.json"),
        ]
        assert list(c.call("status")["structuredContent"]["handles"]) == [ref]
        scan = c.call("correct", source=ref)["structuredContent"]["handle"]
        projected = c.call("project", source=scan, volume="slabs.nxs")
        assert projected["structuredContent"]["shape"] == [12, 5, 8]
        error = c.call("reconstruct_slabs", **args)
        assert error["isError"]


def check_sweep_and_difference(c, data, scan, volume):
    sweep = c.call(
        "preview",
        operation=dict(
            type="centre_sweep",
            axis="z",
            index=8,
            centres=[-1, 0, 1],
            roi=[[8, 9], [1, 7], [5, 10]],
            source=scan,
            method="fbp",
        ),
    )
    assert [b["type"] for b in sweep["content"]] == ["text", "image"]
    assert len(sweep["structuredContent"]["panels"]) == 3
    cropped = data(
        "view", source=volume["handle"], axis="z", index=8, roi=[[8, 9], [1, 7], [5, 10]]
    )
    for panel in sweep["structuredContent"]["panels"]:
        assert panel["labels"][-2:] == cropped["panels"][0]["labels"][-2:]
    same = data("view", source=volume["handle"], other="result.nxs", mode="difference")
    assert same["intensity_limits"] == [0, 0]
    smaller = data("simulate", size=16, n_views=16)
    mismatch = c.call("view", source=volume["handle"], mode="difference", other=smaller["handle"])
    assert mismatch["isError"] and all(
        ref in mismatch["content"][0]["text"] for ref in (volume["handle"], smaller["handle"])
    )


def test_correction_comparison_and_frame_profile(tmp_path):
    import h5py
    import numpy as np

    counts = np.arange(6 * 8 * 12, dtype=np.float32).reshape(6, 8, 12) / 600 + 0.1
    with h5py.File(tmp_path / "raw.h5", "w") as file:
        file["exchange/data"] = counts
        file["exchange/theta"] = np.arange(6) * 10.0
    with connection(tmp_path) as c:
        opened = c.call(
            "open",
            path="raw.h5",
            kind="frames",
            flats=2,
            darks=0,
            pixel_size=1,
            geometry={"kind": "parallel"},
        )
        frames = opened["structuredContent"]["handle"]
        reply = c.call(
            "preview",
            operation=dict(type="correction_preview", source=frames, rows=[0, 4], binning=2),
        )
        assert [b["type"] for b in reply["content"]] == ["text", "image"]
        data = reply["structuredContent"]
        before, after = data["panels"]
        assert "native" in before["array"] and "corrected" in after["array"]
        assert before["labels"][1:] != after["labels"][1:]
        np.testing.assert_allclose(before["intensity_limits"], np.percentile(counts[3], [1, 99]))
        assert before["intensity_limits"] != after["intensity_limits"]
        assert "intensity_limits" not in data
        profile = c.call(
            "view", source=frames, mode="profile", roi=[[3, 4], [0, 1], [2, 12]], stride=[1, 1, 2]
        )["structuredContent"]
        np.testing.assert_allclose(profile["profile"], counts[3, 0, 2::2])
        assert len(profile["coordinates"]) == 5


def test_project_and_backproject_select_members_of_saved_npz(tmp_path):
    def call(name, **params):
        reply = c.call(name, **params)
        assert not reply["isError"], reply
        return reply["structuredContent"]

    with connection(tmp_path) as c:
        sim = call("simulate", size=8, n_views=12)
        scan = sim["members"]["scan"]
        call("save", source=sim["handle"], path="sim.npz")
        for tool, field, member in (
            ("project", "volume", "phantom"),
            ("backproject", "projections", "projections"),
        ):
            actual = call(tool, source=scan, **{field: "sim.npz"})["handle"]
            expected = call(tool, source=scan, **{field: sim["handle"] + "." + member})["handle"]
            assert call(
                "python", code=f"np.array_equal(handles[{actual!r}], handles[{expected!r}])"
            )["value"]
        for source in (sim["members"]["phantom"], "sim.npz"):
            stats = call("view", source=source, mode="statistics")
            assert stats["count"] == 8**3
        candidate = {"type": "reconstruct", "method": "cgls", "iterations": 2}
        settings = {"reference": candidate, "candidates": [candidate]}
        expected_score = call("evaluate", source=scan, **settings)["scores"][0]["mse"]
        assert call("evaluate", source="sim.npz", **settings)["scores"][0]["mse"] == pytest.approx(
            expected_score
        )


def test_backproject_discovers_nexus_projections(tmp_path):
    import numpy as np

    path = tmp_path / "simulation.nxs"
    sim = tj.simulate(size=8, n_views=12)
    tj.save(path, sim)
    with h5py.File(path, "r+") as file:
        del file["/entry/instrument/detector/data"]
        file["other"] = np.zeros_like(sim.scan.projections)
    plain = tmp_path / "arrays.h5"
    with h5py.File(plain, "w") as file:
        file["/entry/projections"] = sim.scan.projections
    with connection(tmp_path) as c:
        scan = c.call("open", kind="scan", path=str(path))["structuredContent"]["handle"]
        for src, expected in ((str(path), True), (str(plain), True), (f"{path}::/other", False)):
            reply = c.call("backproject", source=scan, projections=src)
            assert not reply["isError"], reply
            array = reply["structuredContent"]["handle"]
            value = c.call("python", code=f"bool(np.any(handles[{array!r}]))")
            assert value["structuredContent"]["value"] is expected


def test_projection_preview_keeps_requested_stride(tmp_path):
    import numpy as np

    with h5py.File(tmp_path / "stripes.h5", "w") as file:
        file["data"] = np.tile(np.arange(1536) % 2, (1, 3, 1))
    with connection(tmp_path) as c:
        selected = dict(source="stripes.h5::/data", stride=[1, 1, 2])
        stats = c.call("view", mode="statistics", **selected)["structuredContent"]
        assert stats["mean"] == 0
        (panel,) = c.call("view", mode="projection", **selected)["structuredContent"]["panels"]
        assert panel["intensity_limits"] == [0, 0]
        assert "stride (1, 1, 2), display subsampling (1, 1, 2)" in panel["labels"]


@pytest.mark.parametrize("suffix", ["nxs", "npz"])
def test_backproject_selects_sample_frames_only_for_bare_paths(tmp_path, suffix):
    import numpy as np

    scan = tj.simulate(size=8, n_views=13).scan
    path = tmp_path / f"samples.{suffix}"
    tj.save(path, scan)
    if suffix == "nxs":
        with h5py.File(path, "r+") as file:
            file["/entry/instrument/detector/image_key"][4] = 3
    else:
        with np.load(path) as saved:
            fields = dict(saved)
        fields["image_key"] = np.zeros(13, np.int32)
        fields["image_key"][4] = 3
        np.savez(path, **fields)
    with connection(tmp_path) as c:
        source = c.call("open", path=path.name)["structuredContent"]["handle"]
        actual, expected = (
            c.call("backproject", source=source, projections=ref)["structuredContent"]["handle"]
            for ref in (path.name, f"{source}.projections")
        )
        assert c.call("python", code=f"np.array_equal(handles[{actual!r}], handles[{expected!r}])")[
            "structuredContent"
        ]["value"]
        if suffix != "nxs":
            return
        literal = c.call(
            "backproject", source=source, projections=f"{path}::/entry/instrument/detector/data"
        )
        assert literal["isError"] and "does not match" in literal["content"][0]["text"]


def test_backproject_bliss_projection_blocks(tmp_path):
    from tests.test_esrf_readers import _write_bliss_sequence

    path = tmp_path / "sequence.h5"
    _write_bliss_sequence(path, "measurement/hrsrot")
    with connection(tmp_path) as c:
        scan = c.call("open", path=path.name)["structuredContent"]["handle"]
        volumes = []
        for ref in (path.name, f"{scan}.projections"):
            result = c.call("backproject", source=scan, projections=ref)
            assert not result["isError"], result
            volumes.append(result["structuredContent"]["handle"])
        actual, expected = volumes
        assert c.call("python", code=f"np.array_equal(handles[{actual!r}], handles[{expected!r}])")[
            "structuredContent"
        ]["value"]
