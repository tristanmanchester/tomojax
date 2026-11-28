"""Native MCP conversations use files and durable tomography jobs on the worker."""

import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import h5py
from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image
import pytest

from tests.mcp_client import connection
from tests.worker_process import binary
import tomojax as tj

pytestmark = pytest.mark.surface


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("TOMOJAX_STUDIO_PYTHON", sys.executable)
    executable = binary()

    def connect(profile=None, *, legacy=False):
        return connection(tmp_path, executable=executable, profile=profile, legacy=legacy)

    return connect


def profile(directory, monkeypatch, settings):
    config = directory / "config" / "tomojax"
    config.mkdir(parents=True)
    (config / "connections.toml").write_text(f'[fixture]\npython = "{sys.executable}"\n' + settings)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config.parent))


def checked(c, name, **args):
    reply = c.call(name, **args)
    assert not reply["isError"], reply["content"][0]
    return reply


def ready(c):
    end = time.monotonic() + 30
    status = {}
    while time.monotonic() < end:
        status = checked(c, "status")["structuredContent"]
        if status["executor"] == "ready":
            return
        time.sleep(0.05)
    pytest.fail(f"Executor did not reach ready: {status}")


@pytest.mark.parametrize("legacy", [True, False])
def test_workflow_guidance_and_solver_schema_match_the_installed_server(tmp_path, native, legacy):
    with native(legacy=legacy) as rust, connection(tmp_path, legacy=legacy) as python:
        for method, params in [
            ("prompts/list", {}),
            ("prompts/get", {"name": "find-the-center", "arguments": {"source": "scan.nxs"}}),
            ("resources/read", {"uri": "tomojax://guide/python"}),
            (
                "tools/call",
                {"name": "describe", "arguments": {"operation": "reconstruct", "method": "fbp"}},
            ),
        ]:
            assert rust.rpc(method, params) == python.rpc(method, params)


def image(block):
    assert block["mimeType"] == "image/png"
    with Image.open(BytesIO(base64.b64decode(block["data"]))) as decoded:
        return np.array(decoded)


def test_worker_filesystem_namespace(tmp_path, monkeypatch, native):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap is required for two filesystems at the same absolute path")
    local, remote = (tmp_path / name for name in ("local", "remote"))
    local.mkdir()
    remote.mkdir()
    # Run the real connect and serve processes, with only the SSH transport substituted.
    ssh = tmp_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\nset -eu\n"
        'if [ "$1" = "-G" ]; then echo "hostname fixture"; echo "controlpath /fixture/master"; '
        'echo "controlmaster auto"; echo "controlpersist 3600"; exit; fi\n'
        'while [ "$1" != "--" ]; do shift; done\nshift 2\n'
        'export HOME="$FIXTURE_HOME"\ncd "$HOME"\n'
        'exec bwrap --bind / / --dev /dev --bind "$REMOTE_DATA" "$LOCAL_DATA" -- sh -c "$1"\n'
    )
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FIXTURE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_DATA", str(local))
    monkeypatch.setenv("REMOTE_DATA", str(remote))
    profile(tmp_path, monkeypatch, 'ssh = ["fixture"]\n')
    (local / "laptop.tif").touch()
    for i in range(4201):
        (remote / f"{i}.tif").touch()
    for path, n in [(remote, 8), (local, 4)]:
        with h5py.File(path / "unknown.h5", "w") as f:
            f["signal"] = np.zeros((n, 5))
        sim = tj.simulate("sphere", size=n, n_views=12)
        tj.save(path / "scan.npz", sim.scan)
    with native("fixture") as client:
        ready(client)
        assert (
            checked(client, "inspect", path=str(local))["structuredContent"]["tiff_files"] == 4201
        )
        report = checked(client, "inspect", path=str(local / "unknown.h5"))["structuredContent"]
        assert report["datasets"]["/signal"]["shape"] == [8, 5]
        npz = checked(client, "inspect", path=str(local / "scan.npz"), preview=True)
        assert npz["structuredContent"]["projection"]["shape"] == [12, 8, 8]
        with connection(remote) as reference:
            expected = checked(reference, "inspect", path="scan.npz", preview=True)
        np.testing.assert_array_equal(image(npz["content"][1]), image(expected["content"][1]))
        pixels = image(expected["content"][1])
        Image.fromarray(pixels).save(remote / "slice.png")
        get = checked(client, "download", path=str(local / "slice.png"), to="display.png")
        assert (
            get["structuredContent"]["sha256"]
            == sha256((remote / "slice.png").read_bytes()).hexdigest()
        )
        with Image.open(tmp_path / "display.png") as downloaded:
            np.testing.assert_array_equal(np.array(downloaded), pixels)
        assert not (local / "slice.png").exists()
        blob = np.arange(65537, dtype="<i4").tobytes()
        (tmp_path / "sample.bin").write_bytes(blob)
        transfer = {"from": "sample.bin", "path": str(local / "sample.bin")}
        put = checked(client, "upload", **transfer)
        assert put["structuredContent"]["bytes"] == len(blob)
        assert (remote / "sample.bin").read_bytes() == blob
        assert client.call("upload", **transfer)["isError"]
        assert client.call("download", path=str(local / "sample.bin"), to="display.png")["isError"]
        assert checked(client, "status")["structuredContent"]["connection"] == "fixture"


def test_detached_job_survives_mcp_restart(tmp_path, monkeypatch, native):
    scheduler = Path(__file__).resolve().parents[1] / "studio/crates/tomojax-jobs/tests/slurm"
    control = tmp_path / "controller"
    control.mkdir()
    (control / "hold").touch()
    monkeypatch.setenv("FAKE_SLURM", str(control))
    monkeypatch.setenv("PATH", str(scheduler) + os.pathsep + os.environ["PATH"])
    root = tmp_path / "results"
    profile(tmp_path, monkeypatch, f'jobs = "slurm"\nresults = "{root}"\n')
    with native("fixture") as c:
        bad = c.call(
            "submit",
            operation={
                "type": "reconstruct",
                "source": "input.nxs",
                "config": {
                    "support_mask": {"type": "npy", "path": "mask.npy", "artifact": "session-1"}
                },
            },
        )
        assert bad["isError"] and "artifact" in bad["content"][0]["text"]
        assert not (root / "jobs").exists()
        submission = checked(c, "submit", operation={"type": "simulate", "size": 8, "n_views": 6})
        job = submission["structuredContent"]["job"]
    (control / "hold").unlink()
    with native("fixture") as c:
        end = time.monotonic() + 45
        while time.monotonic() < end:
            rows = checked(c, "jobs")["structuredContent"]["jobs"]
            assert [Path(r["directory"]).name for r in rows] == [job]
            if rows[0]["state"] == "completed":
                break
            assert rows[0]["state"] not in ("failed", "canceled")
            time.sleep(0.1)
        else:
            pytest.fail("Job did not complete")
        done = checked(c, "jobs", job=job)
        planes = [image(b) for b in done["content"] if b["type"] == "image"]
        assert len(planes) == 3
        with h5py.File(root / "jobs" / job / "result" / "result.nxs") as dataset:
            phantom = dataset["entry/phantom/data"][:]
        for axis, plane in enumerate(planes):
            expected = np.take(phantom, 4, axis=axis)
            np.testing.assert_array_equal(plane, np.round(expected * 255))


def test_login_failure_caches_the_remedy_and_submits_nothing(tmp_path, monkeypatch, native):
    ssh = tmp_path / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = -G ]; then echo "hostname fixture"; exit; fi\n'
        'echo probe >> "$PROBES"\necho "Permission denied (publickey)." >&2\nexit 255\n'
    )
    ssh.chmod(0o700)
    probes = tmp_path / "probes"
    monkeypatch.setenv("PROBES", str(probes))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    profile(tmp_path, monkeypatch, 'ssh = ["fixture"]\n')
    with native("fixture") as c:
        first = c.call("status")
        assert first["isError"]
        expected = "tomojax-worker connect fixture --login"
        assert expected in first["content"][0]["text"]
        attempts = probes.read_text()
        start = time.monotonic()
        retry = c.call("submit", operation={"type": "simulate", "size": 8})
        assert time.monotonic() - start < 1
        assert retry["isError"] and expected in retry["content"][0]["text"]
        assert probes.read_text() == attempts


def test_incompatible_executor_refuses_worker_tools(tmp_path, monkeypatch, native):
    fake = tmp_path / "fixture-python"
    fake.write_text(
        "#!/bin/sh\n"
        'mkdir "$TMPDIR/session"\n'
        """printf '{"type":"ready","protocol":1,"schema":"different","root":"%s","versions":{"tomojax":"fixture-version"}}\\n' "$TMPDIR/session"\n"""
        'cat > "$REQUESTS"\n'
    )
    fake.chmod(0o700)
    requests = tmp_path / "requests"
    monkeypatch.setenv("TOMOJAX_STUDIO_PYTHON", str(fake))
    monkeypatch.setenv("REQUESTS", str(requests))
    with native() as c:
        for tool, args in [
            ("describe", {"operation": "align"}),
            ("jobs", {"results": str(tmp_path)}),
        ]:
            r = c.call(tool, **args)
            assert r["isError"]
            assert "fixture-version" in r["content"][0]["text"]
            assert tj.__version__ in r["content"][0]["text"]
    assert requests.read_text() == ""


def test_submit_resolves_nested_and_top_level_paths(tmp_path, monkeypatch, native):
    scheduler = Path(__file__).resolve().parents[1] / "studio/crates/tomojax-jobs/tests/slurm"
    control = tmp_path / "controller"
    control.mkdir()
    (control / "hold").touch()
    monkeypatch.setenv("FAKE_SLURM", str(control))
    monkeypatch.setenv("PATH", str(scheduler) + os.pathsep + os.environ["PATH"])
    root = tmp_path / "results"
    profile(tmp_path, monkeypatch, f'jobs = "slurm"\nresults = "{root}"\n')
    operation = {
        "type": "reconstruct",
        "method": "sirt",
        "source": "scan.npz",
        "config": {"support": "mask.npy"},
    }
    with native("fixture") as client:
        listing = client.rpc("tools/list", {})["result"]["tools"]
        schema = next(t["inputSchema"] for t in listing if t["name"] == "submit")
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate({"operation": operation})
        directory = Path(checked(client, "status")["structuredContent"]["directory"])
        job = checked(client, "submit", operation=operation)["structuredContent"]["job"]
        try:
            script = json.loads((root / "jobs" / job / "script.jsonl").read_text())
            expanded = script["command"]["operation"]
            assert expanded["source"] == {"type": "path", "path": str(directory / "scan.npz")}
            assert expanded["config"] == {
                "type": "SIRTConfig",
                "support": {"type": "npy", "path": str(directory / "mask.npy")},
            }
        finally:
            checked(client, "cancel_job", job=job)
            (control / "hold").unlink()


def test_difference_preserves_float64_storage_and_matches_reference(tmp_path, native):
    geom = tj.simulate(size=8, n_views=12).scan.geometry
    first, second = tmp_path / "left.nxs", tmp_path / "right.nxs"
    delta = 2.0**-25
    for path, value in [(first, 1.0 + delta), (second, 1.0)]:
        tj.save(path, tj.Reconstruction(np.ones((8, 8, 8), np.float32), geom, "fbp", {}))
        with h5py.File(path, "r+") as file:
            volume = file["entry/volume/data"]
            attrs = dict(volume.attrs)
            shape = volume.shape
            del file["entry/volume/data"]
            file.create_dataset(
                "entry/volume/data", data=np.full(shape, value, dtype="<f8")
            ).attrs.update(attrs)
    with native() as rust, connection(tmp_path) as python:
        args = {"source": "left.nxs", "other": "right.nxs", "mode": "difference", "axis": "z"}
        a, b = (checked(client, "view", **args) for client in (rust, python))
        assert a["structuredContent"]["intensity_limits"] == [delta, delta]
        assert a["structuredContent"] == b["structuredContent"]
        np.testing.assert_array_equal(image(a["content"][1]), image(b["content"][1]))


def test_point_drawers_report_the_same_sample_and_draw_the_same_pixels(tmp_path, native):
    array = np.zeros((11, 14), dtype=np.float64)
    array[4, 8] = 5
    np.save(tmp_path / "point.npy", array)
    args = {"source": "point.npy", "marks": [{"type": "point", "at": [4, 8]}], "ticks": False}
    with native() as rust, connection(tmp_path) as python:
        a, b = (checked(client, "view", **args) for client in (rust, python))
        assert a["structuredContent"]["marks"][0]["panels"][0]["value"] == 5
        assert a["structuredContent"] == b["structuredContent"]
        np.testing.assert_array_equal(image(a["content"][1]), image(b["content"][1]))


def test_shape_footprints_and_outlines_match_on_a_nonuniform_plane(tmp_path, native):
    # A uniform rectangle cannot detect a different set of boundary samples in a mean.
    np.save(tmp_path / "plane.npy", np.random.default_rng(100).random((25, 20)))
    marks = [
        {"type": "point", "at": [8, 12], "label": "sample"},
        {"type": "box", "corners": [[2, 5], [9, 14]]},
        {"type": "circle", "at": [9, 7], "radius": 5},
        {"type": "arrow", "from": [15, 8], "to": [5, 6]},
        {"type": "polygon", "points": [[5, 1], [7, 8], [18, 4]]},
        {"type": "polyline", "points": [[5, 1], [7, 8], [18, 4]]},
        {"type": "text", "at": [10, 3], "label": "Hi"},
    ]
    with native() as rust, connection(tmp_path) as python:
        for mark in marks:
            a, b = (
                checked(c, "view", source="plane.npy", marks=[mark], ticks=False)
                for c in (rust, python)
            )
            assert a["structuredContent"] == b["structuredContent"], mark
            np.testing.assert_array_equal(
                image(a["content"][1]), image(b["content"][1]), err_msg=str(mark)
            )


@pytest.mark.parametrize("server", ["rust", "python"])
def test_cancel_task_on_running_reconstruction(tmp_path, native, server):
    with native() if server == "rust" else connection(tmp_path) as client:
        scan = checked(client, "simulate", size=12, n_views=16)["structuredContent"]["members"][
            "scan"
        ]
        task = client.call("reconstruct", source=scan, method="sirt", iterations=10000, tasks=True)
        tid = task["taskId"]
        assert client.task("get", tid)["status"] == "working"
        client.task("cancel", tid)
        assert client.terminal(tid)["status"] == "cancelled"
        assert not checked(client, "status")["structuredContent"]["jobs"]


def test_task_fails_with_lost_handles_after_executor_exit(native):
    with native() as client:
        source = checked(client, "python", code="register(np.array([3]))")["structuredContent"][
            "value"
        ]
        task = client.call("python", tasks=True, code="import os\nos._exit(21)")
        failed = client.terminal(task["taskId"])
        assert failed["status"] == "failed" and failed["error"]["code"] == -32603
        assert source in failed["error"]["message"]
        ready(client)
        status = checked(client, "status")["structuredContent"]
        assert status["worker_loss"]["lost"] == [source] and not status["handles"]
        state = checked(client, "python", code="len(handles)")["structuredContent"]
        assert state["value"] == 0 and state["namespace_generation"] == 2


def test_sigterm_reaps_stopped_native_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("TOMOJAX_STUDIO_PYTHON", sys.executable)
    with connection(tmp_path, executable=binary(), exit_code=-signal.SIGTERM) as client:
        ready(client)
        processes = subprocess.check_output(["ps", "-A", "-o", "pid=,ppid="], text=True)
        (child,) = (
            pid
            for pid, parent in (map(int, line.split()) for line in processes.splitlines())
            if parent == client.process.pid
        )
        os.kill(child, signal.SIGSTOP)
        client.process.terminate()
        assert client.process.wait(timeout=10) == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)


def test_saved_paths_link_to_local_outputs_and_reopen(tmp_path, native):
    with native() as client:
        simulation = checked(client, "simulate", size=8, n_views=8)["structuredContent"]
        scan = simulation["members"]["scan"]
        volume = checked(client, "reconstruct", source=scan, method="fbp")
        assert [part["type"] for part in volume["content"]] == ["text", "image"]
        path = tmp_path / "result #1.nxs"
        saved = checked(
            client, "save", source=volume["structuredContent"]["handle"], path=path.name
        )
        links = [part for part in saved["content"] if part["type"] == "resource_link"]
        assert [item["uri"] for item in links] == [
            path.as_uri(),
            Path(str(path) + ".manifest.json").as_uri(),
        ]
        reopened = checked(client, "open", path=path.name, kind="reconstruction")
        assert reopened["structuredContent"]["shape"] == [8, 8, 8]
