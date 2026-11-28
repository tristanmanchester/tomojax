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


def test_guidance_and_parameters_agree_across_protocol_eras(native):
    with native(legacy=True) as legacy, native(legacy=False) as modern:
        for method, params in [
            ("prompts/list", {}),
            ("prompts/get", {"name": "find-the-center", "arguments": {"source": "scan.nxs"}}),
            ("resources/read", {"uri": "tomojax://guide/python"}),
            (
                "tools/call",
                {"name": "describe", "arguments": {"operation": "reconstruct", "method": "fbp"}},
            ),
        ]:
            before, after = (c.rpc(method, params)["result"] for c in (legacy, modern))
            assert after.pop("resultType") == "complete"
            after.pop("_meta")
            if "cacheScope" in after:
                assert after.pop("cacheScope") == "private" and after.pop("ttlMs") == 0
            assert before == after


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
        # The worker's file has 8 rows where the laptop's file at the same path has 4.
        code = (
            f"import h5py\nwith h5py.File({str(local / 'unknown.h5')!r}) as f:\n"
            "    signal = f['signal'][:]\n"
            "signal[6, 2] = 3\n"
            "reports = show(signal, marks=[{'type': 'point', 'at': [6, 2]}])\n"
            "signal.shape[0] + reports[0]['panels'][0]['value']"
        )
        assert checked(client, "python", code=code)["structuredContent"]["value"] == 11
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
    with native() as client:
        args = {"source": "left.nxs", "other": "right.nxs", "mode": "difference", "axis": "z"}
        data = checked(client, "view", **args)["structuredContent"]
        assert data["intensity_limits"] == [delta, delta]


def assert_drawers_match(client, path, pixels, marks):
    from tomojax.cli import _marks  # check-public-imports: allow-private
    from tomojax.cli.images import Panel, limits

    reply = checked(client, "view", source=path, marks=marks, ticks=False)
    data = reply["structuredContent"]
    (meta,) = data["panels"]
    panel = Panel(meta["array"], pixels, tuple(meta["labels"]), meta["unit"])
    frame = _marks.frame_of(("row", "col"), [(0, n, 1) for n in pixels.shape], None)
    parsed = _marks.parse(marks)
    overlay = _marks.Overlay(frame, parsed, (), False)
    pictures, reports, _ = _marks.annotate([panel], [limits([panel])], [overlay], parsed)
    for report in reports:
        for found in report["panels"]:
            if "image" in found:
                found["image"] = data["images"][found["image"]]["image"]
    assert data["marks"] == reports
    with Image.open(BytesIO(pictures[0])) as expected:
        np.testing.assert_array_equal(image(reply["content"][1]), np.asarray(expected))
    return reports


def test_point_drawers_report_the_same_sample_and_draw_the_same_pixels(tmp_path, native):
    array = np.zeros((11, 14), dtype=np.float64)
    array[4, 8] = 5
    np.save(tmp_path / "point.npy", array)
    args = {"source": "point.npy", "marks": [{"type": "point", "at": [4, 8]}], "ticks": False}
    with native() as client:
        reports = assert_drawers_match(client, args["source"], array, args["marks"])
        assert reports[0]["panels"][0]["value"] == 5


def test_shape_footprints_and_outlines_match_on_a_nonuniform_plane(tmp_path, native):
    # A uniform rectangle cannot detect a different set of boundary samples in a mean.
    pixels = np.random.default_rng(100).random((25, 20))
    np.save(tmp_path / "plane.npy", pixels)
    marks = [
        {"type": "point", "at": [8, 12], "label": "sample"},
        {"type": "box", "corners": [[2, 5], [9, 14]]},
        {"type": "circle", "at": [9, 7], "radius": 5},
        {"type": "arrow", "from": [15, 8], "to": [5, 6]},
        {"type": "polygon", "points": [[5, 1], [7, 8], [18, 4]]},
        {"type": "polyline", "points": [[5, 1], [7, 8], [18, 4]]},
        {"type": "text", "at": [10, 3], "label": "Hi"},
    ]
    with native() as client:
        for mark in marks:
            assert_drawers_match(client, "plane.npy", pixels, [mark])
        assert_drawers_match(client, "plane.npy", pixels, marks)


def test_cancel_task_on_running_reconstruction(native):
    with native() as client:
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


def test_show_resizes_gray_and_rgb_snapshots(tmp_path, native):
    from tests.test_mcp_python import full_result
    from tomojax.cli.images import Panel, limits, raster, thumbnail

    with native() as c:
        for shape in [(19, 1311), (181, 4187), (1321, 3), (75, 1153, 3)]:
            code = f"show(np.random.default_rng(92).integers(0, 240, {shape}, dtype=np.uint8))"
            reply = full_result(c, checked(c, "python", code=code))
            pixels = np.random.default_rng(92).integers(0, 240, shape, dtype=np.uint8)
            panel = Panel("Python", pixels, (), "unspecified")
            with Image.open(BytesIO(thumbnail(raster(panel, *limits([panel]))))) as expected:
                np.testing.assert_array_equal(image(reply["content"][1]), np.asarray(expected))


def test_inspect_resizes_after_converting_the_full_plane(tmp_path, native):
    values = np.random.default_rng(34).standard_normal((3, 92, 5237)).astype(np.float32)
    with h5py.File(tmp_path / "scan.nxs", "w") as dataset:
        dataset["entry/instrument/detector/data"] = values
        dataset["entry/sample/rotation_angle"] = [0, 60, 120]
    plane = values[1]
    lo, hi = (float(v) for v in np.percentile(plane, [0.5, 99.5]))
    expected = Image.fromarray(
        np.round(np.clip((plane - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    )
    expected.thumbnail((1024, 1024))
    with native() as client:
        reply = checked(client, "inspect", path="scan.nxs", preview=True)
        np.testing.assert_array_equal(image(reply["content"][1]), np.asarray(expected))


def test_queued_executor_leaves_worker_tools_available(tmp_path, monkeypatch, native):
    launcher = tmp_path / "queue"
    launcher.write_text('#!/bin/sh\necho "waiting for resources" >&2\nexec sleep 90\n')
    launcher.chmod(0o700)
    profile(tmp_path, monkeypatch, f'session = "{launcher}"\nresults = "{tmp_path}"\n')
    with native("fixture") as client:
        limit = time.monotonic() + 10
        while time.monotonic() < limit:
            state = checked(client, "status")["structuredContent"]
            if state.get("detail") == "waiting for resources":
                break
        else:
            pytest.fail(f"Missing scheduler message: {state}")
        refused = client.call("simulate", size=8)
        assert refused["isError"]
        assert "waiting in the queue. waiting for resources" in refused["content"][0]["text"]
        # Nothing is wrong with the arguments: the executor is not there yet.
        error = refused["structuredContent"]
        assert error["kind"] == "unavailable"
        assert "status" in error["next"] and "parameters" not in error["next"]
        assert ".." not in refused["content"][0]["text"]
        assert checked(client, "jobs")["structuredContent"]["jobs"] == []
        assert checked(client, "inspect", path=str(tmp_path))["structuredContent"][
            "directory"
        ] == str(tmp_path)
