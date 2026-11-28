"""Native MCP conversations use files and durable tomography jobs on the worker."""

import base64
from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
import time

import h5py
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
