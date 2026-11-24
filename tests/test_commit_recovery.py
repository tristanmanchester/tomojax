"""Interrupt real commit processes at filesystem transitions, using an isolated uid home."""

import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

pytestmark = pytest.mark.surface

# sitecustomize also runs inside the GUI's exec'd helper. Keep fault injection
# outside production code and exercise exactly the same on-disk protocol.
INJECT = """
import os, pwd, signal
from pathlib import Path
from types import SimpleNamespace
pwd.getpwuid = lambda uid: SimpleNamespace(pw_dir=os.environ['TEST_HOME'])
original = Path.replace
step = 0
def replace(path, dest):
    global step
    step += 1
    mode = os.environ.get('FAULT')
    if mode in ('fail', 'pause-fail') and (path.name == 'second' or path.name.endswith('.manifest.json')):
        raise OSError('injected rename failure')
    if mode == 'before-decision' and step == 2:
        os._exit(88)
    result = original(path, dest)
    if mode in ('publish-file', 'publish-directory') and step == 1:
        foreign = Path(os.environ['TEST_HOME'], 'output')
        if mode == 'publish-directory':
            foreign.mkdir()
            foreign = foreign / 'foreign.tif'
        foreign.write_bytes(b'other job')
    if mode == str(step) or (mode == 'rollback' and Path(dest).suffix == '.rollback'):
        os._exit(88)
    if mode in ('pause', 'pause-fail') and step == 1:
        Path(os.environ['TEST_HOME'], 'locked').write_text(str(os.getpid()))
        signal.raise_signal(signal.SIGSTOP)
    return result
Path.replace = replace
if os.environ.get('FAULT', '').startswith('fallback'):
    import ctypes
    original_symbol = ctypes.CDLL.__getattr__
    def symbol(lib, name):
        if name == 'renameat2':
            raise AttributeError(name)
        return original_symbol(lib, name)
    ctypes.CDLL.__getattr__ = symbol
    if os.environ['FAULT'] == 'fallback-crash':
        link = os.link
        def interrupted_link(*args, **kwargs):
            link(*args, **kwargs)
            os._exit(88)
        os.link = interrupted_link
import shutil
rmtree = shutil.rmtree
def remove(path, *a, **kw):
    result = rmtree(path, *a, **kw)
    if os.environ.get('FAULT') == 'cleanup':
        os._exit(88)
    return result
shutil.rmtree = remove
"""
PROGRAM = """
from tomojax.cli._commit import commit, helper, Staged
import msgspec, sys
group = msgspec.json.decode(sys.argv[1], type=list[Staged])
print(msgspec.json.encode((helper if sys.argv[2]=='gui' else commit)(group)).decode())
"""


@pytest.fixture(params=["gui", "supervisor"])
def launch(tmp_path, request):
    (tmp_path / "sitecustomize.py").write_text(INJECT)

    def start(group=(), *, fault=""):
        return subprocess.Popen(
            [sys.executable, "-c", PROGRAM, json.dumps(group), request.param],
            stdout=-1,
            stderr=-1,
            env={
                **os.environ,
                "PYTHONPATH": str(tmp_path),
                "TEST_HOME": str(tmp_path),
                "FAULT": fault,
            },
        )

    return start


def finish(process):
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, stderr
    return json.loads(stdout)


def group_at(root, names, generation):
    group = []
    for name in names:
        stage = root / uuid4().hex / "new" / name
        stage.parent.mkdir(parents=True)
        stage.write_text(generation)
        group.append(dict(destination=str(root / name), staged=str(stage), force=True))
    return group


@pytest.fixture(scope="module")
def slab_plan(tmp_path_factory):
    """Generate real W5 stage plans without letting the worker install their contents."""
    import h5py
    import numpy as np

    root = tmp_path_factory.mktemp("slab-worker")
    raw = root / "raw.h5"
    with h5py.File(raw, "w") as f:
        f["exchange/data"] = np.random.default_rng(5).integers(
            400, 950, size=(12, 4, 8), dtype=np.uint16
        )
        f["exchange/theta"] = np.arange(12) * 15
    with subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", "serve", "--commit", "supervisor"],
        env={**os.environ, "TMPDIR": str(root), "JAX_PLATFORMS": "cpu"},
        stdin=-1,
        stdout=-1,
        text=True,
    ) as proc:
        assert json.loads(proc.stdout.readline())["type"] == "ready"
        serial = 0

        def call(command):
            nonlocal serial
            serial += 1
            proc.stdin.write(json.dumps({"id": str(serial), "command": command}) + "\n")
            proc.stdin.flush()
            while line := proc.stdout.readline():
                row = json.loads(line)
                if row["type"] != "progress":
                    assert row["type"] == "result", row
                    return row["result"]
            pytest.fail("Worker exited while staging")

        try:
            opened = call(
                dict(
                    type="open",
                    path=str(raw),
                    kind="frames",
                    flats=1000,
                    darks=0,
                    detector=dict(nu=8, nv=4, du=1, dv=1),
                    geometry=dict(kind="parallel"),
                )
            )
            source = dict(type="member", handle=opened["value"]["handle"], member="self")

            def stage(path):
                output = call(
                    dict(
                        type="run",
                        operation=dict(
                            type="reconstruct_slabs",
                            method="fbp",
                            source=source,
                            path=str(path),
                            force=True,
                            config=dict(type="FBPConfig", backend="jax"),
                        ),
                    )
                )
                return output["value"]["plan"]

            yield stage
        finally:
            proc.stdin.close()


@pytest.mark.parametrize("fault", ["1", "2", "3", "4", "5", "6", "cleanup", "fail"])
def test_group_decision_survives_crash(launch, tmp_path, fault, slab_plan):
    names = ["volume.nxs", "volume.nxs.manifest.json"]
    group = slab_plan(tmp_path / names[0])
    staged = {Path(item["destination"]).name: Path(item["staged"]).read_bytes() for item in group}
    for name in names:
        (tmp_path / name).write_text("old")
    failed = launch(group, fault=fault)
    failed.communicate(timeout=15)
    assert failed.returncode != 0
    if fault == "5":
        interrupted = launch(fault="rollback")
        interrupted.communicate(timeout=15)
        assert interrupted.returncode != 0
    committed = fault in {"6", "cleanup"}
    report = finish(launch())
    if fault != "fail":
        assert report["confirmed" if committed else "restored"] == [
            str(tmp_path / n) for n in names
        ]
    assert all(
        (tmp_path / name).read_bytes() == (staged[name] if committed else b"old") for name in names
    )
    assert all(not Path(item["staged"]).parent.parent.exists() for item in group)
    assert finish(launch()) == {}


def test_interrupted_tiff_export_then_overlapping_writers(launch, tmp_path):
    dest = tmp_path / "slices"
    dest.mkdir()
    stage = tmp_path / "work/new/slices"
    stage.mkdir(parents=True)
    for name in ("x", "y"):
        (dest / name).write_text("old")
        (stage / name).write_text("A")
    a = launch([dict(destination=str(dest), staged=str(stage), force=True)], fault="3")
    a.communicate(timeout=15)
    assert a.returncode != 0
    b = launch(group_at(dest, ["x"], "B"))
    c = launch(group_at(dest, ["y"], "C"))
    reports = [finish(b), finish(c)]
    assert sorted(next(r["restored"] for r in reports if r)) == sorted(
        str(dest / n) for n in ("x", "y")
    )
    assert finish(launch()) == {}
    assert [(dest / name).read_text() for name in ("x", "y")] == ["B", "C"]


@pytest.mark.parametrize("fail", [False, True])
def test_waiting_writer_follows_first_decision(tmp_path, launch, fail):
    import signal
    import time

    names = ["first", "second"]
    for name in names:
        (tmp_path / name).write_text("old")
    a = launch(group_at(tmp_path, names, "A"), fault="pause-fail" if fail else "pause")
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "locked").exists():
            assert a.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        pid = int((tmp_path / "locked").read_text())
        b = launch(group_at(tmp_path, ["first"], "B"))
        os.kill(pid, signal.SIGCONT)
        if fail:
            _, error = a.communicate(timeout=10)
            assert a.returncode and b"Restored" in error
        else:
            finish(a)
        finish(b)
        assert (tmp_path / "first").read_text() == "B"
        assert (tmp_path / "second").read_text() == ("old" if fail else "A")
    finally:
        a.kill()
        a.wait()


def test_recover_new_tiff_directory_containing_another_stage(launch, tmp_path):
    dest = tmp_path / "slices"
    stage = tmp_path / "work/new/slices"
    stage.mkdir(parents=True)
    (stage / "slice_000.tif").write_bytes(b"old job")
    process = launch(
        [dict(destination=str(dest), staged=str(stage), force=False)], fault="before-decision"
    )
    process.communicate(timeout=15)
    assert process.returncode != 0 and dest.is_dir()

    second = group_at(dest, ["other"], "new job")
    restored = finish(launch(second))
    assert restored["restored"] == [str(dest / "slice_000.tif")]
    assert list(dest.iterdir()) == [dest / "other"]
    assert (dest / "other").read_text() == "new job"
    assert finish(launch()) == {}


@pytest.mark.parametrize("directory", [False, True])
def test_destination_appears_after_journal(launch, tmp_path, directory):
    group = group_at(tmp_path, ["output"], "our job")
    group[0]["force"] = False
    stage = Path(group[0]["staged"])
    if directory:
        stage.unlink()
        stage.mkdir()
        (stage / "slice_000.tif").write_bytes(b"our job")
    process = launch(group, fault="publish-directory" if directory else "publish-file")
    _, error = process.communicate(timeout=15)
    assert process.returncode and b"destination appeared" in error
    dest = tmp_path / "output"
    if directory:
        assert list(dest.iterdir()) == [dest / "foreign.tif"]
        dest = dest / "foreign.tif"
    assert dest.read_bytes() == b"other job"
    assert finish(launch()) == {}


def test_recovery_preserves_replaced_output(launch, tmp_path):
    group = group_at(tmp_path, ["output"], "our job")
    group[0]["force"] = False
    process = launch(group, fault="before-decision")
    process.communicate(timeout=15)
    assert process.returncode != 0
    dest = tmp_path / "output"
    assert dest.read_text() == "our job"
    foreign = tmp_path / "other"
    foreign.write_text("other job")
    foreign.replace(dest)
    finish(launch())
    assert dest.read_text() == "other job"


def test_link_fallback_and_interrupted_install(launch, tmp_path):
    for fault in ("fallback-crash", "fallback"):
        group = group_at(tmp_path, ["output"], "ours")
        group[0]["force"] = False
        proc = launch(group, fault=fault)
        if fault == "fallback-crash":
            proc.communicate(timeout=15)
            assert proc.returncode != 0 and (tmp_path / "output").read_text() == "ours"
            finish(launch())
            assert not (tmp_path / "output").exists()
        else:
            finish(proc)
            assert (tmp_path / "output").read_text() == "ours"

    stage = tmp_path / "work/new/slices"
    stage.mkdir(parents=True)
    dest = tmp_path / "slices"
    proc = launch([dict(destination=str(dest), staged=str(stage), force=False)], fault="fallback")
    _, error = proc.communicate(timeout=15)
    assert proc.returncode and b"directory install requires renameat2" in error
    assert not dest.exists()
