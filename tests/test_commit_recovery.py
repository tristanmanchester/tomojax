"""Interrupt real commit processes at filesystem transitions, using an isolated uid home."""

import json
import os
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
    if mode in ('fail', 'pause-fail') and path.name == 'second':
        raise OSError('injected rename failure')
    result = original(path, dest)
    if mode == str(step) or (mode == 'rollback' and path.name.startswith('backup-')):
        os._exit(88)
    if mode in ('pause', 'pause-fail') and step == 1:
        Path(os.environ['TEST_HOME'], 'locked').write_text(str(os.getpid()))
        signal.raise_signal(signal.SIGSTOP)
    return result
Path.replace = replace
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


@pytest.mark.parametrize("fault", ["1", "2", "3", "4", "5", "6", "cleanup", "fail"])
def test_group_decision_survives_crash(launch, tmp_path, fault):
    names = ["first", "second"]
    group = group_at(tmp_path, names, "new")
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
    assert all((tmp_path / name).read_text() == ("new" if committed else "old") for name in names)
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
