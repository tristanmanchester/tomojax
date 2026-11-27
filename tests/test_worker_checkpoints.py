"""A real worker runs checkpointed jobs that a cancel or SIGTERM stops and a rerun resumes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import tomojax as tj

pytestmark = pytest.mark.surface


class Worker:
    """``tomojax serve`` spoken to over its pipes, one request at a time."""

    def __init__(self, cwd: Path) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tomojax.cli.main", "serve"],
            cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )  # fmt: skip
        self.buffer = b""
        assert self.read()["type"] == "ready"

    def read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        while b"\n" not in self.buffer:
            assert select.select([self.proc.stdout], [], [], 120)[0], "worker output stalled"
            data = os.read(self.proc.stdout.fileno(), 65536)
            assert data, "unexpected worker EOF"
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def send(self, rid: str, command: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"id": rid, "command": command}).encode() + b"\n")

    def run(self, rid: str, operation: dict[str, Any], stop: str | None = None) -> dict[str, Any]:
        """Run ``operation``; ``stop`` ("cancel" or "sigterm") ends it once its solve advances."""
        self.send(rid, {"type": "run", "operation": operation})
        solve = None
        while True:
            row = self.read()
            if row.get("request") != rid:
                continue
            if row["type"] != "progress":
                return row
            record = row["record"]
            if record["event"] == "started" and record["unit"] in ("iterations", "slabs", "steps"):
                solve = record["task"]
            elif stop and record["event"] == "advanced" and record["task"] == solve:
                if stop == "cancel":
                    self.send("c", {"type": "control", "action": "cancel", "target": rid})
                else:
                    self.proc.send_signal(signal.SIGTERM)
                stop = None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.send("q", {"type": "shutdown"})
            self.proc.wait(30)


@pytest.fixture
def scan_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.nxs"
    tj.save(path, tj.simulate("shepp", size=16, n_views=24, seed=1).scan)
    return path


def _reconstruct(scan: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    source = {"type": "path", "path": str(scan)}
    operation = {"type": "reconstruct", "source": source, "method": "sirt", "iterations": 300}
    return operation if checkpoint is None else {**operation, "checkpoint": str(checkpoint)}


def _array(worker: Worker, reply: dict[str, Any], member: str) -> np.ndarray:
    """A result member's values, read through the worker."""
    rid = f"read-{reply['request']}-{member}"
    worker.send(rid, {"type": "read", "source": reply["result"]["value"][member]["storage"]})
    while (row := worker.read()).get("request") != rid or row["type"] == "progress":
        pass
    return np.load(row["result"]["value"]["array"]["storage"]["path"])


@pytest.mark.parametrize("stop", ["cancel", "sigterm"])
def test_a_stopped_job_names_its_checkpoint_and_a_rerun_resumes(
    tmp_path: Path, scan_file: Path, stop: str
) -> None:
    run = tmp_path / "run"
    worker = Worker(tmp_path)
    try:
        whole = _array(worker, worker.run("w", _reconstruct(scan_file)), "volume")
        stopped = worker.run("s", _reconstruct(scan_file, run), stop=stop)
        if stop == "sigterm":
            assert worker.proc.wait(60) == 0  # shut down after the job saved
    finally:
        worker.close()
    error = stopped["error"]
    assert stopped["type"] == "error"
    assert error["kind"] == "canceled"
    assert error["checkpoint"] == str(run)
    assert str(run) in error["message"]

    worker = Worker(tmp_path)
    try:
        worker.send("i", {"type": "inspect", "path": str(run)})
        while (reply := worker.read()).get("request") != "i" or reply["type"] == "progress":
            pass
        assert reply["type"] == "result", reply
        report = reply["result"]["value"]["report"]["checkpoint"]
        assert report["run"]["operation"] == "reconstruct"
        assert report["run"]["method"] == "sirt"
        assert report["request"]["operation"]["parameters"]["checkpoint"] == str(run)
        assert report["request"]["operation"]["parameters"]["source"]["path"] == str(scan_file)
        assert report["saved"] is not None
        resumed = worker.run("r", _reconstruct(scan_file, run))
        assert resumed["type"] == "result", resumed
        np.testing.assert_array_equal(_array(worker, resumed, "volume"), whole)
    finally:
        worker.close()
    assert not run.exists()


def test_slabs_from_raw_frames_resume_into_their_output(tmp_path: Path) -> None:
    run = tmp_path / "run"
    code = (
        "scan = tj.simulate('shepp', size=16, n_views=24, seed=2).scan\n"
        "counts = (1000 * np.exp(-np.asarray(scan.projections))).astype(np.float32)\n"
        "register(tj.Frames(counts, scan.geometry, white_level=1000))"
    )
    worker = Worker(tmp_path)
    try:
        worker.send("p", {"type": "python", "code": code})
        while (made := worker.read()).get("request") != "p" or made["type"] == "progress":
            pass
        assert made["type"] == "result", made
        (frames,) = made["result"]["value"]["handles"]
        source = {"type": "member", "handle": frames, "member": "self"}

        def slabs(path: str, checkpoint: Path | None = None) -> dict[str, Any]:
            operation = {
                "type": "reconstruct_slabs",
                "source": source,
                "path": str(tmp_path / path),
                "method": "sirt",
                "config": {"type": "SIRTConfig", "iterations": 300},
            }
            return operation if checkpoint is None else {**operation, "checkpoint": str(checkpoint)}

        assert worker.run("w", slabs("whole.nxs"))["type"] == "result"
        stopped = worker.run("s", slabs("part.nxs", run), stop="cancel")
        assert stopped["type"] == "error", stopped
        assert stopped["error"]["checkpoint"] == str(run)
        assert not (tmp_path / "part.nxs").exists()
        assert (run / "volume.npy").is_file()
        assert worker.run("r", slabs("part.nxs", run))["type"] == "result"
    finally:
        worker.close()
    np.testing.assert_array_equal(
        tj.load_reconstruction(tmp_path / "part.nxs").volume,
        tj.load_reconstruction(tmp_path / "whole.nxs").volume,
    )
    assert not run.exists()


def test_a_canceled_alignment_resumes_to_the_uninterrupted_fit(tmp_path: Path) -> None:
    scan = tmp_path / "scan.nxs"
    tj.save(scan, tj.simulate("shepp", size=16, n_views=24, translation_sigma=0.5, seed=3).scan)
    run = tmp_path / "run"
    source = {"type": "path", "path": str(scan)}
    align = {"type": "align", "source": source, "fit": "translations", "iterations": 6}
    worker = Worker(tmp_path)
    try:
        whole = worker.run("w", align)
        stopped = worker.run("s", {**align, "checkpoint": str(run)}, stop="cancel")
        assert stopped["type"] == "error"
        assert stopped["error"]["checkpoint"] == str(run)
        resumed = worker.run("r", {**align, "checkpoint": str(run)})
        assert resumed["type"] == "result", resumed
        poses = [_array(worker, r, "poses") for r in (whole, resumed)]
    finally:
        worker.close()
    np.testing.assert_array_equal(poses[1], poses[0])
    assert not run.exists()
