"""One scenario table, run through every surface: ``tj.*``, the CLI, ``tomojax serve`` and MCP.

Each scenario names an operation, its input file and its options in the Python
vocabulary. Every surface must accept the same requests with the same defaults and
reach the same numbers, and refuse the same requests for the same parameter. Each
surface's result is read back from the file it saved, with the Python loaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import tomojax as tj
from tomojax.cli.main import main as cli

from ._helpers import write_raw_nxtomo

pytestmark = pytest.mark.surface


@dataclass(frozen=True)
class Scenario:
    """An operation on one input file, with options as ``tj.<operation>`` keywords."""

    name: str
    operation: str
    source: str | None
    options: dict[str, Any] = field(default_factory=dict)
    # The parameter a surface must name when it refuses the request; None to accept it.
    refused: str | None = None


SCENARIOS = [
    Scenario("simulate", "simulate", None, {"phantom": "sphere", "size": 10, "n_views": 12}),
    Scenario(
        "simulate laminography",
        "simulate",
        None,
        {"phantom": "random_shapes", "size": 8, "n_views": 10, "geometry": "lamino", "tilt": 25.0},
    ),
    # Defaults: a well-sampled axial scan takes Fourier inversion, a short cone arc CGLS.
    Scenario("reconstruct by default", "reconstruct", "parallel.nxs"),
    Scenario("reconstruct a short cone arc", "reconstruct", "cone-arc.nxs"),
    Scenario(
        "reconstruct with cgls", "reconstruct", "parallel.nxs", {"method": "cgls", "iterations": 3}
    ),
    Scenario("fit the center", "align", "parallel.nxs", {"method": "opposite-views"}),
    Scenario("correct raw frames", "correct", "raw.nxs"),
    Scenario("inspect", "inspect", "parallel.nxs"),
    Scenario(
        "auto takes no options", "reconstruct", "parallel.nxs", {"iterations": 3}, "iterations"
    ),
    Scenario(
        "fbp takes no iterations",
        "reconstruct",
        "parallel.nxs",
        {"method": "fbp", "iterations": 3},
        "iterations",
    ),
    Scenario(
        "cgls takes no tv_weight",
        "reconstruct",
        "parallel.nxs",
        {"method": "cgls", "tv_weight": 0.1},
        "tv_weight",
    ),
    Scenario("an unknown fit", "align", "parallel.nxs", {"fit": "bogus"}, "fit"),
    Scenario(
        "opposite views of a tilted axis",
        "align",
        "lamino.nxs",
        {"method": "opposite-views"},
        "method",
    ),
    Scenario("an empty simulation", "simulate", None, {"size": 0}, "size"),
    Scenario("binning below one", "correct", "raw.nxs", {"binning": 0}, "binning"),
]


@dataclass(frozen=True)
class Refused:
    """A refusal: the parameter path a surface reports, or None for the CLI's message."""

    parameter: str | None
    message: str


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("inputs")
    # 20 directions over a full turn: Fourier inversion, and opposite views for the center.
    tj.save(directory / "parallel.nxs", tj.simulate("shepp", size=12, n_views=40, rotation=360))
    tj.save(directory / "lamino.nxs", tj.simulate("shepp", size=8, n_views=8, geometry="lamino"))
    arc = tj.simulate("sphere", size=10, n_views=20, geometry="cone", rotation=120.0)
    tj.save(directory / "cone-arc.nxs", arc)
    write_raw_nxtomo(directory / "raw.nxs")
    return directory


# Python ----------------------------------------------------------------------------------


def python(scenario: Scenario, inputs: Path, output: Path) -> object:
    try:
        if scenario.operation == "simulate":
            result = tj.simulate(**scenario.options)
        elif scenario.operation == "correct":
            result = tj.load_frames(inputs / scenario.source).corrected(**scenario.options)
        elif scenario.operation == "inspect":
            return tj.inspect(inputs / scenario.source)
        else:
            operation = getattr(tj, scenario.operation)
            result = operation(tj.load(inputs / scenario.source), **scenario.options)
    except tj.InvalidParameter as error:
        return Refused(error.path, str(error))
    tj.save(output, result)
    return read(scenario, output)


# The command line --------------------------------------------------------------------------


def command_line(scenario: Scenario, inputs: Path, output: Path, capsys) -> object:
    argv = [scenario.operation]
    if scenario.source is not None:
        argv.append(str(inputs / scenario.source))
    argv += ["--json"] if scenario.operation == "inspect" else ["-o", str(output)]
    for key, value in scenario.options.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    try:
        status = cli([*argv, "--progress", "none"])
    except SystemExit as exit_:  # argparse refuses before the command runs
        status = exit_.code
    printed, message = capsys.readouterr()
    # 2 for a malformed command line, 1 for a request the input does not allow.
    if status != 0:
        return Refused(None, message)
    return json.loads(printed) if scenario.operation == "inspect" else read(scenario, output)


# tomojax serve ---------------------------------------------------------------------------


class Worker:
    """A JSON Lines client of a real ``tomojax serve`` process."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.buffer = b""
        self.ids = (f"r{n}" for n in itertools.count())
        assert self.read()["type"] == "ready"

    def read(self) -> dict[str, Any]:
        while b"\n" not in self.buffer:
            assert select.select([self.process.stdout], [], [], 120)[0], "the worker stalled"
            data = os.read(self.process.stdout.fileno(), 1 << 16)
            assert data, "the worker exited"
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        rid = next(self.ids)
        self.process.stdin.write(json.dumps({"id": rid, "command": command}).encode() + b"\n")
        self.process.stdin.flush()
        while True:
            row = self.read()
            if row.get("request") == rid and row["type"] in ("result", "error", "protocol_error"):
                return row


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("worker")
    with subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", "serve"],
        env={**os.environ, "TMPDIR": str(scratch)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    ) as process:
        try:
            client = Worker(process)
            yield client
            assert client.request({"type": "shutdown"})["type"] == "result"
            assert process.wait(30) == 0
        finally:
            if process.poll() is None:
                process.kill()


def serve(scenario: Scenario, inputs: Path, output: Path, worker: Worker) -> object:
    if scenario.operation == "inspect":
        reply = worker.request({"type": "inspect", "path": str(inputs / scenario.source)})
        assert reply["type"] == "result", reply
        return reply["result"]["value"]["report"]
    operation = {"type": scenario.operation, **scenario.options}
    if scenario.source is not None:
        operation["source"] = {"type": "path", "path": str(inputs / scenario.source)}
    reply = worker.request({"type": "run", "operation": operation})
    if reply["type"] == "error" and reply["error"]["kind"] == "invalid_parameter":
        return Refused(reply["error"]["path"], reply["error"]["message"])
    assert reply["type"] == "result", reply
    handle = reply["result"]["value"]["handle"]
    source = {"type": "member", "handle": handle, "member": "self"}
    saved = worker.request(
        {"type": "run", "operation": {"type": "save", "source": source, "path": str(output)}}
    )
    assert saved["type"] == "result", saved
    return read(scenario, output)


# MCP -------------------------------------------------------------------------------------


class Agent:
    """An MCP client of a real ``tomojax mcp`` process, over its stdio."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.ids = itertools.count(1)
        self.rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def send(self, row: dict[str, Any]) -> None:
        self.process.stdin.write(json.dumps(row) + "\n")
        self.process.stdin.flush()

    def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        rid = next(self.ids)
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            assert select.select([self.process.stdout], [], [], 120)[0], "the server stalled"
            row = json.loads(self.process.stdout.readline())
            if row.get("id") == rid:
                assert "error" not in row, row
                return row["result"]

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        return self.rpc("tools/call", {"name": name, "arguments": arguments})


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    directory = tmp_path_factory.mktemp("agent")
    env = {**os.environ, "TMPDIR": str(directory), "CLAUDE_PROJECT_DIR": str(directory)}
    with subprocess.Popen(
        [sys.executable, "-m", "tomojax.cli.main", "mcp"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            yield Agent(process)
            process.stdin.close()
            assert process.wait(30) == 0
        finally:
            if process.poll() is None:
                process.kill()


def mcp(scenario: Scenario, inputs: Path, output: Path, agent: Agent) -> object:
    if scenario.operation == "inspect":
        reply = agent.call("inspect", path=str(inputs / scenario.source), detail="full")
        assert not reply["isError"], reply["content"]
        return reply["structuredContent"]
    arguments = dict(scenario.options)
    if scenario.source is not None:
        kind = "frames" if scenario.operation == "correct" else "scan"
        opened = agent.call("open", path=str(inputs / scenario.source), kind=kind)
        assert not opened["isError"], opened["content"]
        arguments["source"] = opened["structuredContent"]["handle"]
    reply = agent.call(scenario.operation, **arguments)
    if reply["isError"] and reply["structuredContent"]["kind"] == "invalid_parameter":
        return Refused(reply["structuredContent"]["path"], reply["structuredContent"]["message"])
    assert not reply["isError"], reply["content"]
    saved = agent.call("save", source=reply["structuredContent"]["handle"], path=str(output))
    assert not saved["isError"], saved["content"]
    return read(scenario, output)


# Comparison ------------------------------------------------------------------------------


def read(scenario: Scenario, path: Path) -> dict[str, Any]:
    """What a saved result says, through the Python loaders."""
    if scenario.operation == "simulate":
        simulation = tj.load_simulation(path)
        return {**_scan(simulation.scan), "phantom": simulation.phantom}
    if scenario.operation == "reconstruct":
        recon = tj.load_reconstruction(path)
        return {
            "volume": np.asarray(recon.volume),
            "method": recon.method,
            "automatic": recon.info.get("automatic", {}).get("method"),
            "grid": recon.grid,
        }
    if scenario.operation == "align":
        alignment = tj.load_alignment(path)
        return {**_scan(alignment.scan), "fit": alignment.info["fit"]}
    return _scan(tj.load(path))


def _scan(scan: tj.Scan) -> dict[str, Any]:
    return {
        "projections": np.asarray(scan.projections),
        "geometry": type(scan.geometry).__name__,
        "grid": scan.grid,
        "detector": scan.detector,
        "angles": scan.angles,
    }


def _inspection(report: dict[str, Any]) -> dict[str, Any]:
    """The facts every inspection report gives of a saved scan."""
    return {
        "projection": {k: report["projection"][k] for k in ("shape", "dtype", "path")},
        "angles": {k: report["angles"][k] for k in ("count", "units")},
        "geometry": report["geometry"]["type"],
        "detector": report["detector_metadata"],
        "valid": report["valid"],
    }


def assert_same(actual: object, expected: object, where: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for key, value in expected.items():
            assert_same(actual[key], value, f"{where}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        # Each surface runs the same code; only reduction order may differ between processes.
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, err_msg=where)
    else:
        assert actual == expected, where


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_surface_agrees(scenario, inputs, worker, agent, tmp_path, capsys):
    outcomes = {
        "python": python(scenario, inputs, tmp_path / "python.nxs"),
        "cli": command_line(scenario, inputs, tmp_path / "cli.nxs", capsys),
        "serve": serve(scenario, inputs, tmp_path / "serve.nxs", worker),
        "mcp": mcp(scenario, inputs, tmp_path / "mcp.nxs", agent),
    }
    if scenario.refused is not None:
        for surface, outcome in outcomes.items():
            assert isinstance(outcome, Refused), f"{surface} accepted {scenario.name}"
            assert not (tmp_path / f"{surface}.nxs").exists(), f"{surface} wrote a refused result"
            if outcome.parameter is None:  # the CLI names the keyword or its flag
                flag = f"--{scenario.refused.replace('_', '-')}"
                named = scenario.refused in outcome.message or flag in outcome.message
            else:
                named = outcome.parameter == scenario.refused
            assert named, f"{surface} did not name {scenario.refused}: {outcome}"
        return
    if scenario.operation == "inspect":
        outcomes = {surface: _inspection(report) for surface, report in outcomes.items()}
    expected = outcomes.pop("python")
    for surface, outcome in outcomes.items():
        assert not isinstance(outcome, Refused), f"{surface} refused {scenario.name}: {outcome}"
        assert_same(outcome, expected, surface)


@pytest.mark.parametrize("random", [False, True], ids=["uniform", "noise"])
def test_export_plan_and_receipt_across_surfaces(tmp_path, worker, agent, capsys, random):
    import tifffile

    grid = tj.Grid(32, 18, 7, 0.03, 0.04, 0.05)
    geom = tj.ParallelGeometry(grid, tj.Detector(8, 4, 1, 1), np.arange(4.0))
    volume = np.random.default_rng(19).random(grid.shape, dtype=np.float32)
    if not random:
        volume.fill(0.25)
    source = tj.Reconstruction(volume, geom, "fbp", {}, length_unit="mm")
    native, target = tmp_path / "input.nxs", tmp_path / "out.tif"
    tj.save(native, source)
    options = dict(dtype="uint8", range=[0.0, 1.0], axis="y", crop=[[1, 5], None, None], binning=2)
    plan = tj.export(target, source, **options, compress=True, dry_run=True)
    cli_args = [
        "export",
        str(native),
        "-o",
        str(target),
        "--dtype",
        "uint8",
        "--range",
        "0",
        "1",
        "--axis",
        "y",
        "--crop",
        "1:5,:,:",
        "--binning",
        "2",
    ]
    opened = worker.request(dict(type="open", path=str(native), kind="reconstruction"))
    assert opened["type"] == "result", opened
    member = dict(type="member", handle=opened["result"]["value"]["handle"], member="volume")
    op = dict(type="export", source=member, path=str(target), compress=True, **options)
    handle = agent.call("open", path=str(native), kind="reconstruction")["structuredContent"][
        "handle"
    ]

    def mcp_export(**extra):
        reply = agent.call(
            "export", source=handle, path=str(target), compress=True, **options, **extra
        )
        assert not reply["isError"], reply["content"]
        return reply["structuredContent"]

    assert cli([*cli_args, "--compress", "--dry-run"]) == 0
    plans = [
        json.loads(capsys.readouterr().out),
        worker.request(dict(type="validate", operation=op))["result"]["value"]["details"],
        mcp_export(dry_run=True),
    ]
    for other in plans:
        for k in plan.keys() - {"free_bytes"}:
            assert other[k] == plan[k], k
    assert not target.exists()
    for flag, codec in [("--no-compress", "NONE"), ("--compress", "LZW")]:
        assert cli([*cli_args, flag, "--force"]) == 0
        cli_receipt = json.loads(capsys.readouterr().out)
        assert cli_receipt["bytes_written"] == target.stat().st_size
        with tifffile.TiffFile(target) as f:
            assert f.pages[0].compression.name == codec
    raw = worker.request(dict(type="run", operation={**op, "force": True}))
    assert raw["type"] == "result", raw
    worker_receipt = raw["result"]["info"]["summary"]
    assert worker_receipt["bytes_written"] == target.stat().st_size
    mcp_receipt = mcp_export(force=True)
    assert mcp_receipt["bytes_written"] == target.stat().st_size
    for receipt in (worker_receipt, mcp_receipt, cli_receipt):
        for k in plan.keys() - {"paths", "free_bytes", "destination_exists"}:
            assert receipt[k] == plan[k], k
        assert receipt["clipped"] == {"below": 0, "above": 0}
