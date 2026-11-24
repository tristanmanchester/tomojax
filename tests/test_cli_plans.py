"""Plans people read: one resolution per run, its header, dry runs and memory hints."""

from __future__ import annotations

import json

import numpy as np
import pytest

import tomojax as tj
from tomojax import progress
from tomojax.cli._plan_view import plan_text
from tomojax.cli.main import main
from tomojax.wire.resources import Placement, Snapshot

pytestmark = pytest.mark.surface


@pytest.fixture(scope="module")
def scan_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("scan") / "scan.nxs"
    assert main(["simulate", "-o", str(path), "--size", "16", "--progress", "none"]) == 0
    return path


def test_plan_text_names_the_method_grid_and_estimated_memory(scan_path):
    from tomojax._planning import resolve_reconstruction
    from tomojax.cli._plans import plan_product

    scan = tj.load(scan_path)
    work = resolve_reconstruction(scan, "fista", region="cyl", iterations=7)
    unknown = Placement(
        ("gpu:0",),
        "slabbed",
        3 * 2**20,
        5 * 2**30,
        (Snapshot(0, "gpu:0", "jax", None, None, None),),
    )
    rows = plan_text(scan, work, unknown).splitlines()
    assert rows[0] == "method      fista: Explicit method selection."
    assert rows[1] == "iterations  7"
    assert rows[2].startswith("grid        ") and "voxels at 1 scan units" in rows[2]
    assert rows[3] == "region      cyl, zeroed outside the cylinder"
    assert rows[4] == (
        "placement   estimated: slabbed on gpu:0; needs host 3.0 MiB, device 5.0 GiB; "
        "available unknown"
    )
    assert plan_product(scan, work).method == "fista"


def _count_resolutions(monkeypatch):
    from tomojax import _planning

    calls = []
    resolve = _planning.resolve_reconstruction

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(_planning, "resolve_reconstruction", counted)
    return calls


def test_a_run_resolves_once_and_says_what_it_will_do(monkeypatch, tmp_path, scan_path, capsys):
    calls = _count_resolutions(monkeypatch)
    argv = ["reconstruct", str(scan_path), "-o", str(tmp_path / "r.nxs"), "--method", "sirt"]
    assert main([*argv, "--iterations", "2", "--progress", "json"]) == 0
    assert len(calls) == 1
    notes = [
        r["message"]
        for r in map(json.loads, capsys.readouterr().out.splitlines())
        if r["event"] == "note"
    ]
    assert any(n.startswith("reconstruct: sirt, 2 iterations · 16x16x16 at 1 scan units · ") for n in notes)


def test_running_out_of_memory_says_what_the_plan_needed(monkeypatch, tmp_path, scan_path, capsys):
    from tomojax import _planning, _workflow

    def exhausted(*_args, **_kwargs):
        raise MemoryError("out of memory allocating the volume")

    monkeypatch.setattr(_workflow, "reconstruct_resolved", exhausted)
    argv = ["reconstruct", str(scan_path), "-o", str(tmp_path / "r.nxs"), "--region", "cyl"]
    assert main(argv) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[0] == "tomojax reconstruct: error: out of memory allocating the volume"
    assert err[1].startswith("  hint: the plan estimated ")
    assert "--region cyl" not in err[1] and "a smaller --grid" in err[1]
    # Before any plan exists there is nothing to say about memory.
    monkeypatch.setattr(_planning, "resolve_reconstruction", exhausted)
    assert main([*argv, "--force"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "tomojax reconstruct: error: out of memory allocating the volume"
    ]


def _tree(events):
    """(name, label, depth) of every task under the outer align task, in order."""
    depth, rows = {}, []
    for event in events:
        outer = isinstance(event, progress.Started) and event.name == "align" and not depth
        if isinstance(event, progress.Started) and (outer or event.parent in depth):
            depth[event.task] = 0 if event.parent not in depth else depth[event.parent] + 1
            rows.append((event.name, event.label, depth[event.task]))
    return rows


def test_the_cli_alignment_has_the_task_tree_of_tj_align(tmp_path, scan_path, capsys):
    events = []
    with progress.watch(events.append):
        tj.align(tj.load(scan_path), iterations=1, image_iterations=2)
    argv = ["align", str(scan_path), "-o", str(tmp_path / "a.nxs"), "--progress", "json"]
    assert main([*argv, "--iterations", "1", "--image-iterations", "2"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    started = [
        progress.Started(
            r["task"], r["parent"], r["name"], r["label"], r["total"], r["unit"], r["time"]
        )
        for r in records
        if r["event"] == "started"
    ]
    library = _tree(events)
    assert library and _tree(started) == library


def test_alignment_previews_are_zero_outside_the_cylinder_at_every_level(scan_path):
    from tomojax._smooth_alignment import resolve_alignment

    scan = tj.load(scan_path)
    radius = resolve_alignment(scan, region="cyl").cylinder.radius
    planes = []

    def keep(event):
        if isinstance(event, progress.Preview) and "z" in event.arrays:
            planes.append(event.arrays["z"])

    with progress.watch(keep, previews=True):
        tj.align(scan, region="cyl", iterations=1, image_iterations=2)
    shapes = {plane.data.shape for plane in planes}
    assert len(shapes) >= 2  # a coarse level and the final one
    for plane in planes:
        rows, columns = plane.axes
        y = rows.coordinates.origin + np.arange(plane.data.shape[0]) * rows.coordinates.spacing
        x = (
            columns.coordinates.origin
            + np.arange(plane.data.shape[1]) * columns.coordinates.spacing
        )
        outside = np.hypot(x[None, :], y[:, None]) > radius * (1 + 1e-9)
        assert outside.any() and not plane.data[outside].any()


def test_a_long_path_survives_the_inspect_report_whole(tmp_path):
    import subprocess
    import sys

    deep = tmp_path / ("a_rather_long_directory_name_" * 4) / "scan_with_a_long_name.nxs"
    deep.parent.mkdir()
    assert main(["simulate", "-o", str(deep), "--size", "8", "--progress", "none"]) == 0
    process = subprocess.run(
        [sys.executable, "-m", "tomojax.cli.main", "inspect", str(deep)],
        capture_output=True,
        text=True,
        check=True,
        env={"COLUMNS": "40", "PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu"},
    )
    lines = process.stdout.splitlines()
    assert lines[:2] == ["File", f"  path: {deep}"]
    assert lines[-1] == "Valid: yes"
    assert "\x1b[" not in process.stdout
