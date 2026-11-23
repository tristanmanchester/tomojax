"""Progress is a record of completed work delivered on the operation's thread."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax import progress
from tomojax.recon import CGLSConfig


def validate_tree(events, *, interrupted=False):
    """Reconstruct the active stack using only public events."""
    stack, seen, counts = [], set(), {}
    previous = 0.0
    for event in events:
        assert event.time >= previous
        previous = event.time
        if isinstance(event, progress.Started):
            assert event.task not in seen
            assert event.parent == (stack[-1] if stack else None)
            stack.append(event.task)
            seen.add(event.task)
            counts[event.task] = 0
        elif isinstance(event, progress.Note):
            assert event.task == (stack[-1] if stack else None)
        elif isinstance(event, progress.Advanced):
            assert event.task in stack
            assert counts[event.task] <= event.done
            assert event.total is None or event.done <= event.total
            counts[event.task] = event.done
        else:
            assert isinstance(event, progress.Finished)
            assert event.task == stack.pop()
    assert interrupted or not stack


def scan():
    grid = tj.Grid(12, 10, 2, 1, 1, 1)
    detector = tj.Detector(14, 2, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 13, endpoint=False))
    truth = np.zeros((2, 10, 12), np.float32)
    truth[:, 2:8, 3:8] = 1
    truth[0, 3, 4] = 2
    return tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)


@pytest.mark.parametrize("kind", [progress.Started, progress.Advanced, progress.Finished])
def test_handler_exception_is_the_last_delivery(kind):
    events = []
    failure = RuntimeError("out of memory")

    def handler(event):
        events.append(event)
        if isinstance(event, kind):
            raise failure

    subject = scan()
    with progress.watch(handler), pytest.raises(RuntimeError) as caught:
        tj.reconstruct(subject, "cgls", config=CGLSConfig(backend="jax", iterations=3))
    assert caught.value is failure
    validate_tree(events, interrupted=True)
    assert isinstance(events[-1], kind)


@pytest.mark.parametrize("other_thread", [False, True])
def test_chunk_stops_at_completed_work(other_thread):
    # check-public-imports: allow-private
    from tomojax.recon._chunks import run_chunks

    token = progress.Control()
    completed = []
    events = []
    owner = get_ident()

    @jax.jit
    def updates(state, end):
        return jax.lax.fori_loop(state, end, lambda _, k: k + 1, state)

    def advance(state, end):
        state = updates(state, end)
        completed.append(state)
        return state

    def handler(event):
        assert get_ident() == owner
        events.append(event)
        if isinstance(event, progress.Advanced):
            if other_thread:
                pool.submit(token.cancel).result()
            else:
                token.cancel()

    with (
        ThreadPoolExecutor(1) as pool,
        progress.watch(handler, control=token),
        pytest.raises(progress.Cancelled),
        progress.task("reconstruct", "counter", 20, "updates"),
    ):
        run_chunks(
            jnp.int32(0),
            20,
            advance,
            lambda s: (int(s), False, {}),
            task=progress.current_task.get(),
        )
    validate_tree(events)
    assert [int(s) for s in completed] == [1]
    assert events[-1].status == "cancelled"


def test_nested_watches_are_thread_local():
    barrier = Barrier(2)

    def observe(label):
        outer, inner = [], []
        with progress.watch(outer.append), progress.task("load", label, None, "views"):
            barrier.wait()
            with (
                progress.watch(inner.append),
                progress.task("load", label + " inner", None, "views"),
            ):
                pass
            with progress.task("correction", label + " child", None, "rows"):
                pass
        for events in (outer, inner):
            validate_tree(events)
        return [
            [e.label for e in events if isinstance(e, progress.Started)]
            for events in (outer, inner)
        ]

    with ThreadPoolExecutor(2) as executor:
        both = list(executor.map(observe, ["a", "b"]))
    assert both == [[["a", "a child"], ["a inner"]], [["b", "b child"], ["b inner"]]]


def test_metric_snapshot_and_strict_json():
    original = {"loss": float("nan"), "value": 3}
    event = progress.Advanced(2, 1, 4, original, 0.5)
    original["value"] = -1
    assert isinstance(event.metrics, MappingProxyType)
    assert progress.to_json(event) == {
        "schema": 2,
        "event": "advanced",
        "task": 2,
        "done": 1,
        "total": 4,
        "metrics": {"loss": None, "value": 3},
        "time": 0.5,
        "eta": None,
    }


@pytest.mark.parametrize("final", [False, True])
def test_streamed_cancellation_on_advance(monkeypatch, final):
    from tomojax.backends import host_stream

    reads = []
    subject = scan()

    class Sources(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    monkeypatch.setattr(host_stream, "_SOURCES", Sources())
    events, observed = [], []
    token = progress.Control()

    def handler(event):
        events.append(event)
        if isinstance(event, progress.Advanced) and ((event.done == 4) == final):
            observed.append(len(reads))
            token.cancel()

    config = CGLSConfig(backend="jax", stream=True, views_per_batch=4, iterations=4, tolerance=0)
    with progress.watch(handler, control=token):
        if final:
            result = tj.reconstruct(subject, "cgls", config=config)
            assert result.info["effective_iterations"] == 4
        else:
            with pytest.raises(progress.Cancelled):
                tj.reconstruct(subject, "cgls", config=config)
    validate_tree(events)
    assert observed == [len(reads)] and reads
    assert events[-1].status == ("done" if final else "cancelled")


@pytest.mark.parametrize("cancel", [False, True])
def test_cli_json_records_arrive_during_reconstruction(tmp_path, cancel):
    import json
    import signal
    import subprocess
    import sys

    from ._helpers import write_projection_dataset

    path, output = tmp_path / "input.nxs", tmp_path / "result.nxs"
    rng = np.random.default_rng(19)
    write_projection_dataset(
        path,
        projections=rng.uniform(size=(48, 4, 48)).astype(np.float32),
        angles=np.linspace(0, 180, 48, endpoint=False),
        grid=tj.Grid(48, 48, 4, 1, 1, 1),
    )
    config = tmp_path / "jax.toml"
    config.write_text('backend="jax"\ntolerance=0.0\n', encoding="utf8")
    command = [
        sys.executable,
        "-m",
        "tomojax.cli.main",
        "reconstruct",
        str(path),
        "-o",
        str(output),
        "--method",
        "cgls",
        "--iterations",
        "12",
        "--config",
        str(config),
        "--progress",
        "json",
    ]
    events = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            record = json.loads(line)
            assert record.pop("schema") == 2
            if record["event"] == "error":
                assert cancel and record["kind"] == "cancelled"
                continue
            if record["event"] == "result":
                assert not cancel and str(output) in record["outputs"]
                continue
            cls = {
                "started": progress.Started,
                "advanced": progress.Advanced,
                "finished": progress.Finished,
                "note": progress.Note,
            }[record.pop("event")]
            events.append(cls(**record))
            if cancel and isinstance(events[-1], progress.Advanced) and events[-1].done == 1:
                process.send_signal(signal.SIGINT)
        assert process.wait(timeout=60) == (130 if cancel else 0)
    validate_tree(events)
    assert [e.name for e in events if isinstance(e, progress.Started)] == (
        ["load", "reconstruct"] if cancel else ["load", "reconstruct", "save", "write"]
    )
    assert events[-1].status == ("cancelled" if cancel else "done")
    assert output.exists() != cancel


@pytest.mark.parametrize("method", ["cgls", "fista"])
def test_watching_preserves_recurrence_and_stop(method):
    """CPU is bit-exact. A deterministic single-view GPU operator has zero ULP spread."""
    from tomojax.recon import FISTAConfig

    grid = tj.Grid(6, 1, 3, 1, 1, 1)
    detector = tj.Detector(6, 3, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, [0.0])
    data = np.linspace(0.2, 1.7, 18, dtype=np.float32).reshape(1, 3, 6)
    subject = tj.Scan(data, geometry)
    cfg = (
        CGLSConfig(backend="jax", iterations=8, tolerance=0.01)
        if method == "cgls"
        else FISTAConfig(
            backend="jax", iterations=8, tv_weight=0.01, lipschitz=2.0, tolerance=0.95, patience=2
        )
    )
    reference = tj.reconstruct(subject, method, config=cfg)
    events = []
    with progress.watch(events.append):
        result = tj.reconstruct(subject, method, config=cfg)
    validate_tree(events)
    np.testing.assert_array_equal(result.volume, reference.volume)
    if method == "fista":
        assert result.info["loss"] == reference.info["loss"]
    else:
        np.testing.assert_allclose(result.volume[:, 0, :], data[0], atol=2e-6, rtol=0)
    key = "termination" if method == "cgls" else "early_stop"
    assert result.info[key] == reference.info[key]
    advances = [event.done for event in events if isinstance(event, progress.Advanced)]
    assert (
        advances[-1]
        == result.info["effective_iterations"]
        == reference.info["effective_iterations"]
    )
    assert len(events) == len(advances) + 2


def test_bar_redraws_on_a_terminal():
    import os
    import pty
    import subprocess
    import sys

    code = """from tomojax import progress
from tomojax.cli._progress import run_command
def operation():
    with progress.task('load','Reading views',5,'views'):
        progress.current_task.get().advance(2,{})
raise SystemExit(run_command('correct','auto',operation))
"""
    reader, writer = pty.openpty()
    with os.fdopen(reader, "rb", buffering=0) as terminal, os.fdopen(writer, "wb") as stderr:
        process = subprocess.run(
            [sys.executable, "-c", code],
            stderr=stderr,
            capture_output=False,
            stdout=subprocess.PIPE,
            check=True,
        )
        assert process.stdout == b""
        output = terminal.read(4096)
    assert b"Reading views" in output and b"2/5" in output
    assert output.count(b"\r\x1b[2K") >= 3


@pytest.mark.parametrize("method", ["sirt", "accelerated", "mlem", "spdhg", "spdhg-stream"])
@pytest.mark.parametrize("seconds", [0.01, 0.25])
def test_fixed_update_solvers_preserve_the_schedule(monkeypatch, method, seconds):
    """The single-view problem repeated exactly on archived main on both devices."""
    from itertools import count

    from tomojax.recon import MLEMConfig, SIRTConfig, SPDHGConfig

    grid = tj.Grid(6, 1, 3, 1, 1, 1)
    detector = tj.Detector(6, 3, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, [0.0])
    data = np.linspace(0.2, 1.7, 18, dtype=np.float32).reshape(1, 3, 6)
    subject = tj.Scan(data, geometry)
    options = {"iterations": 7, "backend": "jax"}
    if method.startswith("spdhg"):
        cfg = SPDHGConfig(
            **options,
            views_per_block=2,
            log_every=2,
            tv_weight=0.03,
            stream=method.endswith("-stream"),
        )
        method = "spdhg"
    elif method == "mlem":
        cfg = MLEMConfig(**options, views_per_batch=2)
    else:
        cfg = SIRTConfig(**options, views_per_batch=2, acceleration=method == "accelerated")
        method = "sirt"
    reference = tj.reconstruct(subject, method, config=cfg)
    monkeypatch.setattr("tomojax.recon._chunks.monotonic", count(step=seconds).__next__)
    events = []
    with progress.watch(events.append):
        result = tj.reconstruct(subject, method, config=cfg)
    validate_tree(events)
    updates = [event for event in events if isinstance(event, progress.Advanced)]
    assert [e.done for e in updates] == (list(range(1, 8)) if seconds == 0.25 else [1, 2, 7])
    assert len(events) == len(updates) + 2
    assert result.info["effective_iterations"] == updates[-1].done
    np.testing.assert_array_equal(result.volume, reference.volume)
    if method == "spdhg":
        np.testing.assert_array_equal(result.info["loss"], reference.info["loss"])
        for event in updates:
            expected = reference.info["loss"][event.done - 1] if event.done % 2 == 0 else None
            assert event.metrics["loss"] == expected
    else:
        # A single view through independent one-voxel rays measures the volume itself.
        np.testing.assert_allclose(result.volume[:, 0, :], data[0], rtol=0, atol=2e-7)


@pytest.mark.parametrize("failure", ["cancel", "handler"])
def test_host_slab_exit_joins_workers_and_closes_work_files(tmp_path, failure):
    import os
    from pathlib import Path
    import threading

    from tomojax.geometry import ConeBeam, ConeGeometry

    grid = tj.Grid(5, 4, 9, 1, 1.1, 0.9)
    detector = tj.Detector(8, 11, 1.6, 1.5, (0.3, -0.4))
    geometry = ConeGeometry(
        grid, detector, np.linspace(3, 363, 7, endpoint=False), ConeBeam(12, 26)
    )
    data = np.random.default_rng(56).uniform(size=(7, 11, 8)).astype(np.float32)
    out = np.memmap(tmp_path / "output", np.float32, "w+", shape=(9, 4, 5))

    def mapped_files():
        files = []
        for path in Path("/proc/self/fd").iterdir():
            try:
                name = os.readlink(path)
            except FileNotFoundError:
                continue
            if str(tmp_path) in name:
                files.append(name)
        return sorted(files)

    before = mapped_files()
    workers = set(threading.enumerate())
    failure_object = RuntimeError("out of memory")
    token = progress.Control()
    events = []

    def handler(event):
        events.append(event)
        if isinstance(event, progress.Advanced):
            assert event.done == 1
            assert any("(deleted)" in name for name in mapped_files())
            if failure == "handler":
                raise failure_object
            token.cancel()

    with (
        progress.watch(handler, control=token),
        pytest.raises(RuntimeError if failure == "handler" else progress.Cancelled) as caught,
    ):
        tj.reconstruct(
            tj.Scan(data, geometry),
            "cgls",
            out=out,
            config=CGLSConfig(backend="jax", iterations=4, tolerance=0),
        )
    if failure == "handler":
        assert caught.value is failure_object
        assert isinstance(events[-1], progress.Advanced)
    else:
        assert events[-1].status == "cancelled"
    validate_tree(events, interrupted=failure == "handler")
    assert mapped_files() == before
    assert not (set(threading.enumerate()) - workers)
    assert np.any(out != 0)


@pytest.mark.parametrize("method", ["fbp", "fdk", "fourier"])
def test_slab_advance_observes_successfully_written_output(method):
    from tomojax.geometry import ConeBeam, ConeGeometry
    from tomojax.recon import (
        FBPConfig,
        FBPHostConfig,
        FDKConfig,
        FDKHostConfig,
        FourierConfig,
        fbp_host,
        fdk_host,
        fourier,
    )

    grid = tj.Grid(7, 5, 5, 1, 1, 1)
    detector = tj.Detector(9, 5, 1, 1)
    angles = np.linspace(0, 360 if method == "fdk" else 180, 16, endpoint=False)
    if method == "fdk":
        geometry = ConeGeometry(grid, detector, angles, ConeBeam(40, 60))
        solve, config = fdk_host, FDKHostConfig(slices_per_batch=1, fdk=FDKConfig(backend="jax"))
    else:
        geometry = tj.ParallelGeometry(grid, detector, angles)
        solve, config = (
            (fbp_host, FBPHostConfig(slices_per_batch=1, fbp=FBPConfig(backend="jax")))
            if method == "fbp"
            else (fourier, FourierConfig(slices_per_batch=1, backend="numpy"))
        )
    out = np.full((5, 5, 7), np.nan, np.float32)
    data = np.random.default_rng(17).random((16, 5, 9), dtype=np.float32)
    stop = progress.Control()
    events = []

    def handler(event):
        events.append(event)
        if isinstance(event, progress.Advanced):
            assert event.done == 1
            assert np.isfinite(out[0]).all()
            stop.cancel()

    with progress.watch(handler, control=stop), pytest.raises(progress.Cancelled):
        solve(geometry, grid, detector, data, config=config, out=out)
    validate_tree(events)
    assert events[0].label == method
    assert events[-1].status == "cancelled"
    assert np.isnan(out[-1]).all()


@pytest.mark.parametrize("method", ["sirt", "accelerated", "mlem", "spdhg"])
def test_chunked_schedules_on_overlapping_rays(method, monkeypatch):
    """CPU's one-call result is exact. Multi-view GPU values varied on archived main."""
    from itertools import count

    from tomojax.recon import MLEMConfig, SIRTConfig, SPDHGConfig

    subject = scan()
    options = {"iterations": 7, "backend": "jax"}
    config = {
        "sirt": SIRTConfig(**options, subsets=3, views_per_batch=2),
        "accelerated": SIRTConfig(**options, acceleration=True, views_per_batch=2),
        "mlem": MLEMConfig(**options, subsets=3, views_per_batch=2),
        "spdhg": SPDHGConfig(**options, views_per_block=2, tv_weight=0.03, log_every=2),
    }[method]
    if method == "accelerated":
        method = "sirt"
    reference = tj.reconstruct(subject, method, config=config)
    monkeypatch.setattr("tomojax.recon._chunks.monotonic", count(step=0.25).__next__)
    events = []
    with progress.watch(events.append):
        result = tj.reconstruct(subject, method, config=config)
    validate_tree(events)
    assert [e.done for e in events if isinstance(e, progress.Advanced)] == list(range(1, 8))
    assert result.info["effective_iterations"] == reference.info["effective_iterations"] == 7
    assert result.info.get("termination") == reference.info.get("termination")
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(result.volume, reference.volume)
        if method == "spdhg":
            np.testing.assert_array_equal(result.info["loss"], reference.info["loss"])


@pytest.fixture(scope="module")
def alignment_scan():
    grid = tj.Grid(12, 10, 6, 1, 1, 1)
    detector = tj.Detector(14, 8, 1, 1)
    angles = np.linspace(0, 360, 18, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt=-25)
    truth = np.zeros((6, 10, 12), np.float32)
    truth[1:5, 2:6, 2:7] = 1
    truth[2:4, 6:8, 7:10] = 0.6
    data = np.array(tj.project(geometry, truth))
    data[8:] = np.roll(data[8:], 1, axis=2)
    data += np.random.default_rng(7).normal(0, 0.01, data.shape).astype(np.float32)
    return tj.Scan(data, geometry)


@pytest.mark.parametrize("end", ["complete", "cancel", "oom"])
def test_alignment_reports_the_completed_step_record(alignment_scan, end):
    import json

    from tomojax.alignment import AlignConfig

    events = []
    token = progress.Control()
    problem = RuntimeError("out of memory")

    def observe(event):
        json.dumps(progress.to_json(event), allow_nan=False)
        events.append(event)
        if isinstance(event, progress.Advanced):
            if end == "oom":
                raise problem
            if end == "cancel":
                token.cancel()

    cfg = AlignConfig(
        levels=(2, 1),
        coarse_iterations=2,
        iterations=2,
        image_iterations=3,
        initial_image_iterations=3,
        refine_iterations=2,
        tolerance=0,
    )
    with progress.watch(observe, control=token):
        if end != "complete":
            with pytest.raises(progress.Cancelled if end == "cancel" else RuntimeError) as caught:
                tj.align(alignment_scan, grid=alignment_scan.grid, config=cfg)
            validate_tree(events, interrupted=end == "oom")
            assert len([e for e in events if isinstance(e, progress.Advanced)]) == 1
            if end == "oom":
                assert caught.value is problem
                assert isinstance(events[-1], progress.Advanced)
            else:
                assert [e.status for e in events if isinstance(e, progress.Finished)] == [
                    "cancelled",
                    "cancelled",
                ]
            return
        result = tj.align(alignment_scan, grid=alignment_scan.grid, config=cfg)
    validate_tree(events)
    started = [e for e in events if isinstance(e, progress.Started)]
    assert [e.name for e in started] == ["align", "level", "level", "refine"]
    assert started[0].label == "translations"
    for task, level in zip(started[1:3], result.info["levels"], strict=True):
        updates = [e for e in events if isinstance(e, progress.Advanced) and e.task == task.task]
        assert updates and str(level["binning"]) in task.label
        assert [e.done for e in updates] == list(range(1, len(level["steps"]) + 1))
        assert all("undamped_proposed_pose_step_rms" in e.metrics for e in updates)
        assert [dict(e.metrics) for e in updates] == level["steps"]
    refine = [e for e in events if isinstance(e, progress.Advanced) and e.task == started[-1].task]
    assert [e.done for e in refine] == list(range(1, 1 + len(result.info["refinement"]["steps"])))


def test_align_json_distinguishes_usage_and_io_errors(tmp_path, capsys):
    import json

    from tomojax.cli.main import main

    data = str(tmp_path / "missing.nxs")
    with pytest.raises(SystemExit) as usage:
        main(["align", data, "--dry-run", "--progress", "json"])
    assert usage.value.code == 2
    assert json.loads(capsys.readouterr().out)["event"] == "error"
    status = main(["align", data, "--progress", "json", "-o", str(tmp_path / "out.nxs")])
    assert status == 1
    assert json.loads(capsys.readouterr().out)["kind"] == "input"


@pytest.mark.parametrize(
    ("tolerance", "counts", "stops"),
    [
        (
            0,
            [2, 4, 1],
            ["step_budget_reached", "step_budget_reached", "pose_steps_below_tolerance"],
        ),
        (
            10,
            [2, 2, 2],
            ["pose_steps_below_tolerance", "pose_steps_below_tolerance", "step_budget_reached"],
        ),
    ],
)
def test_alignment_keeps_image_poses_and_stops(alignment_scan, tolerance, counts, stops):
    """These budgets and stops repeated on archived main, including eight GPU runs."""
    from tomojax.alignment import AlignConfig

    cfg = AlignConfig(
        levels=(2, 1),
        coarse_iterations=2,
        iterations=4,
        image_iterations=3,
        initial_image_iterations=3,
        refine_iterations=2,
        tolerance=tolerance,
        min_iterations=2,
    )
    reference = tj.align(alignment_scan, grid=alignment_scan.grid, config=cfg)
    events = []
    with progress.watch(events.append):
        result = tj.align(alignment_scan, grid=alignment_scan.grid, config=cfg)
    validate_tree(events)
    for sample in (reference, result):
        levels = [*sample.info["levels"], sample.info["refinement"]]
        assert [len(level["steps"]) for level in levels] == counts
        assert [level["status"] for level in levels] == stops
    np.testing.assert_array_equal(result.image, reference.image)
    np.testing.assert_array_equal(result.scan.poses, reference.scan.poses)


@pytest.mark.parametrize("interrupt", [False, True])
def test_load_reports_completed_views_and_fused_corrections(tmp_path, monkeypatch, interrupt):
    import threading

    from ._helpers import write_raw_nxtomo

    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    monkeypatch.setattr("tomojax.corrections._engine._SIMPLE_FRAME_BYTES", 16)
    owner, events, token = get_ident(), [], progress.Control()
    workers = set(threading.enumerate())

    def observe(event):
        assert get_ident() == owner
        events.append(event)
        if interrupt and isinstance(event, progress.Advanced):
            token.cancel()

    with progress.watch(observe, control=token):
        if interrupt:
            with pytest.raises(progress.Cancelled):
                tj.load(path)
        else:
            result = tj.load(path)
            np.testing.assert_allclose(result.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)
    validate_tree(events)
    load, correction = [e for e in events if isinstance(e, progress.Started)]
    assert (load.name, load.total, load.unit) == ("load", 2, "views")
    assert (correction.name, correction.label, correction.total, correction.parent) == (
        "correction",
        "flat_dark, log",
        2,
        load.task,
    )
    reads = [
        (e.done, e.total)
        for e in events
        if isinstance(e, progress.Advanced) and e.task == load.task
    ]
    writes = [
        e.done for e in events if isinstance(e, progress.Advanced) and e.task == correction.task
    ]
    assert reads == ([(1, 2)] if interrupt else [(1, 2), (2, 2)])
    assert writes == ([] if interrupt else [1, 2])
    assert [e.status for e in events if isinstance(e, progress.Finished)] == [
        "cancelled" if interrupt else "done"
    ] * 2
    assert set(threading.enumerate()) <= workers


@pytest.mark.parametrize("interrupt", ["no", "cancel", "key error"])
def test_psi_reads_report_written_batches_and_release_owned_mapping(
    tmp_path, monkeypatch, interrupt
):
    from .test_psi_reader import _write_v73

    phase = np.arange(7 * 4 * 6, dtype=np.float32).reshape(7, 4, 6)
    path = tmp_path / "phase.mat"
    _write_v73(path, phase, np.arange(7), 1e-6)
    monkeypatch.setattr("tomojax.core.host_volume.host_available_memory_bytes", lambda: 1)
    from tomojax.core import host_volume

    original = host_volume.host_array
    allocated = []

    def allocate(*args, **kwargs):
        array = original(*args, **kwargs)
        allocated.append(array)
        return array

    monkeypatch.setattr(host_volume, "host_array", allocate)
    events, stop, problem = [], progress.Control(), KeyError("reader observer")

    def observe(event):
        events.append(event)
        if isinstance(event, progress.Advanced):
            assert not allocated[0].base.closed
            np.testing.assert_array_equal(allocated[0][: event.done], phase[: event.done])
            if interrupt == "key error":
                raise problem
            if interrupt == "cancel":
                stop.cancel()

    with progress.watch(observe, control=stop):
        if interrupt == "no":
            result = tj.load(path)
            np.testing.assert_array_equal(result.projections, phase)
        else:
            with pytest.raises(
                KeyError if interrupt == "key error" else progress.Cancelled
            ) as caught:
                tj.load(path)
            assert allocated[0].base.closed
            if interrupt == "key error":
                assert caught.value is problem
    validate_tree(events, interrupted=interrupt == "key error")
    assert [e.done for e in events if isinstance(e, progress.Advanced)] == (
        [3, 6, 7] if interrupt == "no" else [3]
    )
    if interrupt != "key error":
        assert events[-1].status == ("done" if interrupt == "no" else "cancelled")


def test_corrections_report_each_executed_group_on_the_caller(monkeypatch):
    from tomojax.corrections import BeamHardening, RejectViews, Stripes

    grid = tj.Grid(8, 8, 4, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(8, 4, 1, 1), np.arange(11))
    frames = tj.Frames(np.full((11, 4, 8), 500, np.uint16), geometry, white_level=1000)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 11 * 8)
    steps = (Stripes(3), RejectViews(), BeamHardening((1.0,)), BeamHardening((1.0,)))
    events, owner = [], get_ident()

    def observe(event):
        assert get_ident() == owner
        events.append(event)

    with progress.watch(observe):
        result = frames.corrected(*steps, views_per_batch=3)
    np.testing.assert_allclose(result.projections, np.log(2), rtol=1e-6)
    validate_tree(events)
    groups = [e for e in events if isinstance(e, progress.Started)]
    assert [(e.label, e.unit) for e in groups] == [
        ("flat_dark, log", "view batches"),
        ("stripes", "row batches"),
        ("reject_views", "view batches"),
        ("beam_hardening, beam_hardening", "view batches"),
    ]
    for group in groups:
        assert group.total == 4
        assert [
            e.done for e in events if isinstance(e, progress.Advanced) and e.task == group.task
        ] == [1, 2, 3, 4]


@pytest.mark.parametrize("end", ["early", "cancel", "last", "failed"])
def test_slab_task_spans_the_open_iterator(end):
    from dataclasses import replace

    from tomojax.recon import SIRTConfig

    grid = tj.Grid(6, 6, 11, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(7, 11, 1, 1), np.arange(7))
    frames = tj.Frames(np.full((7, 11, 7), 500, np.uint16), geometry, white_level=1000)
    events, token = [], progress.Control()

    def observe(event):
        events.append(event)
        if isinstance(event, progress.Advanced) and (
            end == "cancel" or (end == "last" and event.done == 2)
        ):
            token.cancel()

    target = replace(grid, vz=0.7) if end == "failed" else grid
    with progress.watch(observe, control=token):
        context = tj.reconstruct_slabs(
            frames, "sirt", grid=target, config=SIRTConfig(backend="jax", iterations=2)
        )
        assert events == []
        with context as slabs:
            assert len(events) == 1 and events[0].name == "reconstruct"
            if end == "failed":
                with pytest.raises(ValueError, match="independent axial"):
                    next(slabs)
            else:
                first, _ = next(slabs)
                assert first == slice(0, 8)
                if end == "cancel":
                    with pytest.raises(progress.Cancelled):
                        next(slabs)
                elif end == "last":
                    assert [part for part, _ in slabs] == [slice(8, 11)]
            assert not any(isinstance(e, progress.Finished) for e in events)
        validate_tree(events)
    assert [e.done for e in events if isinstance(e, progress.Advanced)] == {
        "cancel": [1],
        "early": [1],
        "last": [1, 2],
        "failed": [],
    }[end]
    assert events[-1].status == {"cancel": "cancelled", "failed": "failed"}.get(end, "done")


def test_correct_cli_emits_only_json_and_preserves_usage_errors(tmp_path, capsys):
    import json

    from tomojax.cli.main import main

    from ._helpers import write_raw_nxtomo

    source, output = tmp_path / "raw.nxs", tmp_path / "scan.nxs"
    command = ["correct", str(source), "-o", str(output), "--progress", "json"]
    assert main(command) == 1
    missing = json.loads(capsys.readouterr().out)
    assert missing["schema"] == 2 and missing["event"] == "error" and missing["kind"] == "input"
    write_raw_nxtomo(source)
    with pytest.raises(SystemExit) as usage:
        main([*command, "--binning", "0"])
    assert usage.value.code == 2 and json.loads(capsys.readouterr().out)["event"] == "error"
    assert main(command) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(r.pop("schema") == 2 for r in records)
    kinds = {
        "started": progress.Started,
        "advanced": progress.Advanced,
        "finished": progress.Finished,
        "note": progress.Note,
    }
    result = records.pop()
    assert result["event"] == "result" and result["outputs"] == [str(output)]
    events = [kinds[r.pop("event")](**r) for r in records]
    validate_tree(events)
    assert [e.name for e in events if isinstance(e, progress.Started)] == [
        "metadata",
        "load",
        "correction",
        "save",
    ]
    np.testing.assert_allclose(tj.load(output).projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)

    assert main([*command, "--force", "--views", "99"]) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    error = records.pop()
    assert error["event"] == "error" and error["kind"] == "input"
    assert "99" in error["message"]
    for record in records:
        assert record.pop("schema") == 2
    validate_tree([kinds[r.pop("event")](**r) for r in records])


def test_observer_oom_during_a_slab_read_reaches_caller_without_retry(tmp_path):
    from contextlib import closing

    from tomojax.recon import SIRTConfig, api

    from ._helpers import write_raw_nxtomo

    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    grid = tj.Grid(6, 6, 11, 1, 1, 1)
    detector = tj.Detector(7, 11, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.arange(7))
    events, reads = [], []
    error = RuntimeError("out of memory")

    def read(first, stop):
        reads.append((first, stop))
        tj.load(path)
        raise AssertionError("a failed reader cannot produce a slab")

    def observe(event):
        events.append(event)
        if isinstance(event, progress.Advanced):
            raise error

    slabs = api.reconstruct_slabs("sirt", geometry, grid, detector, read, config=SIRTConfig())
    with progress.watch(observe), closing(slabs), pytest.raises(RuntimeError) as failure:
        next(slabs)
    validate_tree(events, interrupted=True)
    assert failure.value is error and reads == [(0, 8)]
    assert isinstance(events[-1], progress.Advanced)


@pytest.mark.parametrize("at", [1, 4])
def test_fused_row_correction_counts_completed_rows_and_joins_its_workers(monkeypatch, at):
    import threading

    from tomojax.corrections import Stripes

    grid = tj.Grid(8, 8, 4, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(8, 4, 1, 1), np.arange(11))
    frames = tj.Frames(np.full((11, 4, 8), 500, np.uint16), geometry, white_level=1000)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 11 * 8)
    events, token, owner = [], progress.Control(), get_ident()
    workers = set(threading.enumerate())

    def observe(event):
        assert get_ident() == owner
        events.append(event)
        if isinstance(event, progress.Advanced) and event.done == at:
            token.cancel()

    with progress.watch(observe, control=token):
        if at == 1:
            with pytest.raises(progress.Cancelled):
                frames.corrected(Stripes(3))
        else:
            result = frames.corrected(Stripes(3))
            np.testing.assert_allclose(result.projections, np.log(2), rtol=1e-6)
    validate_tree(events)
    assert len([e for e in events if isinstance(e, progress.Started)]) == 1
    assert (events[0].label, events[0].unit, events[0].total) == (
        "flat_dark, log, stripes",
        "row batches",
        4,
    )
    assert [e.done for e in events if isinstance(e, progress.Advanced)] == list(range(1, at + 1))
    assert events[-1].status == ("cancelled" if at == 1 else "done")
    assert set(threading.enumerate()) <= workers


def test_cancelling_a_correction_drains_the_already_submitted_host_copy(monkeypatch):
    import importlib

    from tomojax.corrections import correct_frames

    engine = importlib.import_module(correct_frames.__module__)
    original = engine.first_pass
    submitted, received = [], []

    class Download:
        def __init__(self, data):
            self.data = data

        def copy_to_host_async(self):
            submitted.append(self)
            self.data.copy_to_host_async()

        def __array__(self, dtype=None, copy=None):
            received.append(self)
            return np.asarray(self.data, dtype=dtype)

    def first_pass(*args, **kwargs):
        result, *metrics = original(*args, **kwargs)
        return Download(result), *metrics

    monkeypatch.setattr(engine, "first_pass", first_pass)
    token, events = progress.Control(), []

    def observe(event):
        events.append(event)
        if isinstance(event, progress.Advanced):
            assert len(submitted) == 2 and received == submitted[:1]
            token.cancel()

    with progress.watch(observe, control=token), pytest.raises(progress.Cancelled):
        correct_frames(
            np.ones((4, 2, 3)),
            flats=None,
            darks=None,
            flat_positions=None,
            white_level=2,
            epsilon=1e-6,
            steps=(),
            views_per_batch=1,
        )
    validate_tree(events)
    assert received == submitted
    assert events[-1].status == "cancelled"


@pytest.mark.parametrize("action", ["resume", "cancel"])
def test_control_blocks_inside_task_and_discount_paused_time(monkeypatch, action):
    from threading import Event

    clock = [10.0]
    monkeypatch.setattr(progress, "monotonic", lambda: clock[0])
    control = progress.Control()
    events = []
    paused = Event()

    def observe(event):
        events.append(event)
        if isinstance(event, progress.Paused):
            paused.set()

    def run():
        with progress.watch(observe, control=control), progress.task("load", "test", 4, "views"):
            active = progress.current_task.get()
            clock[0] += 1
            active.advance(1, {})
            control.pause()
            assert control.state == "pause-requested"
            progress.check_cancelled()
            clock[0] += 1
            active.advance(2, {})

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(run)
        try:
            assert paused.wait(5)
            assert control.state == "paused" and not future.done()
            clock[0] += 100
        finally:
            getattr(control, action)()
        if action == "cancel":
            with pytest.raises(progress.Cancelled):
                future.result(timeout=5)
        else:
            future.result(timeout=5)
    assert [e.eta for e in events if isinstance(e, progress.Advanced)] == (
        [3.0, 2.0] if action == "resume" else [3.0]
    )
    assert [type(e) for e in events[2:-1]] == (
        [progress.Paused, progress.Resumed, progress.Advanced]
        if action == "resume"
        else [progress.Paused]
    )
    assert events[-1].status == ("done" if action == "resume" else "cancelled")


def test_error_kinds_use_structured_status_and_keep_parameter_path():
    from tomojax.errors import classify, within

    class Exhausted(jax.errors.JaxRuntimeError):
        @property
        def error_code(self):
            return 8

    assert classify(Exhausted("no message")).kind == "out_of_memory"
    assert (
        classify(jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: out of memory")).kind == "internal"
    )
    wrapper = ValueError("solve could not allocate")
    wrapper.__cause__ = MemoryError()
    assert classify(wrapper).kind == "out_of_memory"
    with pytest.raises(tj.InvalidParameter, match="unknown") as failure, within("config"):
        raise tj.InvalidParameter("filter", "unknown")
    assert classify(failure.value).path == "config.filter"
    assert classify(failure.value).kind == "invalid_parameter"


def test_logger_notes_between_and_inside_operations(caplog):
    import logging

    logger = logging.getLogger("tomojax.contract")
    events = []
    with progress.watch(events.append):
        with progress.task("load", "first", None, "views"):
            logger.warning("during %s", "load")
        logger.warning("between operations")
        with progress.task("save", "second", None, "files"):
            logger.warning("during save")
    validate_tree(events)
    notes = [progress.to_json(e) for e in events if isinstance(e, progress.Note)]
    assert [(n["task"], n["message"]) for n in notes] == [
        (1, "during load"),
        (None, "between operations"),
        (2, "during save"),
    ]
    assert [r.getMessage() for r in caplog.records] == [n["message"] for n in notes]


@pytest.mark.parametrize(
    ("action", "count"), [("ignore", 0), ("always", 2), ("default", 1), ("module", 1), ("once", 1)]
)
def test_warning_notes_follow_filters_and_inner_watch(action, count):
    import warnings

    outer, inner = [], []
    with warnings.catch_warnings(record=True) as ordinary, progress.watch(outer.append):
        warnings.simplefilter(action)
        with progress.watch(inner.append), progress.task("load", "test", None, "views"):
            for _ in range(2):
                progress.warn("message", stacklevel=1)
    assert [e.message for e in inner if isinstance(e, progress.Note)] == ["message"] * count
    assert len(ordinary) == count and not outer


def test_promoted_and_unrelated_warnings():
    import warnings

    events = []
    with progress.watch(events.append), warnings.catch_warnings(record=True) as ordinary:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(1) as pool:
            pool.submit(warnings.warn, "other thread").result()
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="promoted"):
            progress.warn("promoted", stacklevel=1)
    assert not events and [str(w.message) for w in ordinary] == ["other thread"]


def test_metric_inventory_and_unknown_emission():
    from pathlib import Path

    from tomojax.wire.metrics import table

    doc = Path(__file__).resolve().parents[1] / "docs/progress-metrics.md"
    assert doc.read_text() == "# Progress metrics\n\n" + table() + "\n"
    with (
        progress.watch(lambda event: None),
        progress.task("align", "test", 1, "steps"),
        pytest.raises(ValueError, match="undeclared.*typo"),
    ):
        progress.current_task.get().advance(1, {"typo": 1})


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_load_counts_completed_storage_reads(tmp_path, monkeypatch, suffix):
    original = scan()
    path = tmp_path / ("measurements" + suffix)
    tj.save(path, original)
    monkeypatch.setattr("tomojax._data._io_nxtomo._TILE_BYTES", 3 * 2 * 14 * 4)
    events = []
    with progress.watch(events.append):
        loaded = tj.load(path)
    validate_tree(events)
    (task,) = [e for e in events if isinstance(e, progress.Started)]
    assert (task.name, task.label, task.total, task.unit) == ("load", path.name, 13, "views")
    assert [e.done for e in events if isinstance(e, progress.Advanced)] == (
        [3, 6, 9, 12, 13] if suffix == ".nxs" else [13]
    )
    np.testing.assert_array_equal(loaded.projections, original.projections)


def test_metadata_and_pixel_reads_have_distinct_lifetimes(tmp_path):
    from ._helpers import write_raw_nxtomo

    path = tmp_path / "exposure.nxs"
    write_raw_nxtomo(path)
    events = []
    with progress.watch(events.append):
        frames = tj.load_frames(path)
    assert [(e.name, e.label) for e in events if isinstance(e, progress.Started)] == [
        ("metadata", path.name)
    ]
    assert not any(isinstance(e, progress.Advanced) for e in events)
    events.clear()
    with progress.watch(events.append):
        result = frames.corrected(views_per_batch=1)
    load, correction = [e for e in events if isinstance(e, progress.Started)]
    assert (load.label, load.total, correction.parent) == (path.name, 2, load.task)
    assert [e.done for e in events if isinstance(e, progress.Advanced) and e.task == load.task] == [
        1,
        2,
    ]
    np.testing.assert_allclose(result.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)


def test_public_operations_report_at_the_work(tmp_path):
    events = []
    path = tmp_path / "synthetic.nxs"
    raw = tmp_path / "phantom.raw"
    with progress.watch(events.append):
        synthetic = tj.simulate("cube", size=8, n_views=4)
        tj.save(path, synthetic)
        tj.inspect(path)
        projections = tj.project(synthetic.scan.geometry, synthetic.phantom)
        tj.backproject(synthetic.scan.geometry, projections)
        tj.export(raw, synthetic)
    validate_tree(events)
    roots = [e for e in events if isinstance(e, progress.Started) and e.parent is None]
    assert [e.name for e in roots] == [
        "simulate",
        "save",
        "inspect",
        "project",
        "backproject",
        "export",
    ]
    assert [
        e.done for e in events if isinstance(e, progress.Advanced) and e.task == roots[-1].task
    ] == list(range(1, 9))
    np.testing.assert_array_equal(np.fromfile(raw, "<f4").reshape(8, 8, 8), synthetic.phantom)
