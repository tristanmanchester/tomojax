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
        elif isinstance(event, progress.Advanced):
            assert stack[-1] == event.task
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

    token = progress.Cancel()
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
        progress.watch(handler, cancel=token),
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
        "schema": 1,
        "event": "advanced",
        "task": 2,
        "done": 1,
        "total": 4,
        "metrics": {"loss": None, "value": 3},
        "time": 0.5,
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
    token = progress.Cancel()

    def handler(event):
        events.append(event)
        if isinstance(event, progress.Advanced) and ((event.done == 4) == final):
            observed.append(len(reads))
            token.cancel()

    config = CGLSConfig(backend="jax", stream=True, views_per_batch=4, iterations=4, tolerance=0)
    with progress.watch(handler, cancel=token):
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
            assert record.pop("schema") == 1
            cls = {
                "started": progress.Started,
                "advanced": progress.Advanced,
                "finished": progress.Finished,
            }[record.pop("event")]
            events.append(cls(**record))
            if cancel and isinstance(events[-1], progress.Advanced) and events[-1].done == 1:
                process.send_signal(signal.SIGINT)
        assert process.wait(timeout=60) == (130 if cancel else 0)
    validate_tree(events)
    assert events[0].name == "reconstruct"
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
    token = progress.Cancel()
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
        progress.watch(handler, cancel=token),
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
    stop = progress.Cancel()
    events = []

    def handler(event):
        events.append(event)
        if isinstance(event, progress.Advanced):
            assert event.done == 1
            assert np.isfinite(out[0]).all()
            stop.cancel()

    with progress.watch(handler, cancel=stop), pytest.raises(progress.Cancelled):
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
