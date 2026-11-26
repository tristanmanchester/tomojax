"""Checkpointed runs stop on SIGTERM or a kill and resume to the uninterrupted result."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import tomojax as tj
from tomojax import progress
from tomojax.core import checkpoint
from tomojax.errors import Canceled, InvalidParameter
from tomojax.recon import FBPConfig, FourierConfig, MLEMConfig, SIRTConfig

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def every_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save at every loop boundary, so a stop at any of them has something to resume."""
    monkeypatch.setattr(checkpoint, "INTERVAL", 0.0)


def _terminated_after(advances: int) -> Callable[[progress.Event], None]:
    """A progress handler that sends this process SIGTERM after ``advances`` reports."""
    seen = [0]

    def handler(event: progress.Event) -> None:
        if isinstance(event, progress.Advanced):
            seen[0] += 1
            if seen[0] == advances:
                os.kill(os.getpid(), signal.SIGTERM)

    return handler


def _first_report(run: Callable[[], Any]) -> tuple[Any, int | None]:
    """``run()`` and the completed count of its first progress report."""
    done: list[int] = []

    def handler(event: progress.Event) -> None:
        if isinstance(event, progress.Advanced):
            done.append(event.done)

    with progress.watch(handler):
        result = run()
    return result, done[0] if done else None


def _scan(geometry: str = "parallel", views: int = 24) -> tj.Scan:
    return tj.simulate("shepp", size=16, n_views=views, geometry=geometry).scan


SOLVES = [
    pytest.param("cgls", {"iterations": 12}, id="cgls"),
    pytest.param("fista", {"iterations": 12, "tv_weight": 1e-3}, id="fista"),
    pytest.param("sirt", {"iterations": 12}, id="sirt"),
    pytest.param(
        "sirt", {"config": SIRTConfig(iterations=12, acceleration=True)}, id="sirt-accelerated"
    ),
    pytest.param("mlem", {"iterations": 12}, id="mlem"),
    pytest.param("spdhg", {"iterations": 120, "tv_weight": 1e-3}, id="spdhg"),
]


@pytest.mark.usefixtures("every_boundary")
@pytest.mark.parametrize(("method", "options"), SOLVES)
def test_a_stopped_solve_resumes_to_the_uninterrupted_volume(
    tmp_path: Path, method: str, options: dict[str, Any]
) -> None:
    scan = _scan()
    path = tmp_path / "run"
    plain = np.asarray(tj.reconstruct(scan, method, **options).volume)
    whole = np.asarray(tj.reconstruct(scan, method, checkpoint=path, **options).volume)
    assert not path.exists()  # removed once the run returns

    with pytest.raises(Canceled, match="SIGTERM"), progress.watch(_terminated_after(3)):
        tj.reconstruct(scan, method, checkpoint=path, **options)
    assert (path / "checkpoint.json").is_file()
    recon, first = _first_report(lambda: tj.reconstruct(scan, method, checkpoint=path, **options))

    assert first is not None and first >= 3  # it continued; a fresh start reports 1 first
    np.testing.assert_array_equal(whole, plain)
    np.testing.assert_array_equal(np.asarray(recon.volume), whole)
    assert not path.exists()


HOST_SOLVES = [
    pytest.param("cgls", "parallel", {}, id="cuda-cgls"),
    pytest.param("cgls", "cone", {"out": True}, id="host-cone-cgls"),
    pytest.param("cgls", "parallel", {"out": True}, id="host-parallel-cgls"),
    pytest.param("fista", "cone", {"out": True, "tv_weight": 1e-3}, id="host-cone-fista"),
    pytest.param("fista", "parallel", {"out": True, "tv_weight": 1e-3}, id="host-fista"),
    pytest.param("fista", "parallel", {"tv_weight": 0.0, "nonnegative": True}, id="owned-fista"),
    pytest.param("sirt", "cone", {"out": True}, id="host-cone-sirt"),
    pytest.param("mlem", "cone", {"out": True}, id="host-cone-mlem"),
]


@pytest.mark.gpu
@pytest.mark.usefixtures("every_boundary")
@pytest.mark.parametrize(("method", "geometry", "options"), HOST_SOLVES)
def test_the_cuda_and_host_solvers_resume_to_the_uninterrupted_volume(
    tmp_path: Path, method: str, geometry: str, options: dict[str, Any]
) -> None:
    # Eight views make owned FISTA's residuals smaller than its volume.
    scan = _scan(geometry, views=8 if "nonnegative" in options else 24)
    grid = scan.grid
    hosted = options.pop("out", False)
    path = tmp_path / "run"

    def run(**more: Any) -> np.ndarray:
        out = np.zeros(grid.shape, np.float32) if hosted else None
        volume = tj.reconstruct(scan, method, iterations=8, out=out, **options, **more).volume
        return np.asarray(volume)

    plain, whole = run(), run(checkpoint=path)
    with pytest.raises(Canceled), progress.watch(_terminated_after(3)):
        run(checkpoint=path)
    resumed, first = _first_report(lambda: run(checkpoint=path))

    assert first is not None and first >= 3
    np.testing.assert_array_equal(whole, plain)
    np.testing.assert_array_equal(resumed, whole)


AUTOMATIC_FISTA = [
    pytest.param("parallel", False, id="fista"),
    pytest.param("parallel", True, marks=pytest.mark.gpu, id="host-fista"),
    pytest.param("cone", True, marks=pytest.mark.gpu, id="host-cone-fista"),
]


@pytest.mark.usefixtures("every_boundary")
@pytest.mark.parametrize(("geometry", "hosted"), AUTOMATIC_FISTA)
def test_an_automatic_fista_resumes_to_the_same_weight_and_stopping_update(
    tmp_path: Path, geometry: str, hosted: bool
) -> None:
    scan = _scan(geometry)
    path = tmp_path / "run"

    def run(**more: Any) -> tj.Reconstruction:
        out = np.zeros(scan.grid.shape, np.float32) if hosted else None
        return tj.reconstruct(scan, "fista", out=out, **more)

    plain = run()
    with pytest.raises(Canceled), progress.watch(_terminated_after(3)):
        run(checkpoint=path)
    resumed, first = _first_report(lambda: run(checkpoint=path))

    stop = plain.info["effective_iterations"]
    assert plain.info["early_stop"] and isinstance(stop, int)
    assert first is not None and 3 <= first < stop  # stopped and resumed before the noise stop
    stopping = (
        "effective_iterations",
        "early_stop",
        "noise_score_rms",
        "resolved_tv_weight",
        "effective_tv_weight",
        "loss",
    )
    assert {k: resumed.info[k] for k in stopping} == {k: plain.info[k] for k in stopping}
    np.testing.assert_array_equal(np.asarray(resumed.volume), np.asarray(plain.volume))


@pytest.mark.usefixtures("every_boundary")
@pytest.mark.parametrize("method", ["sirt", "mlem"])
def test_independent_plane_slabs_resume_from_the_first_unfinished_slab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    from tomojax.recon import _axial_slabs  # check-public-imports: allow-private

    monkeypatch.setattr(
        _axial_slabs, "axial_slab_depth", lambda _g, grid, *_: 4 if grid.nz > 4 else None
    )
    scan = _scan()
    path = tmp_path / "run"
    whole = tj.reconstruct(scan, method, iterations=6, checkpoint=path)
    assert whole.info["host_slabs"]
    with monkeypatch.context() as patched:
        _terminate_after(patched, 2)
        with pytest.raises(Canceled):
            tj.reconstruct(scan, method, iterations=6, checkpoint=path)
    saved = _terminate_after(monkeypatch, -1)
    resumed = tj.reconstruct(scan, method, iterations=6, checkpoint=path)

    assert [key for key in saved if "/z" in key] == ["slabs-0/z8-12", "slabs-0/z12-16"]
    np.testing.assert_array_equal(np.asarray(resumed.volume), np.asarray(whole.volume))


def test_a_checkpoint_of_another_run_is_refused(tmp_path: Path) -> None:
    scan = _scan()
    path = tmp_path / "run"
    with pytest.raises(Canceled), progress.watch(_terminated_after(1)):
        tj.reconstruct(scan, "cgls", iterations=20, checkpoint=path)
    with pytest.raises(InvalidParameter, match="another run.*settings") as caught:
        tj.reconstruct(scan, "cgls", iterations=21, checkpoint=path)
    assert caught.value.path == "checkpoint"
    other = tj.Scan(np.asarray(scan.projections) * 2, scan.geometry)
    with pytest.raises(InvalidParameter, match="another run.*input"):
        tj.reconstruct(other, "cgls", iterations=20, checkpoint=path)


def test_a_fourier_checkpoint_is_refused_with_another_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = _scan()
    path = tmp_path / "run"
    out = np.zeros(scan.grid.shape, np.float32)
    native = FourierConfig(backend="numpy", slices_per_batch=4)
    with monkeypatch.context() as patched:
        _terminate_after(patched, 1)
        with pytest.raises(Canceled):
            tj.reconstruct(scan, "fourier", config=native, out=out, checkpoint=path)
    with pytest.raises(InvalidParameter, match="another run.*settings"):
        tj.reconstruct(scan, "fourier", config=native, filter="ramp", out=out, checkpoint=path)


def test_a_directory_of_other_files_is_refused(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(InvalidParameter, match="no TomoJAX checkpoint"):
        tj.reconstruct(_scan(), "cgls", iterations=2, checkpoint=tmp_path)
    assert (tmp_path / "notes.txt").read_text() == "mine"


def test_one_process_at_a_time_holds_a_checkpoint(tmp_path: Path) -> None:
    with (
        checkpoint.opened(tmp_path / "run"),
        pytest.raises(InvalidParameter, match="in use by another process"),
    ):
        checkpoint.Session(tmp_path / "run")


def test_a_save_interrupted_part_way_leaves_the_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = checkpoint.Session(tmp_path)
    session.bind({"run": "a"})
    session.save("solve-0", (np.arange(4.0), np.ones(3)), {"count": 1})
    writes = [0]
    write = np.lib.format.write_array

    def failing(*args: Any, **kwargs: Any) -> None:
        writes[0] += 1
        if writes[0] == 2:
            raise KeyboardInterrupt  # as a kill would, between the two arrays
        write(*args, **kwargs)

    monkeypatch.setattr(np.lib.format, "write_array", failing)
    with pytest.raises(KeyboardInterrupt):
        session.save("solve-0", (np.arange(4.0) + 1, np.zeros(3)), {"count": 2})
    session.close()
    (tmp_path / "state-000099").mkdir()  # as a kill during a save leaves it

    reopened = checkpoint.Session(tmp_path)
    saved = reopened.load("solve-0")
    assert saved is not None
    assert saved.record == {"count": 1}
    np.testing.assert_array_equal(saved.arrays[0], np.arange(4.0))
    assert len(list(tmp_path.glob("state-*"))) == 1


def test_saves_wait_the_interval_and_twenty_save_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(checkpoint, "monotonic", lambda: clock[0])
    session = checkpoint.Session(tmp_path)
    assert not session.due()
    clock[0] += checkpoint.INTERVAL
    assert session.due()
    session.saved_at, session.save_seconds = clock[0], checkpoint.INTERVAL / 10
    clock[0] += checkpoint.INTERVAL
    assert not session.due()  # twenty saves of a tenth of the interval: twice the interval
    clock[0] += checkpoint.INTERVAL
    assert session.due()
    session.close()


_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    import tomojax as tj
    from tomojax.core import checkpoint

    checkpoint.INTERVAL = 0.0
    scan = tj.simulate("shepp", size=24, n_views=32, geometry="parallel").scan
    volume = tj.reconstruct(scan, "sirt", iterations=600, checkpoint=sys.argv[1]).volume
    np.save(sys.argv[2], np.asarray(volume))
    """
)


def _child(*args: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    return subprocess.Popen([sys.executable, "-c", _CHILD, *args], env=env)


@pytest.mark.slow
def test_kills_at_any_moment_resume_to_the_uninterrupted_volume(tmp_path: Path) -> None:
    whole, part = tmp_path / "whole.npy", tmp_path / "resumed.npy"
    start = time.monotonic()
    assert _child(str(tmp_path / "once"), str(whole)).wait() == 0
    seconds = time.monotonic() - start
    rng = np.random.default_rng(7)
    kills = 0
    while not part.exists() and kills < 30:
        child = _child(str(tmp_path / "run"), str(part))
        try:
            child.wait(timeout=float(rng.uniform(0.3, 0.9) * seconds))
        except subprocess.TimeoutExpired:
            child.kill()  # SIGKILL: no handler, mid-save as often as not
            child.wait()
            kills += 1
    assert 0 < kills < 30
    np.testing.assert_array_equal(np.load(part), np.load(whole))


def test_sigterm_outside_a_checkpointed_loop_ends_the_process(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import os, signal, sys
        from tomojax.core import checkpoint

        with checkpoint.opened(sys.argv[1]):
            os.kill(os.getpid(), signal.SIGTERM)
            print("survived")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "run")], capture_output=True, check=False
    )
    assert done.returncode == -signal.SIGTERM
    assert b"survived" not in done.stdout


def _terminate_after(monkeypatch: pytest.MonkeyPatch, count: int, of: str = "/z") -> list[str]:
    """Send SIGTERM once ``count`` saved keys contain ``of``; return every saved key."""
    save, keys = checkpoint.Session.save, []

    def terminating(session: checkpoint.Session, key: str, *args: Any) -> None:
        save(session, key, *args)
        keys.append(key)
        if sum(of in k for k in keys) == count:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(checkpoint.Session, "save", terminating)
    return keys


DIRECT = [
    pytest.param("parallel", "fbp", None, id="axial-fbp"),
    pytest.param("lamino", "fbp", None, id="laminography-fbp"),
    pytest.param("parallel", "fourier", FourierConfig(backend="numpy", slices_per_batch=4)),
    pytest.param(
        "parallel",
        "fourier",
        FourierConfig(backend="numpy", slices_per_batch=4, filter="ramp"),
        id="fourier-ramp",
    ),
    pytest.param("cone", "fbp", None, id="fdk"),
]


@pytest.mark.parametrize("storage", ["memory", "file"])
@pytest.mark.parametrize(("geometry", "method", "config"), DIRECT)
def test_slab_reconstructions_into_out_resume_from_the_first_unfinished_slab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    geometry: str,
    method: str,
    config: Any,
    storage: str,
) -> None:
    fdk, fbp = map(importlib.import_module, ("tomojax.recon.fdk", "tomojax.recon.fbp_host"))
    ranges = fdk.host_slab_ranges
    monkeypatch.setattr(fdk, "host_slab_ranges", lambda grid, devices, _: ranges(grid, devices, 4))
    monkeypatch.setattr(fbp, "_output_memory", lambda *_: 1)  # one-plane laminography slabs
    # Axial FBP slabs are at most 16 planes deep: 64 planes make four.
    size = 24 if geometry == "lamino" else 64
    scan = tj.simulate("shepp", size=size, n_views=32, geometry=geometry).scan
    path = tmp_path / "run"

    def run(**options: Any) -> np.ndarray:
        if storage == "file":
            name = tmp_path / "volume.npy"
            mode = "r+" if name.exists() else "w+"
            out = np.lib.format.open_memmap(name, mode, np.float32, scan.grid.shape)
        else:
            out = np.zeros(scan.grid.shape, np.float32)
        return np.array(tj.reconstruct(scan, method, config=config, out=out, **options).volume)

    with monkeypatch.context() as patched:
        every = _terminate_after(patched, -1)
        whole = run(checkpoint=tmp_path / "whole")
        slabs = sum("/z" in key for key in every)
        _terminate_after(patched, 2)
        with pytest.raises(Canceled):
            run(checkpoint=path)
    saved = _terminate_after(monkeypatch, -1)
    resumed = run(checkpoint=path)

    assert sum("/z" in key for key in saved) == slabs - 2  # only the slabs left over
    np.testing.assert_array_equal(resumed, whole)
    np.testing.assert_array_equal(whole, run())
    assert not path.exists()


SLABS = [
    pytest.param("fbp", FBPConfig(backend="jax"), id="fbp"),
    pytest.param("fourier", FourierConfig(backend="numpy", slices_per_batch=2), id="fourier"),
    pytest.param(
        "fourier",
        FourierConfig(backend="numpy", slices_per_batch=2, filter="hann"),
        id="fourier-hann",
    ),
    pytest.param("sirt", SIRTConfig(iterations=4), id="sirt"),
    pytest.param("mlem", MLEMConfig(iterations=4), id="mlem"),
]


@pytest.mark.parametrize(("method", "config"), SLABS)
def test_slabs_from_frames_continue_after_the_last_finished_slab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, config: Any
) -> None:
    from tomojax.recon import _iterative_slabs  # check-public-imports: allow-private

    monkeypatch.setattr(_iterative_slabs, "device_free_memory_bytes", lambda: 1)  # 1-plane slabs
    scan = tj.simulate("shepp", size=32, n_views=24, geometry="parallel").scan  # 4 FBP slabs
    counts = 1000 * np.exp(-np.asarray(scan.projections))
    frames = tj.Frames(counts.astype(np.float32), scan.geometry, white_level=1000)
    path = tmp_path / "run"
    with tj.reconstruct_slabs(frames, method, config=config) as slabs:
        whole = np.concatenate([recon.volume for _, recon in slabs])
    written = np.full(whole.shape, np.nan, np.float32)

    def run() -> list[int]:
        starts = []
        with tj.reconstruct_slabs(frames, method, config=config, checkpoint=path) as slabs:
            for z, recon in slabs:
                written[z] = recon.volume
                starts.append(z.start)
        return starts

    with monkeypatch.context() as patched:
        _terminate_after(patched, 2)
        with pytest.raises(Canceled):
            run()
    finished = int(np.isfinite(written[:, 0, 0]).sum())
    starts = run()

    assert starts[0] == finished > 0  # the first slab after the finished ones
    np.testing.assert_array_equal(written, whole)
    assert not path.exists()


def test_slabs_from_frames_left_early_keep_their_checkpoint(tmp_path: Path) -> None:
    scan = tj.simulate("shepp", size=16, n_views=24, geometry="parallel").scan
    counts = 1000 * np.exp(-np.asarray(scan.projections))
    frames = tj.Frames(counts.astype(np.float32), scan.geometry, white_level=1000)
    config = FourierConfig(backend="numpy", slices_per_batch=4)
    with tj.reconstruct_slabs(frames, "fourier", config=config, checkpoint=tmp_path / "run") as s:
        next(s)
        next(s)
    with tj.reconstruct_slabs(frames, "fourier", config=config, checkpoint=tmp_path / "run") as s:
        assert next(s)[0].start == 4  # the first slab was finished; the second was not
