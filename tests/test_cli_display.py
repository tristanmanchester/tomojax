"""The live progress display: what a terminal shows and what a log keeps."""

from __future__ import annotations

import os
import pty
import re
import subprocess
import sys
import threading
import time

import pytest

from tomojax import progress

pytestmark = pytest.mark.surface


@pytest.fixture(name="displays")
def display_module():
    """The display module under test (a white-box test of its internals)."""
    from tomojax.cli import _display  # check-public-imports: allow-private

    return _display


def _run_on_a_terminal(code: str) -> tuple[bytes, bytes]:
    """Run ``code`` with stderr on a pseudo-terminal and stdout on a pipe."""
    reader, writer = pty.openpty()
    with os.fdopen(reader, "rb", buffering=0) as terminal, os.fdopen(writer, "wb") as stderr:
        process = subprocess.run(
            [sys.executable, "-c", code],
            stderr=stderr,
            stdout=subprocess.PIPE,
            check=True,
            env={**os.environ, "COLUMNS": "120", "JAX_PLATFORMS": "cpu"},
        )
        chunks = []
        while True:
            try:
                chunk = terminal.read(65536)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunk) < 65536:
                break
    return process.stdout, b"".join(chunks)


def test_a_terminal_shows_rows_and_keeps_a_line_per_finished_task():
    code = """import time
from tomojax import progress
import tomojax.cli._display as display
from tomojax.cli._progress import run_command
display._SUMMARY_SECONDS = 0
def operation():
    with progress.task('solve', 'cgls', 4, 'iterations'):
        for done, value in enumerate((9.0, None, 4.0, 2.0), start=1):
            progress.current_task.get().advance(done, {'normal_residual_norm': value})
            time.sleep(0.15)
    print('report line that belongs on stdout')
raise SystemExit(run_command('reconstruct', 'auto', operation))
"""
    stdout, terminal = _run_on_a_terminal(code)
    # Stdout is a pipe here: its data stays on stdout, whole.
    assert stdout == b"report line that belongs on stdout\n"
    shown = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", terminal.decode())
    assert "cgls" in shown and "/4 iterations" in shown and "left" in shown
    assert "normal residual 9 → 2" in shown
    assert "✓ cgls  4/4 iterations" in shown


def _events(display, *events):
    for event in events:
        display(event)


def _drawn(capsys, display):
    display._draw()
    return capsys.readouterr().err


def test_finished_tasks_at_the_top_two_levels_leave_a_line(capsys, monkeypatch, displays):
    monkeypatch.setattr(displays, "_SUMMARY_SECONDS", 1.0)
    display = displays.Display(progress.Control(), live=False)
    _events(
        display,
        progress.Started(1, None, "align", "translations", None, "fit", 0.0),
        progress.Started(2, 1, "level", "binning 2", 12, "steps", 0.0),
        progress.Started(3, 2, "step", "step 1", None, "", 0.0),
        progress.Advanced(2, 1, 12, {"objective": 4.38, "pose_step_rms": 0.5}, 1.0),
        progress.Finished(3, "done", 1.5),
        progress.Advanced(2, 2, 12, {"objective": float("nan")}, 2.0),
        progress.Advanced(2, 3, 12, {"objective": 0.0833}, 3.0),
        progress.Finished(2, "done", 3.0),
        progress.Started(4, 1, "level", "binning 1", 8, "steps", 3.0),
        progress.Advanced(4, 1, 8, {}, 3.2),
        progress.Finished(4, "done", 3.5),  # under a second: no line
        progress.Finished(1, "cancelled", 3.6),
    )
    lines = _drawn(capsys, display).splitlines()
    assert lines == [
        "✓ binning 2  3/12 steps  objective 4.38 → 0.0833  3.0s",
        "✗ translations  0 fit  3.6s  (cancelled)",
    ]


def test_notes_appear_once_and_info_only_with_live_rows(capsys, displays):
    quiet = displays.Display(progress.Control(), live=False)
    _events(
        quiet,
        progress.Note(None, "info", "an info line", 0.0),
        progress.Note(None, "warning", "a warning", 0.0),
        progress.Note(None, "debug", "a debug line", 0.0),
        progress.Placed(1, ("gpu:0",), 0.0),
        progress.Placed(1, ("gpu:0",), 0.1),
    )
    assert _drawn(capsys, quiet).splitlines() == ["warning: a warning", "devices: gpu:0"]
    assert _drawn(capsys, quiet) == ""


def test_cancelling_is_announced_once(capsys, displays):
    token = progress.Control()
    display = displays.Display(token, live=False)
    token.cancel()
    assert "cancelling at the next checkpoint" in _drawn(capsys, display)
    assert _drawn(capsys, display) == ""


def test_the_handler_never_waits_for_drawing(monkeypatch, displays):
    release = threading.Event()
    drawing = threading.Event()

    class Blocked:
        def print(self, *_args, **_kwargs):
            drawing.set()
            release.wait(5)

    monkeypatch.setattr(displays._console, "console", lambda **_kwargs: Blocked())
    display = displays.Display(progress.Control(), live=False)
    with display:
        display(progress.Note(None, "warning", "start drawing", 0.0))
        assert drawing.wait(2)  # the display thread is stuck in a slow print
        start = time.perf_counter()
        for done in range(1000):
            display(progress.Note(None, "warning", f"note {done}", 0.0))
        assert time.perf_counter() - start < 0.5
        release.set()


def test_bar_off_a_terminal_prints_finished_tasks_and_warnings_only(capsys, monkeypatch, displays):
    from tomojax.cli._progress import run_command  # check-public-imports: allow-private

    monkeypatch.setattr(displays, "_SUMMARY_SECONDS", 0.0)

    def operation():
        with progress.task("solve", "sirt", 2, "iterations"):
            progress.current_task.get().advance(1, {})
            progress.note("info", "not shown off a terminal")
            progress.note("warning", "shown")
            progress.current_task.get().advance(2, {})

    assert run_command("reconstruct", "bar", operation) == 0
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "warning: shown"
    assert lines[1].startswith("✓ sirt  2/2 iterations")
    assert len(lines) == 2
