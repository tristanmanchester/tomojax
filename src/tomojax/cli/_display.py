"""The live progress display on stderr: nested tasks, and a line left for each finished one.

The watch handler only records each event under a short lock. A display thread
draws about ten times a second from a snapshot taken under that lock, and prints
queued lines (finished tasks, notes, devices) above the live rows. The thread is
the only one that calls Rich, and it never holds the state lock while drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
import sys
from threading import Event, Lock, Thread
from time import monotonic
from typing import TYPE_CHECKING

from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from tomojax import progress
from tomojax.cli import _console
from tomojax.core import format_duration

if TYPE_CHECKING:
    from types import TracebackType
    from typing import TextIO

# The metrics the display shows, by wire name; the others stay in JSON. A unit
# follows the value where it says something the name does not.
METRICS = {
    "loss": ("loss", ""),
    "normal_residual_norm": ("normal residual", ""),
    "objective": ("objective", ""),
    "pose_step_rms": ("pose step", " working px"),
}
_DEPTH_SHOWN = 4
_SUMMARY_DEPTH = 2
_SUMMARY_SECONDS = 1.0  # quicker tasks leave no line
_PERIOD = 0.1
_BAR = 20  # progress bar width in cells


def number(value: float) -> str:
    """A metric to four significant figures."""
    return f"{value:.4g}"


@dataclass
class _Task:
    label: str
    parent: int | None
    unit: str
    start: float
    total: int | None
    done: int = 0
    advanced: bool = False
    eta: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    first: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Row:
    depth: int
    label: str
    done: int
    total: int | None
    unit: str
    metrics: tuple[tuple[str, float], ...]
    elapsed: float
    eta: float | None


class Display:
    """Show a command's progress on a terminal, or only its finished tasks and notes.

    Parameters
    ----------
    token : progress.Control
        The command's cancellation token; once set, the display says it is stopping.
    live : bool
        Draw live rows (a terminal), or print finished tasks and notes only.
    """

    def __init__(self, token: progress.Control, *, live: bool) -> None:
        self._token = token
        self._live = live
        self._lock = Lock()
        self._tasks: dict[int, _Task] = {}
        self._depth: dict[int, int] = {}
        self._lines: list[Text] = []
        self._devices: tuple[str, ...] = ()
        self._clock: float | None = None  # monotonic() minus the watch's time
        self._canceling = False
        self._stop = Event()

        def draw_until_stopped() -> None:
            while not self._stop.wait(_PERIOD):
                self._draw()

        self._thread = Thread(target=draw_until_stopped, name="tomojax-display", daemon=True)
        self._rich: Live | None = None
        self._stdout: TextIO | None = None

    def __call__(self, event: progress.Event) -> None:
        """Record ``event``; drawing happens on the display thread."""
        with self._lock:
            if self._clock is None:
                self._clock = monotonic() - event.time
            if isinstance(event, progress.Started):
                parent = event.parent
                depth = 1 if parent is None else self._depth.get(parent, 0) + 1
                self._depth[event.task] = depth
                task = _Task(event.label, parent, event.unit, event.time, event.total)
                self._tasks[event.task] = task
            elif isinstance(event, progress.Advanced):
                self._advanced(event)
            elif isinstance(event, progress.Finished):
                self._finished(event)
            elif isinstance(event, progress.Placed) and event.devices != self._devices:
                self._devices = event.devices
                self._lines.append(Text(f"devices: {', '.join(event.devices)}", style="dim"))
            elif isinstance(event, progress.Note):
                self._note(event)

    def _advanced(self, event: progress.Advanced) -> None:
        task = self._tasks.get(event.task)
        if task is None:
            return
        task.advanced, task.done, task.total, task.eta = True, event.done, event.total, event.eta
        for name, value in event.metrics.items():
            if name in METRICS and isinstance(value, int | float) and math.isfinite(value):
                task.metrics[name] = float(value)
                _ = task.first.setdefault(name, float(value))

    def _finished(self, event: progress.Finished) -> None:
        task = self._tasks.pop(event.task, None)
        depth = self._depth.pop(event.task, 0)
        if task is None or depth > _SUMMARY_DEPTH:
            return
        quick = event.time - task.start < _SUMMARY_SECONDS
        if event.status == "done" and (quick or not task.advanced):
            return
        mark, style = ("✓", "green") if event.status == "done" else ("✗", "red")
        line = Text.assemble((f"{mark} ", style), (task.label, "bold"))
        count = str(task.done) if task.total is None else f"{task.done}/{task.total}"
        line.append(f"  {count}{f' {task.unit}' if task.unit else ''}")
        name = next(iter(task.first), None)
        if name is not None:
            label, unit = METRICS[name]
            first, last = number(task.first[name]), number(task.metrics[name])
            line.append(f"  {label} {first} → {last}{unit}")
        line.append(f"  {format_duration(event.time - task.start)}", style="dim")
        if event.status != "done":
            line.append(f"  ({event.status})", style="red")
        self._lines.append(line)

    def _note(self, event: progress.Note) -> None:
        if event.level == "debug" or (event.level == "info" and not self._live):
            return
        if event.level == "info":
            self._lines.append(Text(event.message, style="dim"))
        else:
            style = "yellow" if event.level == "warning" else "bold red"
            self._lines.append(Text.assemble((f"{event.level}: ", style), event.message))

    def write_line(self, text: str) -> None:
        """Queue a line the command wrote to a terminal's stdout, in order with the rest."""
        with self._lock:
            self._lines.append(Text(text))

    def _snapshot(self) -> tuple[list[Text], list[_Row]]:
        with self._lock:
            lines, self._lines = self._lines, []
            now = 0.0 if self._clock is None else monotonic() - self._clock
            rows = [
                _Row(
                    self._depth.get(key, 1),
                    task.label,
                    task.done,
                    task.total,
                    task.unit,
                    tuple((name, task.metrics[name]) for name in METRICS if name in task.metrics),
                    now - task.start,
                    task.eta,
                )
                for key, task in self._tasks.items()
            ]
            if self._token.canceled and not self._canceling:
                self._canceling = True
                lines.append(
                    Text("canceling at the next checkpoint (Ctrl-C again to stop now)", "yellow")
                )
            return lines, rows[-_DEPTH_SHOWN:]

    def _table(self, rows: list[_Row]) -> Table:
        cells = []
        for row in rows:
            count = str(row.done) if row.total is None else f"{row.done}/{row.total}"
            if row.unit and (row.done or row.total):
                count += f" {row.unit}"
            metrics = "  ".join(
                f"{METRICS[name][0]} {number(value)}{METRICS[name][1]}"
                for name, value in row.metrics
            )
            timing = format_duration(row.elapsed)
            if row.eta is not None and (row.total is None or row.done < row.total):
                timing += f" · {format_duration(row.eta)} left"
            label = f"{'  ' * (row.depth - 1)}{row.label}"
            cells.append((label, row, count if row.done or row.total else "", metrics, timing))
        # A narrow terminal shortens the metrics first: the rest keeps its width.
        table = Table.grid(padding=(0, 2))
        table.add_column(
            no_wrap=True, min_width=min(32, max((len(c[0]) for c in cells), default=0))
        )
        table.add_column(no_wrap=True, width=_BAR)
        table.add_column(no_wrap=True, min_width=max((len(c[2]) for c in cells), default=0))
        table.add_column(ratio=1)  # the one column Rich may narrow
        table.add_column(
            no_wrap=True, style="dim", min_width=max((len(c[4]) for c in cells), default=0)
        )
        for label, row, count, metrics, timing in cells:
            bar = (
                ProgressBar(total=max(row.total, 1), completed=row.done, width=_BAR)
                if row.total
                else Text("")
            )
            shown = Text(metrics, no_wrap=True, overflow="ellipsis")
            table.add_row(Text(label, overflow="ellipsis"), bar, count, shown, timing)
        return table

    def _draw(self) -> None:
        lines, rows = self._snapshot()
        if self._rich is not None:
            for line in lines:
                self._rich.console.print(line)
            self._rich.update(self._table(rows), refresh=True)
        else:
            out = _console.console()
            for line in lines:
                out.print(line)

    def __enter__(self) -> Display:
        if self._live:
            self._rich = Live(
                console=_console.console(),
                auto_refresh=False,
                transient=True,
                redirect_stdout=False,
                redirect_stderr=True,
            )
            self._rich.start()
            if sys.stdout.isatty():  # print above the rows, after the lines before it
                self._stdout = sys.stdout
                sys.stdout = _Lines(self)
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        if self._stdout is not None:
            sys.stdout.flush()
            sys.stdout = self._stdout
        self._draw()
        if self._rich is not None:
            self._rich.stop()


class _Lines(io.TextIOBase):
    """A terminal's stdout while the display runs: whole lines go to its queue."""

    def __init__(self, display: Display) -> None:
        self._display = display
        self._partial = ""

    def write(self, text: str) -> int:
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._display.write_line(line)
        return len(text)

    def flush(self) -> None:
        if self._partial:
            self._display.write_line(self._partial)
            self._partial = ""

    def isatty(self) -> bool:
        return True


__all__ = ["METRICS", "Display", "number"]
