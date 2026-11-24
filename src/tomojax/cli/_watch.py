"""``--watch-dir``: images of a running solve, rewritten at each preview checkpoint.

A run writes only a fixed set of names: the central planes, the pose table of
an alignment, a status file and the temporary names they pass through. A
directory is used only when it is empty, or when it holds nothing but those
names and a status file; anything else is refused, ``--force`` or not, before
anything is deleted. A kernel-held lock keeps a second run out while one is live
and needs no recovery when a run dies.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import stat
from threading import Condition, Thread
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax import progress
from tomojax.cli import _images

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

STATUS = "tomojax-watch.json"
LOCK = ".tomojax-watch.lock"
IMAGES = {"z": "slice_z.png", "y": "slice_y.png", "x": "slice_x.png", "poses": "poses.png"}
OWNED = frozenset(
    {
        STATUS,
        LOCK,
        f".{STATUS}.tmp",
        *IMAGES.values(),
        *(f".{name}.tmp" for name in IMAGES.values()),
    }
)


class WatchDirectory:
    """The images of one run's preview checkpoints, written by one thread.

    Parameters
    ----------
    path : Path
        The directory; created if missing.
    command : str
        The command, for the status file.
    working : bool
        Its planes are a working image (an alignment's), not a reconstruction.
    """

    def __init__(self, path: Path, command: str, *, working: bool) -> None:
        self.path = path
        self._command = command
        self._working = working
        self._lock_file: int | None = None
        self._condition = Condition()
        self._pending: tuple[progress.Preview, str] | None = None
        self._labels: dict[int, tuple[int | None, str]] = {}
        self._checkpoints = 0
        self._written = 0
        self._last: str | None = None
        self._stopped = False
        self._error: BaseException | None = None
        self._started = datetime.now(UTC).isoformat(timespec="seconds")
        self._thread = Thread(target=self._write_until_stopped, name="tomojax-watch", daemon=True)

    def admit(self, *, force: bool, related: Iterable[str | Path | None]) -> None:
        """Lock the directory, check it holds only this command's files, and say so in it.

        Raises
        ------
        FileExistsError
            Another run holds the directory, or it holds earlier images and ``force``
            is not set.
        ValueError
            It holds anything else, or contains an input or output of the command.
        """
        directory = self.path.resolve()
        for item in related:
            if item is not None and Path(item).resolve().is_relative_to(directory):
                raise ValueError(f"--watch-dir {self.path} contains the command's file {item}")
        directory.mkdir(parents=True, exist_ok=True)
        handle = os.open(directory / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(handle)
            raise FileExistsError(f"another run is writing to {self.path}") from exc
        self._lock_file = handle
        try:
            self._check(directory, force=force)
        except BaseException:
            self._unlock()
            raise
        self._status("running")

    def _check(self, directory: Path, *, force: bool) -> None:
        entries = {entry.name: entry for entry in os.scandir(directory)}
        foreign = sorted(name for name in entries if name not in OWNED)
        if foreign:
            raise ValueError(
                f"--watch-dir {self.path} holds other files ({', '.join(foreign[:3])}); "
                "give an empty or new directory"
            )
        for name, entry in entries.items():
            if not stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
                raise ValueError(f"--watch-dir {self.path}: {name} is not a regular file")
        images = [name for name in entries if name.endswith(".png")]
        if images and STATUS not in entries:
            raise ValueError(f"--watch-dir {self.path} holds images no TomoJAX run recorded")
        earlier = [name for name in entries if name != LOCK]
        if earlier and not force:
            raise FileExistsError(f"--watch-dir {self.path} holds an earlier run's images")
        # The status stays until the new one replaces it, so a crash at any point
        # leaves a directory the next run accepts.
        for name in earlier:
            if name != STATUS:
                (directory / name).unlink()

    def _unlock(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            os.close(self._lock_file)
            self._lock_file = None

    def observe(self, event: progress.Event) -> None:
        """Track task labels and keep the newest preview; never blocks on writing."""
        if isinstance(event, progress.Started):
            self._labels[event.task] = (event.parent, event.label)
        elif isinstance(event, progress.Preview):
            path, task = [], event.task
            while task is not None and task in self._labels:
                parent, label = self._labels[task]
                path.append(label)
                task = parent
            with self._condition:
                self._checkpoints += 1
                label = f"{' > '.join(reversed(path))} · checkpoint {self._checkpoints}"
                self._pending = (event, label)
                self._condition.notify()

    def _write_until_stopped(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopped:
                    _ = self._condition.wait()
                if self._pending is None:
                    return
                (event, label), self._pending = self._pending, None
            try:
                self._write(event, label)
            except BaseException as exc:  # reported when the run ends
                self._error = exc
                return

    def _write(self, event: progress.Preview, label: str) -> None:
        elapsed = f"{event.time:.1f} s"
        for key, sample in event.arrays.items():
            name = IMAGES.get(key)
            if name is None:
                continue
            if key == "poses":
                columns = [column.name for column in sample.columns]
                units = [column.unit for column in sample.columns]
                title = f"{label} · {elapsed} · per-view pose change in this fit"
                image = _images.pose_plot(np.asarray(sample.data), columns, units, title)
            else:
                data = _images.sample_image(sample)
                kind = "working image" if self._working else "volume"
                text = f"{key} plane · {label} · {elapsed} · {kind}"
                image = _images.labelled(_images.to_uint8(data, _images.limits(data)), text)
            _ = _images.save_png(image, self.path / name)
        self._written += 1
        self._last = label
        self._status("running")

    def _status(self, status: str) -> None:
        record: dict[str, Any] = {
            "command": self._command,
            "started": self._started,
            "status": status,
            "checkpoints_written": self._written,
            "last_checkpoint": self._last,
        }
        target = self.path / STATUS
        temporary = self.path / f".{STATUS}.tmp"
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(target)

    def __enter__(self) -> WatchDirectory:
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify()
        self._thread.join()
        if self._lock_file is None:  # never admitted: the command stopped before solving
            return
        try:
            if kind is None and self._error is None:
                self._status("done" if self._written else "done, no preview")
            else:
                cancelled = isinstance(error, progress.Cancelled | KeyboardInterrupt)
                self._status("cancelled" if cancelled else "failed")
        finally:
            self._unlock()
        if self._error is not None and kind is None:
            raise OSError(f"writing --watch-dir images failed: {self._error}") from self._error


def watch_directory(path: str | None, command: str, *, working: bool) -> WatchDirectory | None:
    """The ``--watch-dir`` of a command, if it has one."""
    return None if path is None else WatchDirectory(Path(path), command, working=working)


def admit_or_exit(
    parser: argparse.ArgumentParser,
    watch: WatchDirectory | None,
    *,
    force: bool,
    related: Iterable[str | Path | None],
) -> None:
    """Admit ``watch``'s directory, or end the command with a usage error and a hint."""
    from tomojax.cli._progress import ProgressParser

    if watch is None:
        return
    try:
        watch.admit(force=force, related=related)
    except FileExistsError as exc:
        hint = None if "another run" in str(exc) else "pass --force to replace its images"
        if isinstance(parser, ProgressParser):
            parser.fail(str(exc), hint=hint)
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))


__all__ = [
    "IMAGES",
    "LOCK",
    "OWNED",
    "STATUS",
    "WatchDirectory",
    "admit_or_exit",
    "watch_directory",
]
