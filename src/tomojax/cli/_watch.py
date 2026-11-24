"""``--watch-dir``: images of a running solve, rewritten at each preview checkpoint.

A run rewrites the central planes and, for an alignment, the pose table. Each
image is moved into place whole, so a viewer open on the directory never reads
a partial file.
"""

from __future__ import annotations

from pathlib import Path
from threading import Condition, Thread
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax.cli import _images

if TYPE_CHECKING:
    from types import TracebackType

IMAGES = {"z": "slice_z.png", "y": "slice_y.png", "x": "slice_x.png", "poses": "poses.png"}


class WatchDirectory:
    """The images of one run's preview checkpoints, written by one thread.

    Parameters
    ----------
    path : Path
        The directory; created with the first image.
    working : bool
        Its planes are a working image (an alignment's), not a reconstruction.
    """

    def __init__(self, path: Path, *, working: bool) -> None:
        self.path = path
        self._working = working
        self._condition = Condition()
        self._pending: tuple[progress.Preview, str] | None = None
        self._labels: dict[int, tuple[int | None, str]] = {}
        self._checkpoints = 0
        self._stopped = False
        self._error: Exception | None = None
        self._thread = Thread(target=self._write_until_stopped, name="tomojax-watch", daemon=True)

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
            except Exception as exc:  # reported when the run ends
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
        if self._error is not None and kind is None:
            raise OSError(f"writing --watch-dir images failed: {self._error}") from self._error


def watch_directory(path: str | None, *, working: bool) -> WatchDirectory | None:
    """The ``--watch-dir`` of a command, if it has one."""
    return None if path is None else WatchDirectory(Path(path), working=working)


__all__ = ["IMAGES", "WatchDirectory", "watch_directory"]
