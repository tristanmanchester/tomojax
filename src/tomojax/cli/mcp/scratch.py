"""Leased worker scratch directories shared across TomoJAX servers."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

from tomojax.cli._commit import commit, helper

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def scratch(
    root: Path, *, report: dict | None = None, gui: bool = False
) -> Iterator[tuple[Path, int]]:
    """Collect abandoned workers and lease a directory to parent and child."""

    def collect(path: Path) -> None:
        links = list(path.glob(".tomojax-stage-*"))
        if links:
            abandoned = []
            for link in links:
                work = link.readlink()
                if work.name != link.name:
                    raise ValueError(f"Invalid staging reservation {link}")
                abandoned.append(str(work))
            recovered = (helper if gui else commit)([], abandoned=tuple(abandoned))
            if report is not None:
                for state, destinations in recovered.items():
                    report.setdefault(state, []).extend(destinations)
        shutil.rmtree(path)

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "registry.lock").open("ab") as registry:
        fcntl.flock(registry, fcntl.LOCK_EX)
        for old in root.glob("worker-*"):
            try:
                with (old / "lock").open("ab") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    collect(old)
            except BlockingIOError:
                continue
        path = Path(tempfile.mkdtemp(dir=root, prefix="worker-"))
        with (path / "lock").open("xb") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            fcntl.flock(registry, fcntl.LOCK_UN)
            try:
                yield path, lease.fileno()
            finally:
                fcntl.flock(registry, fcntl.LOCK_EX)
                collect(path)
