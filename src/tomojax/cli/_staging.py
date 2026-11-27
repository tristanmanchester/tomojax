"""Private working paths and the session's durable commit point."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from tomojax.cli.execution import ArrayStage
from tomojax.core.input_files import check_output, publishing
from tomojax.io.api import ExportPlan, install_export, publishing as lock_directories
from tomojax.wire.arrays import Npy


@dataclass
class Destination:
    path: Path
    work: Path
    force: bool
    keep_backup: bool = False

    @property
    def stage(self) -> Path:
        return self.work / "new" / self.path.name

    def restore(self, backup: Path, destination: Path) -> None:
        try:
            backup.replace(destination)
        except OSError as err:
            self.keep_backup = True
            raise OSError(
                f"failed to restore {destination}; recover the original at {backup}"
            ) from err

    def clean(self) -> None:
        if not self.keep_backup and self.work.exists():
            shutil.rmtree(self.work)

    def install(self, undo: ExitStack) -> None:
        if not self.stage.exists():
            raise FileNotFoundError(f"missing staged output for {self.path}")
        for parent in reversed(self.path.parents):
            if not parent.exists():
                parent.mkdir()
                undo.callback(parent.rmdir)
        if self.force and os.path.lexists(self.path):
            if not self.path.is_symlink() and self.path.is_dir() != self.stage.is_dir():
                raise ValueError(f"incompatible destination type: {self.path}")
            backup = self.work / "original"
            self.path.replace(backup)
            undo.callback(self.restore, backup, self.path)
        if self.force:
            self.stage.replace(self.path)
        elif self.stage.is_dir():
            # Reserve the name exclusively before replacing the empty directory.
            self.path.mkdir()
            with ExitStack() as reservation:
                reservation.callback(self.path.rmdir)
                self.stage.replace(self.path)
                reservation.pop_all()
        else:
            os.link(self.stage, self.path)
            undo.callback(self.path.unlink)
            return
        undo.callback(self.path.replace, self.stage)


class Stages:
    """One admission owns all its reservations until commit or disposal."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.arrays: dict[str, ArrayStage] = {}
        self.outputs: dict[Path, Destination] = {}
        self.temporary = ExitStack()
        self.export_group: tuple[ExportPlan, dict[Path, Path], bool] | None = None

    def array(self) -> ArrayStage:
        token = uuid4().hex
        private = self.root / (token + ".stage.npy")
        stage = ArrayStage(private, Npy(str(self.root / (token + ".npy")), token))
        self.arrays[token] = stage
        self.temporary.callback(private.unlink, missing_ok=True)
        return stage

    def output(self, destination: Path, *, force: bool) -> Path:
        # Canonicalize parents, but replace a destination symlink itself.
        if not destination.name or destination.name == "..":
            raise ValueError("expected an output file or directory name")
        dest = destination.parent.resolve() / destination.name
        check_output(dest)
        if dest.is_relative_to(self.root) or self.root.is_relative_to(dest):
            raise ValueError("durable destination overlaps session storage")
        for prior in self.outputs:
            if dest.is_relative_to(prior) or prior.is_relative_to(dest):
                raise ValueError("overlapping durable destinations")
        parent = next(parent for parent in dest.parents if parent.exists())
        reservation = Destination(dest, Path(tempfile.mkdtemp(dir=parent)), force)
        self.temporary.callback(reservation.clean)
        reservation.stage.parent.mkdir()
        self.outputs[dest] = reservation
        return reservation.stage

    def export(self, plan: ExportPlan, *, force: bool) -> dict[str, object]:
        paths = plan.format.paths(plan.path)
        staged = {p: self.output(p, force=force) for p in paths}
        primary = staged[plan.path]
        receipt = self.temporary.enter_context(plan.staged(primary))
        for final, generated in zip(paths, plan.format.paths(primary), strict=True):
            if generated != staged[final]:
                generated.replace(staged[final])
        self.export_group = plan, staged, force
        return receipt

    def publish_array(self, token: str) -> Path:
        stage = self.arrays[token]
        public = Path(stage.reference.path)
        stage.path.replace(public)
        del self.arrays[token]
        return public

    def commit(self) -> None:
        """Install a complete group, undoing replacements in reverse on failure."""
        for path in self.outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with lock_directories(list(self.outputs)), ExitStack() as cleanup, ExitStack() as undo:
            staged_paths = self.export_group[1] if self.export_group else ()
            grouped = {path.parent.resolve() / path.name for path in staged_paths}
            remaining = {p: dest for p, dest in self.outputs.items() if p not in grouped}
            _ = cleanup.enter_context(publishing(*remaining))
            if self.export_group:
                plan, staged, force = self.export_group
                install_export(
                    staged, plan.format, plan.path, force=force, undo=undo, cleanup=cleanup
                )
            for destination in remaining.values():
                # Retire private backups while publication still holds input protection.
                cleanup.callback(destination.clean)
                destination.install(undo)
            undo.pop_all()
