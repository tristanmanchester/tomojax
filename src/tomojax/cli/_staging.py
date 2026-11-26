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
        if not self.keep_backup:
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
            if self.stage.is_dir() and self.path.is_dir() and not self.path.is_symlink():
                from tomojax._stack_files import tiff_stack_entry

                originals = self.work / "original"
                originals.mkdir()
                for entry in self.path.iterdir():
                    if tiff_stack_entry(entry, self.stage):
                        backup = originals / entry.name
                        entry.replace(backup)
                        undo.callback(self.restore, backup, entry)
                for staged in self.stage.iterdir():
                    destination = self.path / staged.name
                    staged.replace(destination)
                    undo.callback(destination.replace, staged)
                return
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

    def publish_array(self, token: str) -> Path:
        stage = self.arrays[token]
        public = Path(stage.reference.path)
        stage.path.replace(public)
        del self.arrays[token]
        return public

    def commit(self) -> None:
        """Install a complete group, undoing replacements in reverse on failure."""
        with ExitStack() as undo:
            for destination in self.outputs.values():
                destination.install(undo)
            undo.pop_all()
