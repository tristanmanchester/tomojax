"""Private working paths and the session's durable commit point."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
from uuid import uuid4

from tomojax.cli.execution import ArrayStage
from tomojax.cli.products import Staged
from tomojax.wire.arrays import Npy


class Stages:
    """One admission owns all its reservations until commit or disposal."""

    def __init__(self, root: Path, index: Path) -> None:
        self.root = root
        self.index = index
        self.arrays: dict[str, ArrayStage] = {}
        self.outputs: dict[Path, Staged] = {}
        self.temporary = ExitStack()

    def array(self) -> ArrayStage:
        token = uuid4().hex
        private = self.root / (token + ".stage.npy")
        stage = ArrayStage(private, Npy(str(self.root / (token + ".npy")), token))
        self.arrays[token] = stage
        self.temporary.callback(private.unlink, missing_ok=True)
        return stage

    def output(self, destination: Path, *, force: bool) -> Path:
        # Canonicalise parents, but replace a destination symlink itself.
        if not destination.name or destination.name == "..":
            raise ValueError("expected an output file or directory name")
        dest = destination.parent.resolve() / destination.name
        if dest.is_relative_to(self.root) or self.root.is_relative_to(dest):
            raise ValueError("durable destination overlaps session storage")
        for prior in self.outputs:
            if dest.is_relative_to(prior) or prior.is_relative_to(dest):
                raise ValueError("overlapping durable destinations")
        parent = next(parent for parent in dest.parents if parent.exists())
        token = ".tomojax-stage-" + uuid4().hex
        work = parent / token
        link = self.index / token
        link.symlink_to(work)
        work.mkdir(mode=0o700)
        (work / "destination").symlink_to(dest)
        stage = work / "new" / dest.name
        self.outputs[dest] = Staged(str(dest), str(stage), force)
        stage.parent.mkdir()
        return stage

    def discard(self) -> None:
        for item in self.outputs.values():
            work = Path(item.staged).parent.parent
            if work.exists():
                shutil.rmtree(work)
            (self.index / work.name).unlink(missing_ok=True)

    def publish_array(self, token: str) -> Path:
        stage = self.arrays[token]
        public = Path(stage.reference.path)
        stage.path.replace(public)
        del self.arrays[token]
        return public
