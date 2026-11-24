"""Serialise durable output groups across workers, with restartable rollback."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys

import msgspec

from tomojax._stack_files import tiff_stack_entry
from tomojax.cli.products import Staged


class Member(msgspec.Struct):
    destination: str
    staged: str | None
    backup: str
    existed: bool


class Journal(msgspec.Struct):
    group: list[Staged]
    members: list[Member]
    directories: list[str]
    committed: bool = False


def registry() -> Path:
    """Every server for the same uid uses the same on-disk lock."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local/state/tomojax/commits"


def prepare(group: list[Staged]) -> Journal:
    members = []
    directories: set[Path] = set()
    for item in group:
        dest, stage = Path(item.destination), Path(item.staged)
        if not stage.exists():
            raise FileNotFoundError(f"missing staged output for {dest}")
        exists = os.path.lexists(dest)
        if exists and not item.force:
            raise FileExistsError(f"{dest} exists. Pass force to replace it.")
        if exists and not dest.is_symlink() and dest.is_dir() != stage.is_dir():
            raise ValueError(f"incompatible destination type: {dest}")
        directories.update(p for p in dest.parents if not p.exists())
        entries: dict[Path, Path | None] = {dest: stage}
        if stage.is_dir() and not dest.is_symlink():
            if exists:
                entries = {p: None for p in dest.iterdir() if tiff_stack_entry(p, stage)}
            else:
                directories.add(dest)
                entries = {}
            entries.update({dest / p.name: p for p in stage.iterdir()})
        for path, new in entries.items():
            if path != dest and path.is_dir() and not path.is_symlink():
                raise IsADirectoryError(str(path))
            backup = stage.parent.parent / f"backup-{len(members)}"
            members.append(
                Member(str(path), str(new) if new else None, str(backup), os.path.lexists(path))
            )
    return Journal(group, members, sorted(map(str, directories), key=lambda s: len(Path(s).parts)))


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def persist(path: Path, journal: Journal) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(msgspec.json.encode(journal))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve(path: Path, journal: Journal) -> dict[str, list[str]]:
    if not journal.committed:
        for member in reversed(journal.members):
            dest, backup = Path(member.destination), Path(member.backup)
            if os.path.lexists(backup):
                remove(dest)
                backup.replace(dest)
            elif not member.existed:
                remove(dest)
        for name in reversed(journal.directories):
            directory = Path(name)
            if directory.exists():
                directory.rmdir()
    for item in journal.group:
        remove(Path(item.staged).parent.parent)
    path.unlink()
    return {
        "confirmed" if journal.committed else "restored": [m.destination for m in journal.members]
    }


def commit(
    group: list[Staged], *, root: Path | None = None, abandoned: tuple[str, ...] = ()
) -> dict[str, list[str]]:
    """Recover the preceding decision before inspecting any destination of this group."""
    root = root or registry()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / "lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "journal.json"
        report = (
            resolve(path, msgspec.json.decode(path.read_bytes(), type=Journal))
            if path.exists()
            else {}
        )
        for name in abandoned:
            work = Path(name)
            marker = work / "destination"
            if marker.is_symlink():
                report.setdefault("discarded", []).append(str(marker.readlink()))
            remove(work)
        if group:
            journal = prepare(group)
            persist(path, journal)
            try:
                for name in journal.directories:
                    Path(name).mkdir()
                for member in journal.members:
                    destination = Path(member.destination)
                    if member.existed:
                        destination.replace(member.backup)
                    if member.staged:
                        Path(member.staged).replace(destination)
            except Exception as err:
                restored = resolve(path, journal)
                raise OSError(f"{err}. Restored: {restored['restored']}") from err
            journal.committed = True
            persist(path, journal)
            resolve(path, journal)
        return report


def helper(group: list[Staged], *, abandoned: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Commit in a fresh interpreter with none of the worker's file descriptors."""
    child = subprocess.run(
        [sys.executable, "-m", "tomojax.cli._commit"],
        input=msgspec.json.encode((group, abandoned)),
        capture_output=True,
        close_fds=True,
        check=False,
    )
    if child.returncode:
        raise OSError(child.stderr.decode(errors="replace").strip() or "Commit helper exited")
    return msgspec.json.decode(child.stdout)


if __name__ == "__main__":
    try:
        group, abandoned = msgspec.json.decode(
            sys.stdin.buffer.read(), type=tuple[list[Staged], tuple[str, ...]]
        )
        report = commit(group, abandoned=abandoned)
    except Exception as failure:
        sys.exit(str(failure))
    sys.stdout.buffer.write(msgspec.json.encode(report))
