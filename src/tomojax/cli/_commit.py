"""Serialise durable output groups across workers, with restartable rollback."""

from __future__ import annotations

import ctypes
import errno
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
    force: bool
    identity: tuple[int, int, int] | None


class Directory(msgspec.Struct):
    destination: str
    staged: str
    inode: int


class Journal(msgspec.Struct):
    group: list[Staged]
    members: list[Member]
    directories: list[Directory]
    committed: bool = False


def registry() -> Path:
    """Every server for the same uid uses the same on-disk lock."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local/state/tomojax/commits"


def prepare(group: list[Staged], root: Path) -> Journal:
    members = []
    directories: dict[Path, Directory] = {}
    for item in group:
        dest, stage = Path(item.destination), Path(item.staged)
        if dest.is_relative_to(root) or root.is_relative_to(dest):
            raise ValueError(f"Destination overlaps the commit registry: {dest}")
        if not stage.exists():
            raise FileNotFoundError(f"missing staged output for {dest}")
        exists = os.path.lexists(dest)
        if exists and not item.force:
            raise FileExistsError(f"{dest} exists. Pass force to replace it.")
        if exists and not dest.is_symlink() and dest.is_dir() != stage.is_dir():
            raise ValueError(f"incompatible destination type: {dest}")
        for parent in dest.parents:
            if not parent.exists() and parent not in directories:
                empty = stage.parent.parent / f"directory-{len(directories)}"
                empty.mkdir()
                directories[parent] = Directory(str(parent), str(empty), empty.stat().st_ino)
        entries = output_entries(dest, stage, exists=exists)
        whole_tree = stage.is_dir() and not exists
        if whole_tree:
            directories[dest] = Directory(str(dest), str(stage), stage.stat().st_ino)
        for path, new in entries.items():
            backup = stage.parent.parent / f"backup-{len(members)}"
            members.append(
                Member(
                    str(path),
                    str(new) if new and not whole_tree else None,
                    str(backup),
                    item.force and os.path.lexists(path),
                    item.force,
                    file_identity(new) if new else None,
                )
            )
    return Journal(
        group, members, [directories[p] for p in sorted(directories, key=lambda p: len(p.parts))]
    )


def output_entries(dest: Path, stage: Path, *, exists: bool) -> dict[Path, Path | None]:
    """Include stale slices when replacing an existing TIFF stack."""
    if not stage.is_dir() or (exists and dest.is_symlink()):
        return {dest: stage}
    entries: dict[Path, Path | None] = {}
    if exists:
        entries.update((p, None) for p in dest.iterdir() if tiff_stack_entry(p, stage))
    for child in stage.iterdir():
        path = dest / child.name
        if exists and path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(str(path))
        entries[path] = child
    return entries


def file_identity(path: Path) -> tuple[int, int, int]:
    st = path.lstat()
    return st.st_dev, st.st_ino, st.st_mtime_ns


_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)


def no_replace(stage: Path, dest: Path) -> None:
    """Publish a staged name only if the destination is still free."""
    if _renameat2 is not None:
        result = _renameat2(
            _AT_FDCWD, os.fsencode(stage), _AT_FDCWD, os.fsencode(dest), _RENAME_NOREPLACE
        )
        if result == 0:
            return
        code = ctypes.get_errno()
        if code not in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            raise OSError(code, os.strerror(code), str(dest))
    if stage.is_dir() and not stage.is_symlink():
        raise OSError(f"Atomic directory install requires renameat2(RENAME_NOREPLACE): {dest}")
    try:
        os.link(stage, dest, follow_symlinks=False)
    except FileExistsError as err:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dest)) from err
    stage.unlink()


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


def undo_member(member: Member) -> None:
    dest, backup = Path(member.destination), Path(member.backup)
    claimed = backup.with_suffix(".rollback")
    # Claim the name before disposing of it. A concurrent replacement between
    # lstat and rename must survive too, including a crash during this step.
    if not os.path.lexists(claimed):
        try:
            if member.identity == file_identity(dest):
                dest.replace(claimed)
        except (FileNotFoundError, NotADirectoryError):
            pass
    if os.path.lexists(claimed) and member.identity != file_identity(claimed):
        restore(claimed, dest)
    if os.path.lexists(backup):
        restore(backup, dest)


def restore(saved: Path, dest: Path) -> None:
    """Finish a no-replace restore interrupted between link and unlink."""
    try:
        no_replace(saved, dest)
    except FileExistsError:
        if not os.path.samestat(saved.lstat(), dest.lstat()):
            raise
        saved.unlink()


def resolve(path: Path, journal: Journal) -> dict[str, list[str]]:
    if not journal.committed:
        for member in reversed(journal.members):
            undo_member(member)
        for entry in reversed(journal.directories):
            directory = Path(entry.destination)
            try:
                if not os.path.lexists(entry.staged) and directory.lstat().st_ino == entry.inode:
                    directory.rmdir()
            except OSError as err:
                if err.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ENOTEMPTY, errno.EEXIST):
                    raise
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
        path.with_suffix(".tmp").unlink(missing_ok=True)
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
            install(group, path, report)
        return report


def publish(journal: Journal) -> None:
    for directory in journal.directories:
        no_replace(Path(directory.staged), Path(directory.destination))
    for member in journal.members:
        destination = Path(member.destination)
        if member.existed:
            destination.replace(member.backup)
        if member.staged:
            stage = Path(member.staged)
            if member.force:
                stage.replace(destination)
            else:
                no_replace(stage, destination)


def install(group: list[Staged], path: Path, report: dict[str, list[str]]) -> None:
    """Install under the caller's lock. The on-disk marker decides the outcome."""
    try:
        journal = prepare(group, path.parent)
        persist(path, journal)
        publish(journal)
        journal.committed = True
        persist(path, journal)
    except Exception as err:
        if not path.exists():
            for item in group:
                remove(Path(item.staged).parent.parent)
            raise
        decision = msgspec.json.decode(path.read_bytes(), type=Journal)
        recovered = resolve(path, decision)
        if not decision.committed:
            message = (
                f"destination appeared: {err.filename}"
                if isinstance(err, FileExistsError)
                else str(err)
            )
            raise OSError(f"{message}. Restored: {recovered['restored']}") from err
        report.update(recovered)
        return
    try:
        resolve(path, journal)
    except OSError as err:
        # The decision is final. The next lock holder finishes cleanup.
        report["confirmed"] = [member.destination for member in journal.members]
        report["cleanup_pending"] = [str(err)]


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
