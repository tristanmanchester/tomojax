"""Durable reference identities and the handles owned by one server."""

from __future__ import annotations

import base64
from contextlib import closing
import fcntl
from pathlib import Path
import re
import secrets
import sqlite3
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Iterable

ID = re.compile(r"@([a-z]+)-([1-9][0-9]*)-([a-z2-7]{4})")


class Ledger:
    """SQLite serialises issuance across servers. An owner's flock survives PID reuse."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner = uuid4().hex
        self.lock = (directory / self.owner).open("xb")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.path = directory / "ledger"
        self.db = sqlite3.connect(self.path)
        self.live: dict[str, dict[str, Any]] = {}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("CREATE TABLE IF NOT EXISTS settings(tag TEXT)")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS ids("
                "n INTEGER PRIMARY KEY,kind TEXT,owner TEXT,state TEXT)"
            )
            row = self.db.execute("SELECT tag FROM settings").fetchone()
            if row:
                self.tag = row[0]
            else:
                self.tag = base64.b32encode(secrets.token_bytes(3)).decode().lower()[:4]
                self.db.execute("INSERT INTO settings VALUES(?)", (self.tag,))
            for (owner,) in self.db.execute(
                "SELECT DISTINCT owner FROM ids WHERE state IN ('live','reserved')"
            ):
                path = directory / owner
                try:
                    with path.open("rb") as fd:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        path.unlink()
                except FileNotFoundError:
                    pass
                except BlockingIOError:
                    continue
                self.db.execute(
                    "UPDATE ids SET state='lost' WHERE owner=? AND state IN ('reserved','live')",
                    (owner,),
                )

    def issue(self, kind: str, record: dict[str, Any]) -> str:
        """Assign one persistent identity to an accepted worker output."""
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO ids(kind,owner,state) VALUES(?,?,'live')", (kind, self.owner)
            )
        ref = f"@{kind}-{cursor.lastrowid}-{self.tag}"
        self.live[ref] = record
        return ref

    def lookup(self, ref: str) -> dict[str, Any]:
        """Resolve only this server's live references and explain stale IDs."""
        if ref in self.live:
            return self.live[ref]
        match = ID.fullmatch(ref)
        if match:
            kind, number, tag = match.groups()
            if tag != self.tag:
                raise ValueError(f"{ref} belongs to an earlier ledger. Open the input again.")
            row = self.db.execute(
                "SELECT state FROM ids WHERE n=? AND kind=?", (number, kind)
            ).fetchone()
            if row:
                reason = "live in another server" if row[0] == "live" else row[0]
                raise ValueError(f"{ref} is {reason}. Open the input again.")
        raise ValueError(f"Unknown reference {ref}. Live handles: {', '.join(self.live) or 'none'}")

    def closed(self, ref: str) -> None:
        """Retire an acknowledged close or release."""
        with self.db:
            self.db.execute("UPDATE ids SET state='closed' WHERE n=?", (ref.split("-")[1],))
        del self.live[ref]

    def lost(self) -> list[str]:
        """Retire every live reference owned by this generation."""
        with self.db:
            self.db.execute(
                "UPDATE ids SET state='lost' WHERE owner=? AND state IN ('reserved','live')",
                (self.owner,),
            )
        records = list(self.live)
        self.live.clear()
        return records

    def close(self) -> None:
        """Release the server's ledger lease."""
        self.lost()
        self.db.close()
        Path(self.lock.name).unlink()
        self.lock.close()


def reserve(kind: str, ledger: tuple[str, str] | None) -> str:
    """Reserve a final identifier before Python regains control."""
    if ledger is None:
        return f"{kind}-{uuid4().hex}"
    path, owner = ledger
    with closing(sqlite3.connect(path)) as db, db:
        cursor = db.execute(
            "INSERT INTO ids(kind,owner,state) VALUES(?,?,'reserved')", (kind, owner)
        )
        (tag,) = db.execute("SELECT tag FROM settings").fetchone()
        return f"@{kind}-{cursor.lastrowid}-{tag}"


def activate(ledger: tuple[str, str], names: Iterable[str]) -> None:
    """Activate the worker's entire group in the transaction that publishes its roots."""
    path, owner = ledger
    with closing(sqlite3.connect(path)) as db, db:
        for ref in names:
            number = ref.split("-")[1]
            updated = db.execute(
                "UPDATE ids SET state='live' WHERE n=? AND owner=? AND state='reserved'",
                (number, owner),
            )
            if not updated.rowcount:
                raise ValueError(f"Missing reservation for {ref}")
