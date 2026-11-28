"""The references one server has issued and the worker objects they name."""

from __future__ import annotations

from typing import Any

from tomojax.cli.execution import new_reference


class Ledger:
    """The live references of one server."""

    def __init__(self) -> None:
        self.live: dict[str, dict[str, Any]] = {}

    def issue(self, kind: str, record: dict[str, Any]) -> str:
        """Name an accepted worker output."""
        ref = new_reference(kind)
        self.live[ref] = record
        return ref

    def lookup(self, ref: str) -> dict[str, Any]:
        """Resolve a live reference, or list the live ones."""
        if ref not in self.live:
            raise ValueError(
                f"Unknown reference {ref}. Live handles: {', '.join(self.live) or 'none'}"
            )
        return self.live[ref]

    def closed(self, ref: str) -> None:
        """Retire an acknowledged close or release."""
        del self.live[ref]

    def lost(self) -> list[str]:
        """Retire every live reference with the worker that held them."""
        records = list(self.live)
        self.live.clear()
        return records
