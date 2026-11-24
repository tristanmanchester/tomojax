"""Bounded lifetime and byte budget for paged MCP replies on one connection."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextvars import Context
from uuid import uuid4


class Parts:
    """Retain ASCII JSON for paging, in insertion order until expiry or eviction."""

    def __init__(self, *, limit: int = 64 << 20, ttl: float = 900) -> None:
        self.limit = limit
        self.ttl = ttl
        self.entries: OrderedDict[str, tuple[str, asyncio.TimerHandle]] = OrderedDict()
        self.nbytes = 0

    def add(self, raw: str) -> str | None:
        """Store a complete ASCII reply if it can fit, returning the first page URI."""
        size = len(raw)
        if size > self.limit:
            return None
        while self.nbytes + size > self.limit:
            self.discard(next(iter(self.entries)))
        call = uuid4().hex
        timer = asyncio.get_running_loop().call_later(
            self.ttl, self.discard, call, context=Context()
        )
        self.entries[call] = raw, timer
        self.nbytes += size
        return f"tomojax://result/{call}/0"

    def discard(self, call: str) -> None:
        """Release the reply and its timer, including on early eviction."""
        if entry := self.entries.pop(call, None):
            raw, timer = entry
            self.nbytes -= len(raw)
            timer.cancel()

    async def read(self, call: str, n: int) -> dict[str, str | None]:
        """Read a page by index. JSON escaping can at most double its ASCII length."""
        entry = self.entries.get(call)
        if entry is None or entry[1].when() <= asyncio.get_running_loop().time():
            raise ValueError("Result is no longer available. Repeat the call to get a new result.")
        raw, _ = entry
        start = n * 8000
        if not 0 <= start < len(raw):
            raise ValueError("Unknown result part. Repeat the call to get a new result.")
        end = start + 8000
        return {
            "data": raw[start:end],
            "next": f"tomojax://result/{call}/{n + 1}" if end < len(raw) else None,
        }
