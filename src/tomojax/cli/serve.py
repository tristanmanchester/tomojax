"""Serve a local GUI session over JSON Lines."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def build_parser() -> argparse.ArgumentParser:
    """Describe the stdin/stdout command."""
    return argparse.ArgumentParser(prog="tomojax serve", description=__doc__)


@contextmanager
def protocol_pipes() -> Iterator[tuple[int, int]]:
    """Keep command pipes private across imports, native calls and child exec."""
    reader, writer = os.dup(0), os.dup(1)
    try:
        with Path(os.devnull).open("rb") as empty:
            os.dup2(empty.fileno(), 0)
        os.dup2(2, 1)
        yield reader, writer
    finally:
        os.close(reader)
        os.close(writer)


def main() -> int:
    """Start the production worker on private descriptors."""
    build_parser().parse_args()
    with protocol_pipes() as (reader, writer):
        from tomojax.cli._transport import run

        return run(reader, writer)
