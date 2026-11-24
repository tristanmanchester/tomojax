"""Reload re-imports the tool package with the presentation modules it uses."""

from __future__ import annotations

import asyncio
import importlib
import sys

import pytest

pytest.importorskip("mcp")

import tomojax.cli
from tomojax.cli.mcp.core import Core


class _Bus:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def publish(self, event: object) -> None:
        self.events.append(event)


class _Core:
    def __init__(self, install) -> None:
        self.install = install
        self.bus = _Bus()


def test_reload_reimports_the_panel_renderer_with_the_tools():
    before = importlib.import_module("tomojax.cli.images")

    def install() -> list[str]:
        importlib.import_module("tomojax.cli.mcp.tools")
        return ["status"]

    fake = _Core(install)
    asyncio.run(Core.reload(fake))  # pyright: ignore[reportArgumentType]
    after = sys.modules["tomojax.cli.images"]
    assert after is not before
    assert vars(tomojax.cli)["images"] is after
    assert len(fake.bus.events) == 1


def test_failed_reload_restores_every_module_it_removed():
    before = importlib.import_module("tomojax.cli.images")
    tools = importlib.import_module("tomojax.cli.mcp.tools")

    def install() -> list[str]:
        importlib.import_module("tomojax.cli.images")
        raise ImportError("broken checkout")

    fake = _Core(install)
    with pytest.raises(ImportError, match="broken checkout"):
        asyncio.run(Core.reload(fake))  # pyright: ignore[reportArgumentType]
    assert sys.modules["tomojax.cli.images"] is before
    assert vars(tomojax.cli)["images"] is before
    assert sys.modules["tomojax.cli.mcp.tools"] is tools
    assert not fake.bus.events
