"""Stable SDK, subscription, task and worker lifetimes for one stdio connection."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
import signal
import sys
import tempfile
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools.base import Tool
from mcp.server.mcpserver.tools.tool_manager import ToolManager
from mcp.server.subscriptions import InMemorySubscriptionBus, ToolsListChanged
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, ToolAnnotations

from tomojax import __version__
from tomojax.cli.mcp.guidance import install_guidance
from tomojax.cli.mcp.ledger import Ledger
from tomojax.cli.mcp.supervisor import Supervisor, WorkerLost, WorkerProtocolError
from tomojax.cli.mcp.tasks import Tasks

if TYPE_CHECKING:
    from mcp.server.mcpserver.context import Context

    from tomojax.cli.mcp.supervisor import Object

PACKAGE = "tomojax.cli.mcp.tools"
INSTRUCTIONS = """Use TomoJAX for parallel-beam CT, cone-beam CT and laminography.
Start with inspect to identify an acquisition and open to load it. Use describe for
parameter choices and units. status lists devices, current jobs and live data.
A bare handle supplies the whole object to save/export, or the primary array to an array
field (simulation phantom, reconstruction volume, alignment working image, scan projections).
For a scan operation use a scan handle or .scan on a simulation, alignment or reconstruction.
Select another array with .member, e.g. .poses (alpha, beta, phi in degrees, then x, z, y
in the geometry length unit).
Handles last until closed, worker restart or server exit. Use file.h5::/dataset for
an HDF5 dataset and ./@name for a literal path. Relative paths use the project directory.
The usual workflow is inspect, open, correct, align the centre, reconstruct, view slices
and save. Volumes use (z, y, x), projections (view, v, u). Voxel and detector spacings
share one physical length unit. Check slices before trusting a residual. Use dry_run
before large jobs on tools that support it. See tools/list for installed workflows."""


class Adapter(Tool):
    """Let msgspec own validation and schema while the SDK owns tools/call."""

    async def run(
        self,
        arguments: Object,
        context: Context[Any, Any],
        convert_result: bool = False,  # noqa: FBT001, FBT002 - SDK signature
    ) -> CallToolResult:
        """Dispatch validated arguments through the installed handler."""
        del context, convert_result
        try:
            return await self.fn(arguments)
        except (WorkerLost, WorkerProtocolError) as err:
            raise MCPError(code=-32603, message=str(err)) from err


class Server(MCPServer):
    """Replace a whole SDK registry synchronously after candidate handlers validate."""

    def install(self, records: list[Object]) -> None:
        """SDK 2.3 accepts explicit Tool objects only at manager construction."""
        tools = []
        for record in records:
            tool = Adapter.from_function(
                record["handler"],
                name=record["name"],
                description=record["description"],
                annotations=ToolAnnotations(read_only_hint=record["read_only"]),
            )
            tool.parameters = record["schema"]
            tool.fn_metadata.output_schema = record["output_schema"]
            tools.append(tool)
        if len({t.name for t in tools}) != len(tools):
            raise ValueError("Duplicate tool name")
        self._tool_manager = ToolManager(tools=tools)


class Core:
    """Stable state passed into each generation of the tool package."""

    def __init__(self, supervisor: Supervisor) -> None:
        self.supervisor = supervisor
        self.bus = InMemorySubscriptionBus()
        self.parts: dict[str, list[str]] = {}
        self.tasks = Tasks(self.parts)
        self.server = Server(
            "tomojax",
            version=__version__,
            instructions=INSTRUCTIONS,
            extensions=[self.tasks],
            subscriptions=self.bus,
        )
        self.server.resource("tomojax://result/{call}/{n}", mime_type="application/json")(
            self.read_part
        )
        install_guidance(self.server, supervisor)
        self.install()

    def read_part(self, call: str, n: int) -> dict[str, str | None]:
        """Read successive chunks of the serialized original CallToolResult."""
        chunks = self.parts.get(call, [])
        if not 0 <= n < len(chunks):
            raise ValueError("Unknown result part")
        return {
            "data": chunks[n],
            "next": f"tomojax://result/{call}/{n + 1}" if n + 1 < len(chunks) else None,
        }

    def install(self) -> list[str]:
        """Load all handlers before swapping the live tool registry."""
        records = importlib.import_module(PACKAGE).registry(self)
        self.server.install(records)
        self.tasks.eligible = {r["name"] for r in records if r.get("long")}
        return [r["name"] for r in records]

    async def reload(self) -> Object:
        """Publish a fresh registry on the existing subscription bus."""
        previous = {
            k: v for k, v in sys.modules.items() if k == PACKAGE or k.startswith(PACKAGE + ".")
        }
        # A fresh bytecode cache ensures even same-size edits in one second
        # load from source. This synchronous import does not modify the checkout.
        with tempfile.TemporaryDirectory() as cache:
            old_cache = sys.pycache_prefix
            sys.pycache_prefix = cache
            try:
                for name in previous:
                    del sys.modules[name]
                importlib.invalidate_caches()
                names = self.install()
            except Exception:
                for name in tuple(sys.modules):
                    if name == PACKAGE or name.startswith(PACKAGE + "."):
                        del sys.modules[name]
                sys.modules.update(previous)
                vars(sys.modules["tomojax.cli.mcp"])["tools"] = previous[PACKAGE]
                raise
            finally:
                sys.pycache_prefix = old_cache
        await self.bus.publish(ToolsListChanged())
        return {"tools": names}


async def run() -> None:
    """Start the worker concurrently with MCP discovery, then clean up both lifetimes."""
    base = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    ledger = Ledger(state / "tomojax/mcp")
    supervisor = Supervisor(ledger, base, cache / "tomojax/mcp")
    loop = asyncio.get_running_loop()
    terminate = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, terminate.set)
    signal_task = asyncio.create_task(terminate.wait())
    try:
        core = Core(supervisor)
        transport = asyncio.create_task(core.server.run_stdio_async())
        try:
            await asyncio.wait((signal_task, transport), return_when=asyncio.FIRST_COMPLETED)
            if transport.done():
                await transport
        finally:
            await core.tasks.close()
    finally:
        signal_task.cancel()
        await supervisor.close()
        ledger.close()
        loop.remove_signal_handler(signal.SIGTERM)
        if terminate.is_set():
            # The SDK's stdin thread can still be blocked in readline. Honour
            # termination once our process group and leases have been released.
            signal.raise_signal(signal.SIGTERM)
