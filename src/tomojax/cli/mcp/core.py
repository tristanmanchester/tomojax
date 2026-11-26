"""Stable SDK, subscription, task and worker lifetimes for one stdio connection."""

from __future__ import annotations

import asyncio
import importlib
import itertools
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
from tomojax.cli.mcp.parts import Parts
from tomojax.cli.mcp.supervisor import Supervisor, WorkerLost, WorkerProtocolError
from tomojax.cli.mcp.tasks import Tasks

if TYPE_CHECKING:
    from mcp.server.mcpserver.context import Context

    from tomojax.cli.mcp.supervisor import Object

PACKAGE = "tomojax.cli.mcp.tools"
# Presentation modules the tools import. Reload re-imports them with the tool
# package; wire and worker modules change only with a worker restart.
RELOADED = (
    PACKAGE,
    "tomojax.cli.images",
    "tomojax.cli._marks",
    "tomojax.cli._previews",
    "tomojax.cli.mcp.guidance",
)


def _reloaded(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in RELOADED)


INSTRUCTIONS = """Use TomoJAX for parallel-beam CT, cone-beam CT and laminography.
Start with inspect to identify an acquisition and open to load it. Use describe for
parameter choices and units. status lists devices, current jobs and live data.
A bare handle supplies the whole object to save/export, or the primary array to an array
field (simulation phantom, reconstruction volume, alignment working image, scan projections).
For a scan operation use a scan handle or .scan on a simulation, alignment or reconstruction.
Select another array with .member, e.g. .poses (alpha, beta, phi in degrees, then x, z, y
in the geometry length unit).
Handles last until closed, worker restart or server exit. Array fields accept handle members,
session arrays, .npy files and file.h5::/dataset in storage order. For acquisitions, open
and select .projections, or open with kind=frames and select .counts. For a first look use
inspect with preview=true, or preview with correction_preview for corrected rows.
View and project.volume also accept saved TomoJAX volumes by path in logical (z, y, x) order.
Workflow source fields accept acquisition paths. Relative paths use the project directory.
Use ./@name for a literal path starting with @.
The usual workflow is inspect, open, correct, align the center, reconstruct, view slices
and save. Volumes use (z, y, x), projections (view, v, u). Voxel and detector spacings
share one physical length unit. Check slices before trusting a residual. Use dry_run
before large jobs on tools that support it. See tools/list for installed workflows.
To show the user a structure, draw view marks in array indices and ask; make the agreed
region a feature and outline it with view features. For sweeps, loops or custom measures,
use one python call over the same handles."""


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
                annotations=ToolAnnotations(
                    read_only_hint=record["read_only"],
                    # The spec's default is destructive; most tools only add handles.
                    destructive_hint=None if record["read_only"] else record["destructive"],
                    open_world_hint=False,  # local files and the local worker only
                ),
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
        self.parts = Parts()
        self.tasks = Tasks(self.parts)
        # Where each returned view image's planes sit, by render ID, for pixel marks.
        self.renders: dict[str, Any] = {}
        self.render_count = itertools.count(1)
        self.server = Server(
            "tomojax",
            version=__version__,
            instructions=INSTRUCTIONS,
            extensions=[self.tasks],
            subscriptions=self.bus,
        )
        self.server.resource("tomojax://result/{call}/{n}", mime_type="application/json")(
            self.parts.read
        )
        install_guidance(self.server, supervisor)
        self.install()

    def install(self) -> list[str]:
        """Load all handlers before swapping the live tool registry."""
        records = importlib.import_module(PACKAGE).registry(self)
        self.server.install(records)
        self.tasks.eligible = {r["name"] for r in records if r.get("long")}
        return [r["name"] for r in records]

    async def reload(self) -> Object:
        """Publish a fresh registry on the existing subscription bus."""
        previous = {k: v for k, v in sys.modules.items() if _reloaded(k)}
        # A fresh bytecode cache ensures even same-size edits in one second
        # load from source. This synchronous import does not modify the checkout.
        with tempfile.TemporaryDirectory() as cache:
            old_cache = sys.pycache_prefix
            sys.pycache_prefix = cache
            try:
                for name in previous:
                    del sys.modules[name]
                    # `from package import module` reads the package attribute first. A
                    # reloaded parent keeps its attributes: in-flight calls still use it.
                    parent, _, child = name.rpartition(".")
                    owner = sys.modules.get(parent)
                    if parent not in previous and owner is not None and child in vars(owner):
                        delattr(owner, child)
                importlib.invalidate_caches()
                names = self.install()
            except Exception:
                for name in tuple(sys.modules):
                    if _reloaded(name):
                        del sys.modules[name]
                sys.modules.update(previous)
                for name, module in previous.items():
                    parent, _, child = name.rpartition(".")
                    if parent in sys.modules:
                        setattr(sys.modules[parent], child, module)
                raise
            finally:
                sys.pycache_prefix = old_cache
        await self.bus.publish(ToolsListChanged())
        return {"tools": names}


async def run() -> None:
    """Start the worker concurrently with MCP discovery, then clean up both lifetimes."""
    base = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
    supervisor = Supervisor(Ledger(), base)
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
        loop.remove_signal_handler(signal.SIGTERM)
        if terminate.is_set():
            # The SDK's stdin thread can still be blocked in readline. Honor
            # termination once the worker has been stopped.
            signal.raise_signal(signal.SIGTERM)
