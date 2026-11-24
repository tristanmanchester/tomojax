"""Reloadable tool schemas, handlers and MCP results."""

from __future__ import annotations

import asyncio
import base64
from importlib import resources
import json
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import CallToolResult, ImageContent, TextContent
import msgspec

from tomojax.cli import operations as op, wire_options as w
from tomojax.cli.mcp.supervisor import WorkerError
from tomojax.cli.mcp.tools.references import PRIMARY, resolve, wire_arguments
from tomojax.wire.arrays import Id, Record

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tomojax.cli.mcp.core import Core
    from tomojax.cli.mcp.supervisor import Object


class Status(Record):
    """List versions, devices, jobs, live references and worker recovery information."""

    detail: Literal["summary", "full"] = "summary"


class Inspect(Record):
    """Inspect a file or find datasets in a directory. Request a central preview with preview."""

    path: Id
    preview: bool = False
    detail: Literal["summary", "full"] = "summary"


class Open(Record):
    """Open data as a handle. It lasts until close, worker restart or server exit."""

    path: Id
    kind: op.RootKind = "scan"
    flats: float | str | None = None
    darks: float | str | None = None
    angles: str | None = None
    detector: w.Detector | None = None
    pixel_size: float | tuple[float, float] | None = None
    grid: w.Grid | None = None
    geometry: w.DeclaredGeometry | None = None
    exposure_times: float | str | None = None
    flat_exposure_times: float | str | None = None
    data_path: str | None = None
    image_key_path: str | None = None
    angles_path: str | None = None
    apply_poses: bool = True
    detail: Literal["summary", "full"] = "summary"


class Describe(Record):
    """Show expert parameters for an operation or a specific solver method."""

    operation: Id
    method: str | None = None


class Control(Record):
    """Cancel, pause or resume a job by target ID, close a handle, reload tools or restart."""

    action: Literal["cancel", "pause", "resume", "close", "reload", "restart"]
    target: str | None = None


def result(data: Object, *, text: str | None = None, error: bool = False) -> CallToolResult:
    """Return machine-readable fields together with readable text."""
    return CallToolResult(
        content=[TextContent(type="text", text=text or json.dumps(data, indent=2))],
        structured_content=data,
        is_error=error,
    )


def schema(cls: type[Record], definitions: Object) -> Object:
    """Inline the root and enrich matching fields from the worker's parameter document."""
    generated = msgspec.json.schema(cls)
    nested = generated["$defs"]
    root = nested.pop(cls.__name__)
    for name, struct in [(cls.__name__, root), *nested.items()]:
        properties = definitions.get(name, {}).get("properties", {})
        for key, entry in struct["properties"].items():
            entry.update(
                {
                    k: v
                    for k, v in properties.get(key, {}).items()
                    if k in {"description", "x-unit", "x-choices"}
                }
            )
    return {**root, **({"$defs": nested} if nested else {})}


def preview_image(path: Path) -> ImageContent:
    """Read a central projection or volume slice using the CLI's orientation and scaling."""
    from io import BytesIO

    from PIL import Image

    from tomojax.cli._previews import write_previews

    with tempfile.TemporaryDirectory() as directory:
        previews = write_previews(path, directory)
        with Image.open(next(iter(previews.values()))) as image:
            image.thumbnail((1024, 1024))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
    return ImageContent(
        type="image", mime_type="image/png", data=base64.b64encode(buffer.getvalue()).decode()
    )


def tree(path: Path) -> Object:
    """List HDF5 dataset names, shapes and dtypes without loading their values."""
    import h5py

    datasets = {}
    if path.is_file() and h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:

            def visit(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset):
                    datasets["/" + name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

            file.visititems(visit)
    return datasets


async def inspect_directory(
    path: Path, report: Callable[[Path], Awaitable[Object]]
) -> CallToolResult:
    """Collect recognised child acquisitions using the worker's format detection."""
    found = {}
    for child in sorted(path.iterdir()):
        try:
            found[str(child)] = await report(child)
        except WorkerError as failure:
            if failure.kind != "input":
                raise
    return result({"directory": str(path), "datasets": found})


class Tools:
    """Translate agent input and worker outputs for the installed workflows."""

    def __init__(self, core: Core) -> None:
        self.core = core
        self.supervisor = core.supervisor
        self.ledger = self.supervisor.ledger

    async def status(self, args: Status) -> CallToolResult:
        """Read the worker's actual inventory, allocated bytes and active request states."""
        g = await self.supervisor.worker()
        ready = g.ready.result()
        state = await g.request({"type": "status"})
        handles = {}
        for ref, record in self.ledger.live.items():
            handles[ref] = {
                **record["summary"],
                "bytes": state["handles"].get(record.get("worker")),
            }
            if args.detail == "full":
                handles[ref]["manifest"] = record.get("manifest")
        return result(
            {
                "versions": ready["versions"],
                "devices": ready["devices"],
                "allocators": state["allocators"],
                "jobs": state["requests"],
                "handles": handles,
                "python_namespace_bytes": 0,
                "recovery": self.supervisor.report,
            }
        )

    async def inspect(self, args: Inspect) -> CallToolResult:
        """Inspect through the worker with directory discovery and unrecognised-HDF5 fallback."""
        g = await self.supervisor.worker()
        path = (self.supervisor.base / Path(args.path).expanduser()).resolve()

        async def report(item: Path) -> Object:
            output = await g.request({"type": "inspect", "path": str(item)})
            full = output["value"]["report"]
            if args.detail == "full":
                return full
            return {
                k: full[k]
                for k in (
                    "input_path",
                    "valid",
                    "issues",
                    "projection",
                    "volume",
                    "geometry",
                    "angles",
                    "flats_darks",
                )
                if k in full
            }

        try:
            data = await report(path)
        except WorkerError as err:
            if err.kind != "input":
                raise
            if path.is_dir():
                return await inspect_directory(path, report)
            datasets = await asyncio.to_thread(tree, path)
            if not datasets:
                raise
            return result(
                {
                    "path": str(path),
                    "datasets": datasets,
                    "next": "Select data_path and angles_path for open.",
                }
            )
        shape = data.get("projection", {}).get("shape") or data.get("volume", {}).get("shape")
        issues = "; ".join(data.get("issues", []))
        reply = result(data, text=f"{path.name}, shape {shape}. {issues or 'Inspection passed.'}")
        if args.preview:
            reply.content.append(await asyncio.to_thread(preview_image, path))
        return reply

    async def open(self, args: Open) -> CallToolResult:
        """Publish the external identity as part of accepting the open terminal."""
        g = await self.supervisor.worker()
        command = msgspec.to_builtins(args)
        command.pop("detail")

        path = resolve(args.path, base=self.supervisor.base, ledger=self.ledger, field="source")
        assert isinstance(path, dict)
        if path["type"] == "member":
            raise ValueError("open takes a file path. Use the existing reference directly.")
        if path["type"] == "hdf5":
            if args.data_path and args.data_path != path["dataset"]:
                raise ValueError("data_path disagrees with the dataset in path")
            command["data_path"] = path["dataset"]
        command["path"] = path["path"]
        command = wire_arguments(
            {"type": "open", **command}, op.Open, base=self.supervisor.base, ledger=self.ledger
        )

        def publish(output: Object) -> Object:
            value = output["value"]
            kind: str = value["type"]
            primary = value.get(PRIMARY[kind]) or {}
            summary = {
                "kind": kind,
                "shape": primary.get("shape"),
                "unit": primary.get("unit"),
                "path": command["path"],
            }
            alias = {"reconstruction": "recon", "alignment": "align", "simulation": "sim"}.get(
                kind, kind
            )
            ref = self.ledger.issue(
                alias,
                {
                    "kind": kind,
                    "worker": value["handle"],
                    "summary": summary,
                    "manifest": output["manifest"],
                },
            )
            data = {"handle": ref, **summary}
            if args.detail == "full":
                data.update(info=output["info"], manifest=output["manifest"])
            return data

        data = await g.request(command, accept=publish)
        return result(
            data,
            text=f"{data['handle']} opened {data['path']}, "
            f"shape {data['shape']}, units {data['unit']}.",
        )

    async def describe(self, args: Describe) -> CallToolResult:
        """Select operation and config keys from the running worker's real schema."""
        g = await self.supervisor.worker()
        definitions = (await g.request({"type": "describe"}))["parameters"]["$defs"]
        names = {
            d["properties"]["type"]["enum"][0]: k
            for k, d in definitions.items()
            if "type" in d["properties"]
        }
        if args.operation not in names:
            raise ValueError(f"Unknown operation. Choose from {', '.join(sorted(names))}")
        operation = definitions[names[args.operation]]
        entry = operation
        if args.method:
            methods = operation["properties"].get("method", {}).get("x-choices", [])
            if args.method not in methods:
                raise ValueError(f"method must be one of {methods}")
            config = next(
                (k for k in definitions if k.lower() == args.method.lower() + "config"), None
            )
            if args.operation == "align" and args.method == "joint":
                config = "AlignConfig"
            if config:
                entry = definitions[config]
        return result(
            {
                "operation": args.operation,
                "method": args.method,
                "parameters": {k: v for k, v in entry["properties"].items() if k != "type"},
            }
        )

    async def control(self, args: Control) -> CallToolResult:
        """Apply supervisor or off-queue worker controls."""
        if args.action == "reload":
            return result(await self.core.reload())
        if args.action == "restart":
            return result(await self.supervisor.restart())
        if not args.target:
            raise ValueError("target is required. Use status for handle and job IDs.")
        g = await self.supervisor.worker()
        if args.action == "close":
            record = self.ledger.lookup(args.target)
            command = (
                {"type": "release", "artifact": record["artifact"]}
                if "artifact" in record
                else {"type": "close", "handle": record["worker"]}
            )
            ref = args.target

            def closed(reply: Object) -> Object:
                self.ledger.closed(ref)
                return reply

            await g.request(command, accept=closed)
        else:
            await g.request({"type": "control", "action": args.action, "target": args.target})
        return result({"action": args.action, "target": args.target})


def tool_error(name: str, err: ValueError | OSError, spec: Object) -> CallToolResult:
    """Preserve worker categories, nested paths and enumerated parameter choices."""
    kind = err.kind if isinstance(err, WorkerError) else "invalid_parameter"
    path = err.path if isinstance(err, WorkerError) else None
    message = str(err)
    if isinstance(err, msgspec.ValidationError):
        match = re.search(r" - at `\$\.(.+?)`$", message)
        if match:
            path = match[1]
    choices = spec["properties"].get(path, {}).get("enum")
    hints = {
        "out_of_memory": "Try dry_run, slab reconstruction, binning or different devices.",
        "cancelled": "Resubmit when ready.",
        "superseded": "Use the newer job.",
        "input": "Inspect the input path and metadata.",
        "internal": "Check status for worker and device information.",
    }
    next_step = hints.get(kind, f"Check the {name} tool parameters and retry.")
    if choices:
        next_step = "Choose one of " + ", ".join(map(str, choices)) + "."
    data = {"kind": kind, "path": path, "message": message, "next": next_step}
    return result(data, text=f"{kind}: {message}. {next_step}", error=True)


def binding(
    cls: type[Record], handler: Callable[..., Awaitable[CallToolResult]], definitions: Object
) -> Object:
    """Construct a plain tool record with msgspec validation and actionable errors."""
    name = cls.__name__.lower()
    spec = schema(cls, definitions)

    async def invoke(arguments: dict[str, Any]) -> CallToolResult:
        try:
            return await handler(msgspec.convert(arguments, type=cls))
        except (ValueError, OSError) as err:
            return tool_error(name, err, spec)

    return {
        "name": name,
        "description": cls.__doc__,
        "read_only": name in {"status", "inspect", "describe"},
        "schema": spec,
        "output_schema": {"type": "object"},
        "handler": invoke,
    }


def registry(core: Core) -> list[Object]:
    """Build fresh tool descriptions, validators and handlers for this checkout."""
    tools = Tools(core)
    definitions = msgspec.json.decode(
        resources.files("tomojax.cli").joinpath("parameters.json").read_bytes()
    )["$defs"]
    return [
        binding(cls, getattr(tools, cls.__name__.lower()), definitions)
        for cls in (Status, Inspect, Open, Describe, Control)
    ]
