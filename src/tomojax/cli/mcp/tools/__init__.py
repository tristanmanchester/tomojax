"""Reloadable tool schemas, handlers and MCP results."""

from __future__ import annotations

import asyncio
import base64
from functools import partial
from importlib import resources
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import TYPE_CHECKING, Any, Literal, get_args

from mcp_types import CallToolResult, ImageContent, ResourceLink, TextContent
import msgspec

from tomojax.cli import operations as op, wire_options as w
from tomojax.cli.mcp.supervisor import WorkerError, activity
from tomojax.cli.mcp.tools.budget import bounded
from tomojax.cli.mcp.tools.references import PRIMARY, resolve, wire_arguments
from tomojax.cli.mcp.tools.workflows import MODELS, forms
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
    dry_run: bool = False
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
    for field, help_text in {
        "config": "Expert settings. Use describe for the method's keys.",
        "dry_run": "Validate inputs and return a plan for align or reconstruct.",
        "detail": "Use full for the worker info and provenance when available.",
    }.items():
        if field in root["properties"]:
            root["properties"][field]["description"] = help_text
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


def inspect_directory(path: Path) -> CallToolResult:
    """Summarise acquisition entry points and count TIFF frames in one directory pass."""
    formats = {
        ".xtekct": "Nikon",
        ".txrm": "Zeiss",
        ".txm": "Zeiss",
        ".log": "acquisition log",
        ".info": "acquisition metadata",
        ".xml": "acquisition metadata",
        ".npz": "NumPy archive",
        ".mat": "PSI MATLAB acquisition",
        **dict.fromkeys((".nxs", ".nx", ".hdf", ".hdf5", ".h5"), "HDF5"),
    }
    found: dict[str, str] = {}
    subdirectories: list[str] = []
    directory_count = frames = candidates = 0
    with os.scandir(path) as entries:
        for entry in entries:
            suffix = Path(entry.name).suffix.lower()
            if entry.is_dir():
                directory_count += 1
                if len(subdirectories) < 64:
                    subdirectories.append(entry.name)
            elif suffix in {".tiff", ".tif"} and entry.is_file():
                frames += 1
            elif suffix in formats and entry.is_file():
                candidates += 1
                if len(found) < 64:
                    found[entry.path] = formats[suffix]
    return result(
        {
            "directory": str(path),
            "datasets": dict(sorted(found.items())),
            "candidate_count": candidates,
            "subdirectories": sorted(subdirectories),
            "subdirectory_count": directory_count,
            "tiff_files": frames,
            "omitted": candidates - len(found),
        }
    )


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
                "bytes": record.get("bytes", state["handles"].get(record.get("worker"))),
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
        path = (self.supervisor.base / Path(args.path).expanduser()).resolve()
        if path.is_dir():
            return await asyncio.to_thread(inspect_directory, path)
        g = await self.supervisor.worker()

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
        command = msgspec.to_builtins(args)
        command.pop("detail")
        command.pop("dry_run")

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

        if args.dry_run:
            return await self.dry_run(command, detail=args.detail)
        return await self.execute(command, detail=args.detail, direct=True)

    async def workflow(self, name: str, args: Record) -> CallToolResult:
        """Resolve typed references before the worker pins them at admission."""
        params = msgspec.to_builtins(args)
        dry = params.pop("dry_run")
        detail = params.pop("detail")
        command = params["operation"] if name == "edit_scan" else {"type": name, **params}
        model = MODELS[command["type"]]
        if "path" in command:
            destination = self.supervisor.base / Path(command["path"]).expanduser()
            command["path"] = str(destination.parent.resolve() / destination.name)
        config = command.get("config")
        if config is not None and "type" not in config:
            if model is op.Align:
                config["type"] = "AlignConfig"
            else:
                configs = {
                    cls.__name__.removesuffix("Config").lower(): cls.__name__
                    for cls in get_args(op.Config.__value__)
                }
                method = command["method"]
                if method not in configs:
                    raise ValueError(
                        "Choose a specific method or give config its type from describe."
                    )
                config["type"] = configs[method]
        command = wire_arguments(command, model, base=self.supervisor.base, ledger=self.ledger)
        # Validate nested expert keys here too, preserving their full field paths.
        msgspec.convert(command, type=model)
        if dry:
            return await self.dry_run(command, detail=detail)
        return await self.execute(command, detail=detail)

    async def dry_run(self, command: Object, *, detail: str) -> CallToolResult:
        """Use worker preflight and resolve a plan on the same generation."""
        g = await self.supervisor.worker()
        output = await g.request({"type": "validate", "operation": command})
        if command["type"] in ("align", "reconstruct"):
            output = await g.request({"type": "plan", "operation": command})
        data = output["value"]
        if detail == "full":
            data.update(info=output["info"], manifest=output["manifest"])
        return result(data)

    async def execute(
        self, command: Object, *, detail: str, direct: bool = False
    ) -> CallToolResult:
        """Accept outputs in their owning generation and link committed files."""
        g = await self.supervisor.worker()
        started = time.monotonic()

        def publish(output: Object) -> Object:
            value = output["value"]
            kind: str = value["type"]
            data: Object = {"kind": kind}
            if kind == "files":
                data.update(paths=value["paths"], recovery=value["recovery"])
            else:
                data = self.publish_handle(command, output)
            if detail == "full":
                data.update(info=output["info"], manifest=output["manifest"])
            return data

        data = await g.request(
            command if direct else {"type": "run", "operation": command}, accept=publish
        )
        data["wall_seconds"] = round(time.monotonic() - started, 3)
        job = activity.get()
        if job:
            data.update(devices=sorted(job.devices), notes=job.notes)
        summary = [str(data.get("handle") or "Saved"), data["kind"]]
        if data.get("shape"):
            summary.append(" x ".join(map(str, data["shape"])) + " " + str(data["unit"]))
        summary.extend(
            f"{key} {data[key]}"
            for key in ("method", "effective_iterations", "termination", "calibration")
            if key in data
        )
        if auto := data.get("automatic"):
            summary.extend(auto.get("reasons", []))
        if recovery := data.get("recovery"):
            summary.append(f"Commit recovery {recovery}")
        summary.extend(data.get("paths", []))
        summary.extend(data.get("devices", []))
        summary.append(f"{data['wall_seconds']:g} s")
        summary.extend(data.get("notes", []))
        reply = result(data, text=" · ".join(summary))
        for name in data.get("paths", []):
            path = Path(name)
            reply.content.append(
                ResourceLink(type="resource_link", uri=path.as_uri(), name=path.name)
            )
        return reply

    def publish_handle(self, command: Object, output: Object) -> Object:
        """Issue the session identity while the accepting generation is still live."""
        value = output["value"]
        kind: str = value["type"]
        array = value.get(PRIMARY.get(kind, kind)) or value.get("poses") or {}
        info = output["info"]["summary"]
        summary = {
            "kind": kind,
            "shape": array.get("shape"),
            "unit": array.get("unit"),
            "operation": command["type"],
        }
        if command["type"] == "open":
            summary["path"] = command["path"]
        for key in (
            "method",
            "calibration",
            "automatic",
            "effective_iterations",
            "termination",
            "converged",
            "placement",
        ):
            if key in value or key in info:
                summary[key] = value.get(key, info.get(key))
        record = {
            "kind": kind,
            "summary": summary,
            "manifest": output["manifest"],
            "product": value,
        }
        if "handle" in value:
            alias = {
                "reconstruction": "recon",
                "alignment": "align",
                "simulation": "sim",
            }.get(kind, kind)
            record["worker"] = value["handle"]
        else:
            alias = "array"
            record.update(array["storage"])
            record["bytes"] = Path(record["path"]).stat().st_size
            record["related"] = [
                axis["coordinates"]["values"]["artifact"]
                for axis in array["axes"]
                if axis["coordinates"]["type"] == "coordinates"
                and "artifact" in axis["coordinates"]["values"]
            ]
        ref = self.ledger.issue(alias, record)
        data = {"handle": ref, **summary}
        data["members"] = (
            {
                key: ref + "." + key
                for key, item in value.items()
                if isinstance(item, dict) and ("storage" in item or item.get("type") == "member")
            }
            if alias != "array"
            else {}
        )
        if kind == "alignment":
            data["poses"] = {
                "shape": value["poses"]["shape"],
                "columns": value["poses"]["columns"],
                "rms": info.get("pose_rms"),
            }
        return data

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

            async def release() -> None:
                await g.request(command, accept=closed)
                for artifact in record.get("related", []):
                    await g.request({"type": "release", "artifact": artifact})

            closing = asyncio.create_task(release())
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                closing.add_done_callback(lambda done: done.exception())
                raise
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


def workflow_output() -> Object:
    """Describe handle, file, preflight, error and continuation responses."""
    return {
        "type": "object",
        "properties": {
            "handle": {"type": "string"},
            "kind": {"type": "string"},
            "members": {"type": "object"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "recovery": {"type": "object"},
            "type": {"type": "string"},
            "wall_seconds": {"type": "number"},
            "method": {"type": "string"},
            "manifest": {"type": "object"},
            "info": {"type": "object"},
            "continuation": {"type": "string"},
        },
        "anyOf": [
            {"required": [k]} for k in ("handle", "paths", "type", "message", "continuation")
        ],
    }


def binding(
    cls: type[Record],
    handler: Callable[..., Awaitable[CallToolResult]],
    definitions: Object,
    *,
    name: str | None = None,
    parts: dict[str, list[str]] | None = None,
) -> Object:
    """Construct a plain tool record with msgspec validation and actionable errors."""
    name = name or cls.__name__.lower()
    spec = schema(cls, definitions)

    async def invoke(arguments: dict[str, Any]) -> CallToolResult:
        try:
            reply = await handler(msgspec.convert(arguments, type=cls))
        except (ValueError, OSError) as err:
            reply = tool_error(name, err, spec)
        job = activity.get()
        if job and job.notes and reply.structured_content is not None:
            reply.structured_content.setdefault("notes", job.notes)
        return bounded(reply, parts) if parts is not None else reply

    return {
        "name": name,
        "description": (cls.__doc__ or "")
        + (
            " Handles last until close, worker restart or server exit."
            if name
            in {
                "correct",
                "align",
                "simulate",
                "reconstruct",
                "edit_scan",
                "project",
                "backproject",
            }
            else ""
        ),
        "read_only": name in {"status", "inspect", "describe"},
        "schema": spec,
        "output_schema": workflow_output()
        if name in {*MODELS, "edit_scan", "open"}
        else {"type": "object"},
        "handler": invoke,
    }


def registry(core: Core) -> list[Object]:
    """Build fresh tool descriptions, validators and handlers for this checkout."""
    tools = Tools(core)
    definitions = msgspec.json.decode(
        resources.files("tomojax.cli").joinpath("parameters.json").read_bytes()
    )["$defs"]
    return [
        binding(cls, getattr(tools, cls.__name__.lower()), definitions, parts=core.parts)
        for cls in (Status, Inspect, Open, Describe, Control)
    ] + [
        binding(cls, partial(tools.workflow, name), definitions, name=name, parts=core.parts)
        for name, cls in forms().items()
    ]
