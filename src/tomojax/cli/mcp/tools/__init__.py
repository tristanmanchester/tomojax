"""Reloadable tool schemas, handlers and MCP results."""

from __future__ import annotations

import asyncio
from functools import partial
from importlib import resources
import json
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, Literal, get_args

from mcp_types import CallToolResult, ResourceLink, TextContent
import msgspec

from tomojax._parameters import method_fit
from tomojax._typed_arrays import object_mapping
from tomojax.cli import operations as op, wire_options as w
from tomojax.cli._runtime import beyond_detector, solve_summary
from tomojax.cli.mcp.guidance import operation_parameters
from tomojax.cli.mcp.supervisor import WorkerError, activity
from tomojax.cli.mcp.tools import seeing
from tomojax.cli.mcp.tools.budget import bounded
from tomojax.cli.mcp.tools.files import inspect_directory, preview_image, tree
from tomojax.cli.mcp.tools.references import PRIMARY, resolve, wire_arguments
from tomojax.cli.mcp.tools.schemas import expected, schema
from tomojax.cli.mcp.tools.workflows import MODELS, forms
from tomojax.wire.arrays import Id, Record

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tomojax.cli.mcp.core import Core
    from tomojax.cli.mcp.parts import Parts
    from tomojax.cli.mcp.supervisor import Object


class Status(Record):
    """List live handles and job IDs.

    Check device memory and worker versions before starting or retrying a job.
    """

    detail: Literal["summary", "full"] = "summary"


class Inspect(Record):
    """List the scan files and checkpoints in a directory, or inspect one of them.

    For an acquisition, check shapes, geometry, angles and flat/dark metadata before opening
    it; set preview for a central image. For a checkpoint, read the run it resumes.
    """

    path: Id
    preview: bool = False
    detail: Literal["summary", "full"] = "summary"


class Open(Record):
    """Load a scan, raw frames or a saved TomoJAX result.

    Use kind to select the object and supply missing acquisition metadata. The handle lasts
    until close, worker restart or server exit.
    """

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
    apply_poses: bool | msgspec.UnsetType = msgspec.UNSET
    dry_run: bool = False
    detail: Literal["summary", "full"] = "summary"


class Python(Record):
    """Run Python in the worker, next to the loaded data: the path for compound work.

    Use it for sweeps, loops, custom measures, geometry or solvers over live handles,
    instead of many typed calls. In scope: tj, np, handles (ID to object; a feature's
    .mask), register(obj or array) -> ID, register_feature(mask, like=ID or object,
    name=...) -> ID, view(source, mode=, axis=, index=, roi=, stride=, other=, marks=,
    features=, ticks=) with the view tool's arguments, and show(plane, marks=). The last
    expression is the result; arrays over 16 values come back summarized, so register
    big ones. Errors give the exception and your lines; set traceback for the full
    trace. Variables persist between calls; reset or a worker restart clears them and
    restart also drops handles. Long runs can be canceled at TomoJAX's work boundaries.
    See tomojax://guide/python.
    """

    code: str = ""
    reset: bool = False
    traceback: bool = False


class Describe(Record):
    """Look up parameters, defaults, units and choices before configuring an operation.

    Supply method to see its solver settings.
    """

    operation: Id
    method: str | None = None


class Control(Record):
    """Pause, resume or cancel a worker job by its status ID.

    Close a handle to release it, reload tool definitions, or restart the worker to discard its
    handles and Python namespace.
    """

    action: Literal["cancel", "pause", "resume", "close", "reload", "restart"]
    target: str | None = None


def result(data: Object, *, text: str | None = None, error: bool = False) -> CallToolResult:
    """Return machine-readable fields together with readable text."""
    return CallToolResult(
        content=[TextContent(type="text", text=text or json.dumps(data, indent=2))],
        structured_content=data,
        is_error=error,
    )


def elapsed_samples(samples: list[Object]) -> list[Object]:
    now = time.time()
    return [
        {
            **{k: v for k, v in s.items() if k != "time"},
            "seconds_ago": round(max(0, now - s["time"]), 3),
        }
        for s in samples
    ]


def local(base: Path, path: str) -> str:
    """A path the call writes, against ``base`` when relative, without resolving its last part."""
    destination = base / Path(path).expanduser()
    return str(destination.parent.resolve() / destination.name)


def summary_text(data: Object) -> str:
    """One readable line: identity, shape, method, geometry, files, devices and time."""
    summary: list[str] = [str(data.get("handle") or "Saved"), str(data["kind"])]
    if data.get("shape"):
        summary.append(" x ".join(map(str, data["shape"])) + " " + str(data["unit"]))
    summary.extend(f"{key} {data[key]}" for key in ("method",) if key in data)
    if stopped := solve_summary(data):
        summary.append(stopped)
    summary.extend(
        f"{key} {data[key]}"
        for key in (
            "geometry",
            "calibration",
            "checkpoint",
            "slice_count",
            "first_file",
            "last_file",
        )
        if key in data
    )
    if auto := data.get("automatic"):
        summary.extend(auto.get("reasons", []))
    if beyond := beyond_detector(data):
        summary.append(beyond)
    if (poses := data.get("poses")) and poses["rms"] is not None:
        rms = ", ".join(
            f"{column['name']} {value:.4g} {column['unit']}"
            for column, value in zip(poses["columns"], poses["rms"], strict=True)
        )
        summary.append(f"pose RMS {rms}")
    if disturbed := data.get("disturbed_views"):
        summary.append(str(disturbed["message"]))
    summary.extend(data.get("paths", []))
    summary.extend(data.get("devices", []))
    summary.append(f"{data['wall_seconds']:g} s")
    summary.extend(data.get("notes", []))
    return " · ".join(summary)


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
            summary = record["summary"]
            entry = (
                {**summary, "manifest": record.get("manifest")}
                if args.detail == "full"
                else {k: summary.get(k) for k in ("kind", "shape", "unit")}
            )
            entry["bytes"] = record.get("bytes", state["handles"].get(record.get("worker")))
            handles[ref] = entry
        return result(
            {
                "versions": ready["versions"],
                "devices": ready["devices"],
                "allocators": elapsed_samples(state["allocators"]),
                "jobs": state["requests"],
                "handles": handles,
                "python_namespace_bytes": state.get("python_namespace_bytes", 0),
                "worker_loss": self.supervisor.report,
            }
        )

    async def inspect(self, args: Inspect) -> CallToolResult:
        """Inspect through the worker with directory discovery and unrecognized-HDF5 fallback."""
        path = (self.supervisor.base / Path(args.path).expanduser()).resolve()
        from tomojax.io.api import inspect_export

        if (
            path.is_dir()
            and not (path / "checkpoint.json").is_file()
            and not await asyncio.to_thread(inspect_export, path)
        ):
            return result(await asyncio.to_thread(inspect_directory, path))
        g = await self.supervisor.worker()

        async def report(item: Path) -> Object:
            output = await g.request({"type": "inspect", "path": str(item)})
            full = output["value"]["report"]
            if args.detail == "full":
                return full
            return {
                k: full[k]
                for k in (
                    "record",
                    "contents",
                    "input_path",
                    "checkpoint",
                    "valid",
                    "issues",
                    "loading",
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
        if saved := object_mapping(data.get("checkpoint")):
            return result(saved, text=f"Checkpoint of {saved['run']}. {saved['resume']}")
        shape = data.get("projection", {}).get("shape") or data.get("volume", {}).get("shape")
        issues = "; ".join(data.get("issues", []))
        text = f"{path.name}, shape {shape}. {issues or 'Inspection passed.'}"
        if loading := object_mapping(data.get("loading")):
            found = [f"{k} {v}" for k, v in loading.items() if v is not None and k != "message"]
            text += f" Loading: {loading['message'] or ', '.join(found) or 'as recorded'}."
        reply = result(data, text=text)
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
        dry = params.pop("dry_run", False)
        detail = params.pop("detail", "summary")
        command: Object = (
            params["operation"] if name in ("edit_scan", "preview") else {"type": name, **params}
        )
        model = MODELS[command["type"]]
        # Files and directories the call writes resolve against the project directory.
        written: dict[str, str] = {k: command[k] for k in ("path", "checkpoint") if command.get(k)}
        command.update({key: local(self.supervisor.base, path) for key, path in written.items()})
        roi = command.pop("roi", None) if model in (op.Slice, op.CenterSweep) else None
        if roi is not None:
            lo, hi = roi["zyx".index(command["axis"])]
            if not 0 <= lo <= command["index"] < hi:
                raise ValueError("slice index outside volume roi")
        tag_configs(command, model)
        command = wire_arguments(command, model, base=self.supervisor.base, ledger=self.ledger)
        # Validate nested expert keys here too, preserving their full field paths.
        msgspec.convert(command, type=model)
        if name == "preview":
            g = await self.supervisor.worker()
            if model is op.RegionPreview:

                def publish(output: Object) -> Object:
                    value = output["value"]
                    data = self.publish_handle(command, output)
                    return {
                        "summary": {**data, "grid": value["grid"], "bounds": value["bounds"]},
                        "value": value,
                    }

                published = await g.request(command, accept=publish)
                data, value = published["summary"], published["value"]
                reply = seeing.render(seeing.preview_panels(value))
                r = result({**data, **(reply.structured_content or {})})
                r.content.extend(reply.content[1:])
                return r
            async with seeing.temporary(g, command) as value:
                return await asyncio.to_thread(
                    lambda: seeing.render(seeing.preview_panels(value, roi))
                )
        if dry:
            return await self.dry_run(command, detail=detail)
        if model is op.Evaluate:
            g = await self.supervisor.worker()
            started = time.monotonic()
            evaluated = await g.request({"type": "run", "operation": command})
            data = evaluated["value"]
            if detail == "full":
                data.update(manifest=evaluated["manifest"], info=evaluated["info"])
            return result({**data, "wall_seconds": round(time.monotonic() - started, 3)})
        return await self.execute(command, detail=detail)

    async def dry_run(self, command: Object, *, detail: str) -> CallToolResult:
        """Use worker preflight and resolve a plan on the same generation."""
        g = await self.supervisor.worker()
        output = await g.request({"type": "validate", "operation": command})
        if command["type"] in ("align", "reconstruct"):
            output = await g.request({"type": "plan", "operation": command})
        data = output["value"]
        if data["type"] == "validated":
            data.update(operation=command["type"], **data.pop("details", {}))
            if len(data) == 2:
                data["parameters"] = command
        elif placement := data.get("placement"):
            placement["snapshot"] = elapsed_samples(placement["snapshot"])
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
                data.update(output["info"]["summary"], paths=value["paths"])
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
        reply = result(data, text=summary_text(data))
        for name in data.get("paths", []):
            path = Path(name)
            reply.content.append(
                ResourceLink(type="resource_link", uri=path.as_uri(), name=path.name)
            )
        if command["type"] == "align":
            fit = method_fit(command.get("method"), command.get("fit"))
            await seeing.default_panels(self, data, reply, fit=fit)
        elif command["type"] == "reconstruct":
            await seeing.default_panels(self, data, reply, fit=())
        return reply

    def publish_handle(
        self, command: Object, output: Object, *, reserved: tuple[str, dict] | None = None
    ) -> Object:
        """Issue the session identity while the accepting generation is still live."""
        value = output["value"]
        kind: str = value["type"]
        array = value.get(PRIMARY.get(kind, kind)) or value.get("poses") or {}
        info = object_mapping(output["info"]["summary"])
        summary: dict[str, object] = {
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
            "corrections",
            "diagnostics",
            "disturbed_views",
            "geometry",
            "automatic",
            "truncation",
            "placement",
            "name",
            "voxels",
            "bounds",
            "mean",
            "std",
            "recipe",
        ):
            if key in value or key in info:
                summary[key] = value.get(key, info.get(key))
        arrays = object_mapping(output["info"]["arrays"])
        if disturbance := object_mapping(info.get("disturbed_views")):
            summary["disturbed_views"] = {
                key: arrays.get(f"/disturbed_views/{key}", item)
                for key, item in disturbance.items()
            }
        # Every scalar the solver reports: its stopping status, chosen weights, noise levels.
        summary.update(
            (key, item)
            for key, item in info.items()
            if key not in summary and (item is None or isinstance(item, str | bool | int | float))
        )
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
        if reserved is None:
            ref = self.ledger.issue(alias, record)
        else:
            ref, records = reserved
            records[ref] = record
        data = {"handle": ref, **summary}
        if any(pointer.startswith("/corrections/") for pointer in arrays):
            data["info"] = output["info"]
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

    async def python(self, args: Python) -> CallToolResult:
        g = await self.supervisor.worker()
        aliases = {
            ref: resolve(
                ref,
                base=self.supervisor.base,
                ledger=self.ledger,
                field="array" if "artifact" in record else "source",
            )
            for ref, record in self.ledger.live.items()
        }
        command = {
            "type": "python",
            "code": args.code,
            "reset": args.reset,
            "traceback": args.traceback,
            "aliases": aliases,
        }

        def publish(output: Object) -> Object:
            value = output["value"]
            records = {}
            summaries = {}
            for ref, product in value["handles"].items():
                summaries[ref] = self.publish_handle(
                    command, {**output, "value": product}, reserved=(ref, records)
                )
            self.ledger.live.update(records)
            value["handles"] = summaries
            if g.prefix > 1:
                value["namespace_generation"] = g.prefix
                value["restart_notice"] = "Worker restart cleared the preceding namespace."
            return value

        async with seeing.temporary(g, command, publish_python=publish) as value:
            images = value.pop("images")
            reply = result(value)
            for ref in images:
                reply.content.append(await asyncio.to_thread(seeing.python_image, ref))
            return reply

    async def describe(self, args: Describe) -> CallToolResult:
        """Select operation and config keys from the running worker's real schema."""
        return result(await operation_parameters(self.supervisor, args.operation, args.method))

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


def tool_error(
    name: str, err: ValueError | OSError, spec: Object, arguments: Object | None = None
) -> CallToolResult:
    """Preserve worker categories, nested paths and enumerated parameter choices.

    A request that fails validation also gets the expected shape of the argument at fault.
    """
    kind = err.kind if isinstance(err, WorkerError) else "invalid_parameter"
    path = err.path if isinstance(err, WorkerError) else None
    checkpoint = err.checkpoint if isinstance(err, WorkerError) else None
    message = str(err)
    hint = None
    if isinstance(err, msgspec.ValidationError):
        match = re.search(r" - at `\$\.(.+?)`$", message) or re.search(
            r"missing required field `(\w+)`$", message
        )
        if match:
            path = match[1]
            field = re.split(r"[.\[]", path, maxsplit=1)[0]
            hint = expected(spec, field, (arguments or {}).get(field))
    choices = spec["properties"].get(path, {}).get("enum")
    hints = {
        "out_of_memory": "Try dry_run, slab reconstruction, binning or different devices.",
        "canceled": "Resubmit when ready.",
        "superseded": "Use the newer job.",
        "input": "Read the error, correct the Python code and retry."
        if name == "python"
        else "Inspect the input path and metadata.",
        "internal": "Check status for worker and device information.",
    }
    next_step = hints.get(kind, f"Check the {name} tool parameters and retry.")
    if checkpoint is not None:
        next_step = (
            f"Its state is saved in {checkpoint}: call {name} again with the same arguments, "
            "on the same input, to resume (inspect shows what the checkpoint holds)."
        )
    if choices:
        next_step = "Choose one of " + ", ".join(map(str, choices)) + "."
    elif hint:
        next_step = f"Give {hint}."
    data = {"kind": kind, "path": path, "message": message, "next": next_step}
    if checkpoint is not None:
        data["checkpoint"] = checkpoint
    return result(data, text=f"{kind}: {message}. {next_step}", error=True)


def tag_configs(command: Object, model: type[Record]) -> None:
    """Select each expert config's wire type from its enclosing method."""
    settings = (
        [command] if model is not op.Evaluate else [*command["candidates"], command["reference"]]
    )
    for setting in settings:
        config = setting.get("config")
        if config is not None and "type" not in config:
            if setting["type"] == "align":
                config["type"] = "AlignConfig"
            else:
                configs = {
                    cls.__name__.removesuffix("Config").lower(): cls.__name__
                    for cls in get_args(op.Config.__value__)
                }
                method = setting["method"]
                if method not in configs:
                    raise ValueError(
                        "Choose a specific method or give config its type from describe."
                    )
                config["type"] = configs[method]


def workflow_output() -> Object:
    """Describe handle, file, preflight, error and continuation responses."""
    return {
        "type": "object",
        "properties": {
            "handle": {"type": "string"},
            "kind": {"type": "string"},
            "members": {"type": "object"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "type": {"type": "string"},
            "wall_seconds": {"type": "number"},
            "method": {"type": "string"},
            "manifest": {"type": "object"},
            "info": {"type": "object"},
            "continuation": {"type": "string"},
            "checkpoint": {"type": "string"},
        },
        "anyOf": [
            {"required": [k]} for k in ("handle", "paths", "type", "message", "continuation")
        ],
    }


ANY = {"type": "object"}
TEXT = {"type": "string"}
LIST = {"type": "array"}
PLANES = {
    "panels": {"type": "array", "items": ANY},
    "images": {"type": "array", "items": ANY},
    "image_limits": ANY,
    "intensity_limits": LIST,
    "marks": {"type": "array", "items": ANY},
    "features": {"type": "array", "items": ANY},
}
# Concise result schemas: the fields a client can rely on, none of them required, so a
# result clipped to its paging link or an error still conforms.
OUTPUTS: dict[str, Object] = {
    "status": {
        "versions": ANY,
        "devices": LIST,
        "allocators": LIST,
        "jobs": ANY,
        "handles": ANY,
        "python_namespace_bytes": {"type": "integer"},
    },
    "inspect": {
        "valid": {"type": "boolean"},
        "issues": LIST,
        "projection": ANY,
        "volume": ANY,
        "geometry": ANY,
        "directory": TEXT,
        "datasets": ANY,
        "loading": ANY,
        "checkpoints": LIST,
        "run": ANY,
        "request": {},
        "entries": ANY,
        "resume": TEXT,
    },
    "describe": {"operation": TEXT, "parameters": ANY},
    "control": {"action": TEXT, "tools": LIST},
    "python": {"stdout": TEXT, "stderr": TEXT, "value": {}, "handles": ANY},
    "view": {
        **PLANES,
        "count": {"type": "integer"},
        "mean": {"type": ["number", "null"]},
        "histogram": LIST,
        "profile": LIST,
        "columns": LIST,
        "table": LIST,
    },
    "preview": {**PLANES, "handle": TEXT, "bounds": LIST},
}


def output_schema(name: str) -> Object:
    """The declared structured result of one tool."""
    if name in {*MODELS, "edit_scan", "open"}:
        return workflow_output()
    common = {
        "continuation": TEXT,
        "message": TEXT,
        "next": TEXT,
        "notes": LIST,
        "checkpoint": TEXT,
    }
    return {"type": "object", "properties": {**OUTPUTS.get(name, {}), **common}}


def binding(
    cls: type[Record],
    handler: Callable[..., Awaitable[CallToolResult]],
    definitions: Object,
    *,
    name: str | None = None,
    parts: Parts | None = None,
) -> Object:
    """Construct a plain tool record with msgspec validation and actionable errors."""
    name = name or cls.__name__.lower()
    spec = schema(cls, definitions)

    async def invoke(arguments: dict[str, Any]) -> CallToolResult:
        try:
            reply = await handler(msgspec.convert(arguments, type=cls))
        except (ValueError, OSError) as err:
            reply = tool_error(name, err, spec, arguments)
        job = activity.get()
        if job and job.notes and reply.structured_content is not None:
            reply.structured_content.setdefault("notes", job.notes)
        return bounded(reply, parts) if parts is not None else reply

    return {
        "name": name,
        "description": " ".join((cls.__doc__ or "").split())
        + (
            " Handles last until close, worker restart or server exit."
            if name
            in {
                "correct",
                "align",
                "simulate",
                "reconstruct",
                "edit_scan",
                "preview",
                "project",
                "backproject",
            }
            else ""
        ),
        "long": name in {*MODELS, "open", "edit_scan", "preview", "python"},
        "read_only": name in {"status", "inspect", "describe", "view", "preview"},
        # Writes or replaces files, discards handles, or runs arbitrary code.
        "destructive": name in {"save", "export", "reconstruct_slabs", "control", "python"},
        "schema": spec,
        "output_schema": output_schema(name),
        "handler": invoke,
    }


def registry(core: Core) -> list[Object]:
    """Build fresh tool descriptions, validators and handlers for this checkout."""
    tools = Tools(core)
    definitions = msgspec.json.decode(
        resources.files("tomojax.cli").joinpath("parameters.json").read_bytes()
    )["$defs"]
    return (
        [
            binding(cls, getattr(tools, cls.__name__.lower()), definitions, parts=core.parts)
            for cls in (Status, Inspect, Open, Describe, Control, Python)
        ]
        + [binding(seeing.View, partial(seeing.view, tools), definitions, parts=core.parts)]
        + [
            binding(cls, partial(tools.workflow, name), definitions, name=name, parts=core.parts)
            for name, cls in forms().items()
        ]
    )
