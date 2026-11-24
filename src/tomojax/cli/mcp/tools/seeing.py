"""Model-facing array inspection, registered previews and transient worker artifacts."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import ImageContent, TextContent
import numpy as np

from tomojax.cli import operations as op
from tomojax.cli._images import Panel, limits, montage, pose_plot
from tomojax.cli.mcp.tools.references import resolve

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from mcp_types import CallToolResult

    from tomojax.cli.mcp.supervisor import Generation, Object
    from tomojax.cli.mcp.tools import Tools


class View(op.Window, kw_only=True):
    """Inspect a member, saved file, .npy or file.h5::/dataset.

    ROI is half-open in storage order.
    Planes defaults to the central planes. Set axis and index to pick one. Profile reads a
    one-dimensional ROI. Difference subtracts other from source on matching volume grids.
    Pose columns are alpha, beta, phi (degrees), then x, z, y (geometry length).
    """

    source: str
    mode: Literal[
        "planes",
        "projection",
        "sinogram",
        "difference",
        "statistics",
        "profile",
        "pose_table",
        "pose_plot",
    ] = "planes"
    axis: str | None = None
    index: int | None = None
    other: str | None = None
    percentiles: tuple[float, ...] = (1, 5, 50, 95, 99)
    bins: int = 32


def image(data: bytes) -> ImageContent:
    """Wrap a PNG for a tool response."""
    return ImageContent(type="image", mime_type="image/png", data=base64.b64encode(data).decode())


def artifacts(value: Any) -> Iterator[str]:
    """Collect temporary files, including nonuniform coordinates."""
    if isinstance(value, dict):
        if value.get("type") == "npy" and value.get("artifact"):
            yield value["artifact"]
        else:
            for child in value.values():
                yield from artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifacts(child)


@asynccontextmanager
async def temporary(g: Generation, command: Object) -> AsyncIterator[Object]:
    """Hold worker-owned pixels and coordinate vectors until rendering finishes."""
    output = await g.request(command)
    value = output["value"]
    try:
        yield value
    finally:
        for token in set(artifacts(value)):
            await g.request({"type": "release", "artifact": token})


def coordinates(axis: Object, length: int) -> np.ndarray:
    """Resolve physical positions from returned axis metadata."""
    c = axis["coordinates"]
    if c["type"] == "uniform":
        return c["origin"] + np.arange(length) * c["spacing"]
    return np.load(c["values"]["path"], mmap_mode="r", allow_pickle=False)


def plane(
    array: Object,
    pixels: np.ndarray,
    title: str,
    *,
    dim: int | None = None,
    index: int = 0,
    source_index: int | None = None,
) -> Panel:
    """Label the sampled plane in its own registration."""
    axes = array["axes"]
    labels = [
        f"{a['name']}[{a['index']}] = {a['coordinate']:.5g} {a['unit']}" for a in array["fixed"]
    ]
    if dim is not None:
        a = axes[dim]
        at = coordinates(a, array["shape"][dim])[index]
        labels.append(
            f"{a['name']}[{index if source_index is None else source_index}] = {at:.5g} {a['unit']}"
        )
        axes = axes[:dim] + axes[dim + 1 :]
        pixels = np.take(pixels, index, axis=dim)
    for a, n in zip(axes, pixels.shape, strict=True):
        c = coordinates(a, n)
        labels.append(f"{a['name']} [{c[0]:.5g}, {c[-1]:.5g}] {a['unit']}")
    aspect = 1.0
    if axes[0]["unit"] == axes[1]["unit"] and all(
        a["coordinates"]["type"] == "uniform" for a in axes
    ):
        aspect = abs(axes[0]["coordinates"]["spacing"] / axes[1]["coordinates"]["spacing"])
    return Panel(title, pixels, tuple(labels), array["unit"], aspect, array["domain"])


def render(panels: list[Panel]) -> CallToolResult:
    """Share intensity limits within each measured quantity and unit."""
    from tomojax.cli.mcp.tools import result

    groups = {(p.domain, p.unit) for p in panels}
    ranges = {key: limits([p for p in panels if (p.domain, p.unit) == key]) for key in groups}
    scales = [ranges[p.domain, p.unit] for p in panels]
    reply = result(
        {
            **({"intensity_limits": scales[0]} if len(groups) == 1 else {}),
            "panels": [
                {"array": p.title, "labels": p.labels, "unit": p.unit, "intensity_limits": scale}
                for p, scale in zip(panels, scales, strict=True)
            ],
        }
    )
    for i in range(0, len(panels), 3):
        reply.content.append(image(montage(panels[i : i + 3], scales[i : i + 3])))
    return reply


def load(array: Object) -> np.ndarray:
    """Map a worker-published array while its artifact is held."""
    return np.load(array["storage"]["path"], mmap_mode="r", allow_pickle=False)


def preview_panels(value: Object) -> list[Panel]:
    """Select registered planes from an interactive worker product."""
    kind = value["type"]
    if kind == "region":
        return [plane(a, load(a), "Estimated region") for a in value["arrays"].values()]
    if kind == "sweep":
        a = value["planes"]
        pixels = load(a)
        return [plane(a, pixels, "Centre sweep", dim=0, index=i) for i in range(len(pixels))]
    if kind == "correction_preview":
        views = load(value["views"])
        middle = len(views) // 2
        return [
            plane(a, load(a), title, dim=0, index=i)
            for a, title, i in (
                (value["before"], "Before correction / native grid", int(views[middle])),
                (value["after"], "After correction / corrected grid", middle),
            )
        ]
    a = value["plane"]
    return [plane(a, load(a), "Reconstructed slice")]


def reference(tools: Tools, name: str, mode: str) -> Object:
    """Resolve the source by storage type and requested member."""
    if name.startswith("@") and "." not in name and mode.startswith("pose_"):
        name += ".poses"
    src = resolve(name, base=tools.supervisor.base, ledger=tools.ledger)
    assert isinstance(src, dict)
    if (
        src["type"] == "npy"
        and not src.get("artifact")
        and Path(src["path"]).suffix.lower() != ".npy"
    ):
        src["type"] = "path"
    return src


async def view(tools: Tools, args: View) -> CallToolResult:
    """Return a registered image or numerical summary of the requested source."""
    from tomojax.cli.mcp.tools import result

    g = await tools.supervisor.worker()
    src = reference(tools, args.source, args.mode)
    command = {"type": "read", "source": src, "roi": args.roi, "stride": args.stride}
    if args.mode == "statistics":
        value = await g.request(
            {**command, "type": "statistics", "bins": args.bins, "percentiles": args.percentiles}
        )
        return result(value["value"])
    if args.mode in ("profile", "pose_table", "pose_plot"):
        return await series(g, command, args.mode)
    meta = (await g.request({"type": "read", "source": src, "metadata_only": True}))["value"][
        "array"
    ]
    axes = [a["name"] for a in meta["axes"]]
    if args.mode in ("projection", "sinogram") and meta["domain"] == "attenuation":
        raise ValueError(
            f"{args.source} is a volume. Select .projections on its scan or simulation, "
            "or provide a detector dataset."
        )
    other = None
    if args.mode == "difference":
        if not args.other:
            raise ValueError("difference requires other")
        source_b = reference(tools, args.other, args.mode)
        second = (await g.request({"type": "read", "source": source_b, "metadata_only": True}))[
            "value"
        ]["array"]

        def grid(m: Object) -> Object:
            return {
                "unit": m["unit"],
                "axes": {a["name"]: (n, a) for n, a in zip(m["shape"], m["axes"], strict=True)},
            }

        if (
            set(axes) != set("zyx")
            or any(a["frame"] != "volume" for a in meta["axes"])
            or grid(meta) != grid(second)
        ):
            raise ValueError(
                f"Difference requires equal volume grids. {args.source}: {grid(meta)}. "
                f"{args.other}: {grid(second)}"
            )
        other = (source_b, [axes.index(a["name"]) for a in second["axes"]])
    if len(axes) == 2 and args.mode == "planes":
        async with temporary(g, command) as v:
            a = v["array"]
            return render([plane(a, load(a), args.source)])
    if len(axes) != 3:
        raise ValueError(f"{args.mode} needs a three-dimensional array, got {axes}")
    return await plane_views(g, command, args, meta, other)


async def series(g: Generation, command: Object, mode: str) -> CallToolResult:
    """Return a registered line profile or pose traces."""
    from tomojax.cli.mcp.tools import result

    async with temporary(g, command) as v:
        a = v["array"]
        values = load(a)
        if mode == "profile":
            dims = [i for i, n in enumerate(values.shape) if n > 1]
            if len(dims) != 1:
                raise ValueError("profile needs a line ROI with one varying axis")
            ax = a["axes"][dims[0]]
            return result(
                {
                    "profile": values.ravel().tolist(),
                    "unit": a["unit"],
                    "axis": ax["name"],
                    "coordinates": coordinates(ax, len(values.ravel())).tolist(),
                    "axis_unit": ax["unit"],
                }
            )
        if a["domain"] != "poses":
            raise ValueError("choose a registered .poses member with all six columns")
        ax = a["axes"][0]
        views = coordinates(ax, len(values))
        data = {"columns": a["columns"], "views": len(views), "view_unit": ax["unit"]}
        if mode == "pose_table":
            return result({**data, "coordinates": views.tolist(), "table": values.tolist()})
        reply = result(data)
        reply.content.append(image(pose_plot(values, a["columns"], views, ax["unit"])))
        return reply


async def plane_views(
    g: Generation, command: Object, args: View, meta: Object, other: tuple[Object, list[int]] | None
) -> CallToolResult:
    """Choose display planes and read only their sampled hyperslabs."""
    axes = [a["name"] for a in meta["axes"]]
    if args.axis is not None:
        axis_names = [args.axis]
    elif args.mode == "sinogram":
        axis_names = ["v" if "v" in axes else axes[1]]
    elif args.mode == "projection" or "view" in axes:
        axis_names = ["view" if "view" in axes else axes[0]]
    else:
        axis_names = axes if args.index is None else [axes[0]]
    roi = args.roi if args.roi is not None else tuple((0, n) for n in meta["shape"])
    strides = args.stride if args.stride is not None else (1,) * 3
    if len(roi) != 3 or len(strides) != 3 or min(strides) < 1:
        raise ValueError("roi and stride need three dimensions and positive strides")
    panels = []
    for axis in axis_names:
        if axis not in axes:
            raise ValueError(f"axis must be one of {axes}")
        d = axes.index(axis)
        selected = list(roi)
        lo, hi = selected[d]
        if not 0 <= lo < hi <= meta["shape"][d]:
            raise ValueError(f"roi outside {meta['shape']}")
        index = (lo + hi) // 2 if args.index is None else args.index
        if not lo <= index < hi:
            raise ValueError(f"index {index} outside {axis} interval [{lo}, {hi})")
        selected[d] = (index, index + 1)
        stride = [
            max(step, ceil((b - a) / 512)) for step, (a, b) in zip(strides, selected, strict=True)
        ]
        read = {**command, "roi": selected, "stride": stride}
        async with temporary(g, read) as v:
            a = v["array"]
            pixels = np.asarray(load(a))
            title = (
                "Alignment working image " if args.source.endswith(".image") else ""
            ) + args.source
            if other is not None:
                source_b, order = other
                read_b = {
                    **read,
                    "source": source_b,
                    "roi": [selected[i] for i in order],
                    "stride": [stride[i] for i in order],
                }
                async with temporary(g, read_b) as w:
                    pixels = pixels.astype(float) - load(w["array"]).transpose(np.argsort(order))
                    title += " minus " + str(args.other)
            panels.append(plane(a, pixels, title, dim=d, source_index=index))
    return await asyncio.to_thread(render, panels)


async def default_panels(tools: Tools, data: Object, reply: CallToolResult) -> None:
    """Render each published image or pose member of a completed fit."""
    from tomojax.cli.mcp.supervisor import WorkerLost

    for field in ("poses", "image", "volume"):
        if field not in data["members"]:
            continue
        try:
            visual = await view(
                tools,
                View(
                    source=data["members"][field],
                    mode="pose_plot" if field == "poses" else "planes",
                ),
            )
            reply.content.extend(
                block for block in visual.content if isinstance(block, ImageContent)
            )
        except (OSError, ValueError, WorkerLost) as err:
            reply.content.append(TextContent(type="text", text=f"Panel unavailable: {err}"))
