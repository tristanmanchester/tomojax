"""Model-facing array inspection, registered previews and transient worker artifacts."""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import ImageContent, TextContent
import numpy as np
from PIL import Image

from tomojax.cli import _marks, operations as op
from tomojax.cli._planes import plane_windows
from tomojax.cli.images import Panel, label_number, limits, montage, pose_plot, thumbnail
from tomojax.cli.mcp.tools.references import resolve

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from mcp_types import CallToolResult

    from tomojax.cli.mcp.supervisor import Generation, Object
    from tomojax.cli.mcp.tools import Tools


class View(op.Window, kw_only=True):
    """View volume slices, projections, sinograms, pose plots or differences on matching grids.

    Use handle members, session arrays, .npy files, file.h5::/dataset or saved TomoJAX volumes.
    Open acquisitions and select .projections, or use kind=frames and .counts for raw counts.
    Select a plane with axis and index. Specify the half-open ROI in array order.
    Point at something with marks in array indices of the source (z, y, x or view, v, u);
    the result gives each mark's pixel in the image and the value under it. Outline
    feature handles on any source with the same grid. For sweeps or custom measures, use
    python with the same handles.
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
    marks: tuple[_marks.AnyMark, ...] = ()
    features: tuple[str, ...] = ()
    ticks: bool | None = None


def image(data: bytes) -> ImageContent:
    """Wrap a PNG for a tool response."""
    return ImageContent(type="image", mime_type="image/png", data=base64.b64encode(data).decode())


def python_image(ref: Object) -> ImageContent:
    """Render a snapshot staged by show() with its own intensity limits."""
    pixels = np.load(ref["path"], mmap_mode="r", allow_pickle=False)
    return image(thumbnail(Image.fromarray(pixels)))


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
async def temporary(
    g: Generation, command: Object, *, publish_python: Callable[[Object], Object] | None = None
) -> AsyncIterator[Object]:
    """Hold worker-owned pixels and coordinate vectors until rendering finishes."""
    finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def accept(output: Object) -> Object:
        value = output["value"]
        tokens = set(artifacts(value["images"] if publish_python else value))

        def release(_finished: asyncio.Future[None]) -> None:
            if not g.exit.is_set():
                for token in tokens:
                    _, ack = g.submit({"type": "release", "artifact": token})
                    ack.result.add_done_callback(lambda future: future.exception())

        finished.add_done_callback(release)
        return publish_python(output) if publish_python else value

    try:
        yield await g.request(command, accept=accept)
    finally:
        finished.set_result(None)


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
    sampling: tuple[str, ...] = (),
    roi: tuple[tuple[int, int], ...] | None = None,
) -> Panel:
    """Label the sampled plane in its own registration."""
    axes = array["axes"]
    labels = [
        f"{a['name']}[{a['index']}] = {label_number(a['coordinate'])} {a['unit']}"
        for a in array["fixed"]
    ]
    if dim is not None:
        a = axes[dim]
        at = coordinates(a, array["shape"][dim])[index]
        position = index if source_index is None else source_index
        labels.append(f"{a['name']}[{position}] = {label_number(at)} {a['unit']}")
        axes = axes[:dim] + axes[dim + 1 :]
        pixels = np.take(pixels, index, axis=dim)
    windows = []
    for a, n in zip(axes, pixels.shape, strict=True):
        c = coordinates(a, n)
        lo, hi = (0, n) if roi is None else roi["zyx".index(a["name"])]
        if not 0 <= lo < hi <= n:
            raise ValueError(f"roi outside {a['name']} extent [0, {n})")
        windows.append(slice(lo, hi))
        labels.append(f"{a['name']} [{label_number(c[lo])}, {label_number(c[hi - 1])}] {a['unit']}")
    pixels = pixels[tuple(windows)]
    aspect = 1.0
    if axes[0]["unit"] == axes[1]["unit"] and all(
        a["coordinates"]["type"] == "uniform" for a in axes
    ):
        aspect = abs(axes[0]["coordinates"]["spacing"] / axes[1]["coordinates"]["spacing"])
    return Panel(title, pixels, (*labels, *sampling), array["unit"], aspect, array["domain"])


def render(
    panels: list[Panel],
    overlays: list[_marks.Overlay] | None = None,
    marks: tuple[_marks.AnyMark, ...] = (),
) -> CallToolResult:
    """Share intensity limits within each measured quantity and unit.

    With overlays, each plane gets its marks, feature outlines and ticks, and the
    result reports where each mark was drawn and the value under it.
    """
    from tomojax.cli.mcp.tools import result

    groups = {(p.domain, p.unit) for p in panels}
    ranges = {key: limits([p for p in panels if (p.domain, p.unit) == key]) for key in groups}
    scales = [ranges[p.domain, p.unit] for p in panels]
    placed: list[list[Object]] = [[] for _ in marks]

    def decorate(
        picture: Image.Image, j: int, rect: tuple[float, float, float, float], first: int
    ) -> None:
        assert overlays is not None
        cell = _marks.Cell(first // 3, *rect)
        found = _marks.draw(picture, overlays[first + j], cell, panels[first + j].pixels)
        for m, report in enumerate(found):
            placed[m].append({"panel": first + j, **report})

    images = [
        image(
            montage(
                panels[i : i + 3],
                scales[i : i + 3],
                None if overlays is None else partial(decorate, first=i),
            )
        )
        for i in range(0, len(panels), 3)
    ]
    reply = result(
        {
            **({"intensity_limits": scales[0]} if len(groups) == 1 else {}),
            "panels": [
                {"array": p.title, "labels": p.labels, "unit": p.unit, "intensity_limits": scale}
                for p, scale in zip(panels, scales, strict=True)
            ],
            **(
                {
                    "marks": [
                        {
                            "type": _marks.name(mark),
                            **({"label": mark.label} if mark.label else {}),
                            "panels": found,
                        }
                        for mark, found in zip(marks, placed, strict=True)
                    ]
                }
                if marks
                else {}
            ),
        }
    )
    reply.content.extend(images)
    return reply


def load(array: Object) -> np.ndarray:
    """Map a worker-published array while its artifact is held."""
    return np.load(array["storage"]["path"], mmap_mode="r", allow_pickle=False)


def preview_panels(value: Object, roi: tuple[tuple[int, int], ...] | None = None) -> list[Panel]:
    """Select registered planes from an interactive worker product."""
    kind = value["type"]
    if kind == "region":
        return [plane(a, load(a), "Estimated region") for a in value["arrays"].values()]
    if kind == "sweep":
        a = value["planes"]
        pixels = load(a)
        return [
            plane(a, pixels, "Centre sweep", dim=0, index=i, roi=roi) for i in range(len(pixels))
        ]
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
    return [plane(a, load(a), "Reconstructed slice", roi=roi)]


def reference(tools: Tools, name: str, mode: str) -> Object:
    """Resolve the source by storage type and requested member."""
    if name.startswith("@") and "." not in name and mode.startswith("pose_"):
        name += ".poses"
    src = resolve(name, base=tools.supervisor.base, ledger=tools.ledger)
    assert isinstance(src, dict)
    return src


FEATURE_COLOURS: tuple[_marks.Colour, ...] = ("cyan", "magenta", "green", "orange", "red")


def flat(item: Object) -> np.ndarray:
    """The two-dimensional samples of a returned display plane."""
    pixels = load(item["array"])
    return pixels if item["dimension"] is None else np.take(pixels, 0, axis=item["dimension"])


async def metadata(g: Generation, source: Object) -> Object:
    """The registration of a whole source array, without reading pixels."""
    return (await g.request({"type": "read", "source": source, "metadata_only": True}))["value"][
        "array"
    ]


async def outlines(
    tools: Tools,
    g: Generation,
    stack: AsyncExitStack,
    reads: list[tuple[Object, int | None]],
    shape: list[int],
    names: tuple[str, ...],
) -> list[list[tuple[np.ndarray, _marks.Colour]]]:
    """Read each feature's mask on the same planes as the source, in palette order.

    ``reads`` holds the source's requests: a read with a mode returns its planes, a
    plain read one plane with the given fixed dimension.
    """
    if len(names) > len(FEATURE_COLOURS):
        raise ValueError(f"outline at most {len(FEATURE_COLOURS)} features in one view")
    found: list[list[tuple[np.ndarray, _marks.Colour]]] = []
    for colour, name in zip(FEATURE_COLOURS, names, strict=False):
        if tools.ledger.lookup(name.partition(".")[0])["kind"] != "feature":
            raise ValueError(f"{name} is not a feature handle")
        ref = resolve(name, base=tools.supervisor.base, ledger=tools.ledger)
        assert isinstance(ref, dict)
        grid = (await metadata(g, ref))["shape"]
        if grid != shape:
            raise ValueError(f"{name} covers a {grid} grid; the source is {shape}")
        planes = []
        for read, dim in reads:
            value = await stack.enter_async_context(temporary(g, {**read, "source": ref}))
            planes.extend(
                value["planes"]
                if read.get("mode")
                else [{"array": value["array"], "dimension": dim}]
            )
        found.extend([] for _ in range(len(planes) - len(found)))
        for k, item in enumerate(planes):
            found[k].append((np.asarray(flat(item), dtype=bool), colour))
    return found


async def view(tools: Tools, args: View) -> CallToolResult:
    """Return a registered image or numerical summary of the requested source."""
    from tomojax.cli.mcp.tools import result

    g = await tools.supervisor.worker()
    src = reference(tools, args.source, args.mode)
    command = {"type": "read", "source": src, "roi": args.roi, "stride": args.stride}
    decorated = bool(args.marks or args.features or args.ticks)
    if decorated and args.mode not in ("planes", "projection", "sinogram", "difference"):
        raise ValueError("marks, features and ticks need a rendered plane mode")
    if args.mode == "statistics":
        value = await g.request(
            {**command, "type": "statistics", "bins": args.bins, "percentiles": args.percentiles}
        )
        return result(value["value"])
    if args.mode in ("profile", "pose_table", "pose_plot"):
        return await series(g, command, args.mode)
    if args.mode != "difference":
        request = {**command, "mode": args.mode, "axis": args.axis, "index": args.index}
        async with AsyncExitStack() as stack:
            v = await stack.enter_async_context(temporary(g, request))
            title = (
                "Alignment working image " if args.source.endswith(".image") else ""
            ) + args.source
            panels = [
                plane(
                    item["array"],
                    load(item["array"]),
                    title,
                    dim=item["dimension"],
                    source_index=item["index"],
                    sampling=item["sampling"],
                )
                for item in v["planes"]
            ]
            if not decorated:
                return await asyncio.to_thread(render, panels)
            frames = [
                _marks.frame_of(
                    [a["name"] for a in item["array"]["axes"]], item["window"], item["dimension"]
                )
                for item in v["planes"]
            ]
            masks: list[list[tuple[np.ndarray, _marks.Colour]]] = [[] for _ in frames]
            if args.features:
                shape = (await metadata(g, src))["shape"]
                masks = await outlines(tools, g, stack, [(request, None)], shape, args.features)
            return await overlaid(args, panels, frames, masks)
    meta = await metadata(g, src)
    axes = [a["name"] for a in meta["axes"]]
    if not args.other:
        raise ValueError("difference requires other")
    source_b = reference(tools, args.other, args.mode)
    second = await metadata(g, source_b)

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
    return await plane_views(tools, g, command, args, meta, other)


async def overlaid(
    args: View,
    panels: list[Panel],
    frames: list[_marks.Frame],
    masks: list[list[tuple[np.ndarray, _marks.Colour]]],
) -> CallToolResult:
    """Render planes with the requested marks, feature outlines and ticks."""
    _marks.check(args.marks, frames[0].axes)
    ticks = args.ticks if args.ticks is not None else True
    overlays = [
        _marks.Overlay(frame, args.marks, tuple(found), ticks)
        for frame, found in zip(frames, masks, strict=True)
    ]
    reply = await asyncio.to_thread(render, panels, overlays, args.marks)
    if not args.features:
        return reply
    from tomojax.cli.mcp.tools import result

    legend = [
        {"handle": name, "colour": colour}
        for name, colour in zip(args.features, FEATURE_COLOURS, strict=False)
    ]
    labelled = result({**(reply.structured_content or {}), "features": legend})
    labelled.content.extend(reply.content[1:])
    return labelled


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
    tools: Tools,
    g: Generation,
    command: Object,
    args: View,
    meta: Object,
    other: tuple[Object, list[int]],
) -> CallToolResult:
    """Choose display planes and read only their sampled hyperslabs."""
    axes = [a["name"] for a in meta["axes"]]
    roi = args.roi if args.roi is not None else tuple((0, n) for n in meta["shape"])
    strides = args.stride if args.stride is not None else (1,) * len(axes)
    if len(roi) != 3 or len(strides) != 3 or min(strides) < 1:
        raise ValueError("roi and stride need three dimensions and positive strides")
    panels = []
    frames = []
    reads: list[tuple[Object, int | None]] = []
    async with AsyncExitStack() as stack:
        for d, index, selected, stride, sampling in plane_windows(
            axes, meta["shape"], args.mode, args.axis, args.index, roi, strides
        ):
            read = {**command, "roi": selected, "stride": stride}
            reads.append((read, d))
            window = [(lo, hi, step) for (lo, hi), step in zip(selected, stride, strict=True)]
            frames.append(_marks.frame_of(axes, window, d))
            a = (await stack.enter_async_context(temporary(g, read)))["array"]
            pixels = np.asarray(load(a))
            title = (
                "Alignment working image " if args.source.endswith(".image") else ""
            ) + args.source
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
            panels.append(plane(a, pixels, title, dim=d, source_index=index, sampling=sampling))
        if not (args.marks or args.features or args.ticks):
            return await asyncio.to_thread(render, panels)
        masks: list[list[tuple[np.ndarray, _marks.Colour]]] = [[] for _ in frames]
        if args.features:
            masks = await outlines(tools, g, stack, reads, meta["shape"], args.features)
        return await overlaid(args, panels, frames, masks)


async def default_panels(
    tools: Tools, data: Object, reply: CallToolResult, *, fit: str | tuple[str, ...]
) -> None:
    """Render each published image or pose member of a completed fit."""
    from tomojax.cli.mcp.supervisor import WorkerLost

    blocks = (fit,) if isinstance(fit, str) else fit
    poses = bool({"translations", "rotations"}.intersection(blocks))
    for field in ("poses", "image", "volume"):
        if field not in data["members"] or (field == "poses" and not poses):
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
