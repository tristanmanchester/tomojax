"""Model-facing array inspection, registered previews and transient worker artifacts."""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import ImageContent, TextContent
import numpy as np
from PIL import Image

from tomojax.cli import _marks, operations as op
from tomojax.cli._planes import plane_windows
from tomojax.cli.images import (
    MAX_EDGE,
    MAX_PIXELS,
    Panel,
    compose,
    label_number,
    limits,
    montage,
    png,
    pose_plot,
    thumbnail,
)
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
    Give several sources to compare them side by side on the same planes and limits.
    Point with marks at array indices of the source (z, y, x or view, v, u), or at
    {"pixel": [x, y], "image": render} in an image a view returned (origin top-left);
    each mark's report gives its indices, pixel, value, and the value's percentile in the
    plane. A point or arrow on flat ground, with no peak, trough or edge within 2 samples,
    also gets a warning and the nearest structure's indices. A point with snap locks onto
    the brightest or darkest sample or strongest edge within reach. Images stay within
    1568 px and 1.15 MP, so their pixels are the pixels you see. Outline feature handles
    on any source on the same grid. For sweeps or custom measures, use python with the
    same handles.
    """

    source: str | tuple[str, ...]
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
    grid: bool = False
    limits: tuple[float, float] | None = None


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


def render(panels: list[Panel]) -> CallToolResult:
    """Plain planes sharing intensity limits within each measured quantity and unit."""
    from tomojax.cli.mcp.tools import result

    scales = shared_limits(panels)
    reply = result(
        {
            **({"intensity_limits": scales[0]} if len(set(scales)) == 1 else {}),
            "panels": [
                {"array": p.title, "labels": p.labels, "unit": p.unit, "intensity_limits": scale}
                for p, scale in zip(panels, scales, strict=True)
            ],
        }
    )
    reply.content.extend(
        image(montage(panels[i : i + 3], scales[i : i + 3])) for i in range(0, len(panels), 3)
    )
    return reply


def shared_limits(
    panels: list[Panel], given: tuple[float, float] | None = None
) -> list[tuple[float, float]]:
    """The given limits for every plane, or one range per measured quantity and unit."""
    if given is not None:
        lo, hi = given
        if not hi > lo:
            raise ValueError("limits must be (low, high) with low < high")
        return [given] * len(panels)
    groups = {(p.domain, p.unit) for p in panels}
    ranges = {key: limits([p for p in panels if (p.domain, p.unit) == key]) for key in groups}
    return [ranges[p.domain, p.unit] for p in panels]


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
            plane(a, pixels, "Center sweep", dim=0, index=i, roi=roi) for i in range(len(pixels))
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


FEATURE_COLORS: tuple[_marks.Color, ...] = ("cyan", "magenta", "green", "orange", "red")


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
) -> list[list[tuple[np.ndarray, _marks.Color]]]:
    """Read each feature's mask on the same planes as the source, in palette order.

    ``reads`` holds the source's requests: a read with a mode returns its planes, a
    plain read one plane with the given fixed dimension.
    """
    if len(names) > len(FEATURE_COLORS):
        raise ValueError(f"outline at most {len(FEATURE_COLORS)} features in one view")
    found: list[list[tuple[np.ndarray, _marks.Color]]] = []
    for color, name in zip(FEATURE_COLORS, names, strict=False):
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
            found[k].append((np.asarray(flat(item), dtype=bool), color))
    return found


PLANE_MODES = ("planes", "projection", "sinogram", "difference")
RENDERS = 256  # placements kept per connection for pixel marks


async def view(tools: Tools, args: View) -> CallToolResult:
    """Return a registered image or numerical summary of the requested source."""
    from tomojax.cli.mcp.tools import result

    g = await tools.supervisor.worker()
    names: tuple[str, ...] = (args.source,) if isinstance(args.source, str) else args.source
    if not names:
        raise ValueError("give a source")
    primary = names[0]
    if args.mode not in PLANE_MODES:
        if len(names) > 1 or args.marks or args.features or args.ticks or args.grid or args.limits:
            raise ValueError(
                "several sources, marks, features, ticks, grid and limits need a rendered "
                "plane mode: planes, projection, sinogram or difference"
            )
        src = reference(tools, primary, args.mode)
        command = {"type": "read", "source": src, "roi": args.roi, "stride": args.stride}
        if args.mode == "statistics":
            value = await g.request(
                {**command, "type": "statistics", "bins": args.bins}
                | {"percentiles": args.percentiles}
            )
            return result(value["value"])
        return await series(g, command, args.mode)
    if args.mode == "difference" and len(names) > 1:
        raise ValueError("difference takes one source and other")
    async with AsyncExitStack() as stack:
        panels: list[Panel] = []
        frames: list[_marks.Frame] = []
        first: tuple[list[tuple[Object, int | None]], list[int]] | None = None
        for name in names:
            src = reference(tools, name, args.mode)
            if args.mode == "difference":
                found = await difference_planes(tools, g, stack, args, name, src)
            else:
                found = await mode_planes(g, stack, args, name, src, compare=len(names) > 1)
            new_panels, new_frames, reads, shape = found
            if first is None:
                first = (reads, shape)
            elif shape != first[1]:
                raise ValueError(f"{name} has shape {shape}; {primary} has {first[1]}")
            panels += new_panels
            frames += new_frames
        assert first is not None
        masks: list[list[tuple[np.ndarray, _marks.Color]]] = [[] for _ in frames]
        if args.features:
            planes = await outlines(tools, g, stack, first[0], first[1], args.features)
            masks = planes * len(names)
        return await asyncio.to_thread(finish, tools, args, panels, frames, masks)


async def mode_planes(
    g: Generation, stack: AsyncExitStack, args: View, name: str, src: Object, *, compare: bool
) -> tuple[list[Panel], list[_marks.Frame], list[tuple[Object, int | None]], list[int]]:
    """Read one source's planes in the requested mode, with where each sits in it."""
    request = {"type": "read", "source": src, "roi": args.roi, "stride": args.stride}
    request |= {"mode": args.mode, "axis": args.axis, "index": args.index}
    shape = (await metadata(g, src))["shape"] if compare or args.features else []
    v = await stack.enter_async_context(temporary(g, request))
    title = ("Alignment working image " if name.endswith(".image") else "") + name
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
    frames = [
        _marks.frame_of(
            [a["name"] for a in item["array"]["axes"]], item["window"], item["dimension"]
        )
        for item in v["planes"]
    ]
    return panels, frames, [(request, None)], shape


def finish(
    tools: Tools,
    args: View,
    panels: list[Panel],
    frames: list[_marks.Frame],
    masks: list[list[tuple[np.ndarray, _marks.Color]]],
) -> CallToolResult:
    """Render the planes, keep where each landed for pixel marks, and report the marks."""
    from tomojax.cli.mcp.tools import result

    renders = tools.core.renders

    def placements(render: str) -> list[_marks.Placement]:
        if render not in renders:
            raise ValueError(f"{render} is not one of the last {RENDERS} view images")
        return renders[render]

    marks = _marks.locate(args.marks, placements)
    _marks.check(marks, frames[0].axes)
    scales = shared_limits(panels, args.limits)
    if args.marks or args.features or args.ticks or args.grid:
        ticks = args.ticks if args.ticks is not None else True
        overlays = [
            _marks.Overlay(frame, marks, tuple(found), ticks)
            for frame, found in zip(frames, masks, strict=True)
        ]
        pictures, reports, layouts = _marks.annotate(
            panels, scales, overlays, marks, grid=args.grid
        )
    else:
        pictures, reports, layouts = [], [], []
        for i in range(0, len(panels), 3):
            rects: list[tuple[float, float, float, float]] = []
            pictures.append(png(compose(panels[i : i + 3], scales[i : i + 3], placed=rects)))
            layouts.append(
                [
                    _marks.Placement(
                        frames[i + j], _marks.Cell(i // 3, *rect), panels[i + j].pixels.shape[:2]
                    )
                    for j, rect in enumerate(rects)
                ]
            )
    ids = []
    for layout in layouts:
        ids.append(f"render-{next(tools.core.render_count)}")
        renders[ids[-1]] = layout
        while len(renders) > RENDERS:
            renders.pop(next(iter(renders)))
    for report in reports:
        for found in report["panels"]:  # pyright: ignore[reportGeneralTypeIssues]
            if "image" in found:
                found["image"] = ids[found["image"]]
    sizes = [Image.open(BytesIO(picture)).size for picture in pictures]
    data: Object = {
        "panels": [
            {
                "array": p.title,
                "labels": p.labels,
                "unit": p.unit,
                "intensity_limits": scale,
                "image": ids[i // 3],
            }
            for i, (p, scale) in enumerate(zip(panels, scales, strict=True))
        ],
        "images": [
            {"image": render, "width": w, "height": h}
            for render, (w, h) in zip(ids, sizes, strict=True)
        ],
        "image_limits": {"edge": MAX_EDGE, "pixels": MAX_PIXELS},
    }
    if len(set(scales)) == 1:
        data["intensity_limits"] = scales[0]
    if marks:
        data["marks"] = reports
    if args.features:
        data["features"] = [
            {"handle": name, "color": color}
            for name, color in zip(args.features, FEATURE_COLORS, strict=False)
        ]
    reply = result(data)
    reply.content.extend(image(picture) for picture in pictures)
    return reply


async def difference_planes(
    tools: Tools, g: Generation, stack: AsyncExitStack, args: View, name: str, src: Object
) -> tuple[list[Panel], list[_marks.Frame], list[tuple[Object, int | None]], list[int]]:
    """Read matching planes of the source and ``other`` and subtract them."""
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
            f"Difference requires equal volume grids. {name}: {grid(meta)}. "
            f"{args.other}: {grid(second)}"
        )
    order = [axes.index(a["name"]) for a in second["axes"]]
    roi = args.roi if args.roi is not None else tuple((0, n) for n in meta["shape"])
    strides = args.stride if args.stride is not None else (1,) * len(axes)
    if len(roi) != 3 or len(strides) != 3 or min(strides) < 1:
        raise ValueError("roi and stride need three dimensions and positive strides")
    command = {"type": "read", "source": src}
    panels, frames = [], []
    reads: list[tuple[Object, int | None]] = []
    for d, index, selected, stride, sampling in plane_windows(
        axes, meta["shape"], args.mode, args.axis, args.index, roi, strides
    ):
        read = {**command, "roi": selected, "stride": stride}
        reads.append((read, d))
        window = [(lo, hi, step) for (lo, hi), step in zip(selected, stride, strict=True)]
        frames.append(_marks.frame_of(axes, window, d))
        a = (await stack.enter_async_context(temporary(g, read)))["array"]
        read_b = {
            **read,
            "source": source_b,
            "roi": [selected[i] for i in order],
            "stride": [stride[i] for i in order],
        }
        async with temporary(g, read_b) as w:
            pixels = np.asarray(load(a)).astype(float) - load(w["array"]).transpose(
                np.argsort(order)
            )
        title = f"{name} minus {args.other}"
        panels.append(plane(a, pixels, title, dim=d, source_index=index, sampling=sampling))
    return panels, frames, reads, meta["shape"]


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
