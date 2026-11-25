"""Execute a trusted Python call against the session's handles."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass
from io import BytesIO, StringIO
import traceback
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Any, Literal

import jax
import msgspec
import numpy as np
from PIL import Image

import tomojax as tj
from tomojax.cli import execution as ex, products as p
from tomojax.cli._browse import indexed, pixels as pixels_of
from tomojax.cli._features import feature_handle
from tomojax.cli._handles import handle, stage_array
from tomojax.cli._marks import Overlay, annotate, check, frame_of, parse
from tomojax.cli._planes import plane_windows
from tomojax.cli.envelope import INLINE_LIMIT
from tomojax.cli.images import Panel, limits, raster
from tomojax.cli.mcp.ledger import new_reference
from tomojax.wire.arrays import Member, Npy
from tomojax.wire.progress import Info

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from tomojax.cli import operations as op
    from tomojax.cli._handles import Root
    from tomojax.cli._marks import Colour, Frame
    from tomojax.wire.arrays import Array

_KINDS = {
    tj.Scan: "scan",
    tj.Frames: "frames",
    tj.Simulation: "sim",
    tj.Reconstruction: "recon",
    tj.Alignment: "align",
}
_SHOWN = 10000  # characters kept from each end of a captured stream


def array_bytes(value: object, seen: set[int]) -> int:
    """Bytes of the arrays reachable from ``value`` through containers and dataclasses."""
    if id(value) in seen or isinstance(value, type | ModuleType | FunctionType):
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray | jax.Array):
        return value.nbytes
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, list | tuple | set | frozenset):
        children = value
    elif is_dataclass(value):
        children = (getattr(value, f.name) for f in fields(value))
    else:
        return 0
    return sum(array_bytes(child, seen) for child in children)


# The array a TomoJAX object stands for when it is used as a grid or an image.
_PRIMARY = {
    tj.Scan: "projections",
    tj.Frames: "counts",
    tj.Simulation: "phantom",
    tj.Reconstruction: "volume",
    tj.Alignment: "image",
}


class Registrations:
    """The handles, arrays and images one Python call publishes when it succeeds."""

    def __init__(self, context: ex.ExecutionContext, inputs: HandleInputs) -> None:
        self.context = context
        self.inputs = inputs
        self.roots = []
        self.products = {}
        self.images = []

    def register(self, item: Root | jax.Array | np.ndarray) -> str:
        if isinstance(item, np.ndarray | jax.Array):
            name = new_reference("array")
            a = np.asarray(item)
            self.products[name] = p.ArrayResult(indexed(stage_array(self.context, a), a))
            return name
        kind = next((kind for cls, kind in _KINDS.items() if isinstance(item, cls)), None)
        if not kind:
            raise ValueError("register expects a TomoJAX object or array")
        name = new_reference(kind)
        self.roots.append((name, handle(item)))
        return name

    def registered(self, item: object) -> tuple[Any, Array]:
        """An object's primary array and its registration: axes, coordinates and unit."""
        if isinstance(item, str):
            ref = self.inputs.reference(item)
            if isinstance(ref, Npy):
                data = np.load(ref.path, mmap_mode="r", allow_pickle=False)
                return data, self.context.leases[ref].metadata or indexed(ref, data)
            item = self.context.leases[ref].value
        if isinstance(item, ex.Feature):
            return item.mask, item.registration
        if isinstance(item, ex.RegisteredArray):
            return item.data, item.metadata
        if isinstance(item, tj.Scan | tj.Frames | tj.Simulation | tj.Reconstruction | tj.Alignment):
            key = next(key for cls, key in _PRIMARY.items() if isinstance(item, cls))
            member = handle(item).bind("python").members.get(key)
            if not isinstance(member, ex.RegisteredArray):
                raise ValueError(f"this {type(item).__name__} has no {key}")
            return member.data, member.metadata
        data = np.asarray(item)
        return data, indexed(Member("python", "array"), data)

    def register_feature(
        self, mask: ArrayLike | jax.Array, *, like: object, name: str = "python"
    ) -> str:
        values, registration = self.registered(like)
        feature = ex.Feature(name, np.asarray(mask, dtype=bool), registration, {"type": "python"})
        named = new_reference("feature")
        self.roots.append((named, feature_handle(feature, values)))
        return named

    def show(
        self,
        array: ArrayLike | jax.Array,
        *,
        marks: Sequence[object] = (),
        ticks: bool | None = None,
    ) -> list[dict[str, object]] | None:
        pixels = np.asarray(array)
        if not marks and not ticks:
            panel = Panel("Python", pixels, (), "unspecified")
            self.images.append(stage_array(self.context, raster(panel, *limits([panel]))))
            return None
        names = ("row", "col")
        window = [(0, n, 1) for n in pixels.shape[:2]]
        panel = Panel("Python", pixels, (f"{pixels.shape[0]} x {pixels.shape[1]}",), "unspecified")
        return self.overlaid([panel], [frame_of(names, window, None)], marks, [[]], ticks)

    def view(
        self,
        source: object,
        *,
        mode: Literal["planes", "projection", "sinogram"] = "planes",
        axis: str | None = None,
        index: int | None = None,
        roi: Sequence[tuple[int, int]] | None = None,
        stride: Sequence[int] | None = None,
        other: object = None,
        marks: Sequence[object] = (),
        features: Sequence[object] = (),
        ticks: bool | None = None,
    ) -> list[dict[str, object]]:
        """Render planes like the view tool; with ``other``, the difference from it."""
        data, meta = self.registered(source)
        names = [a.name for a in meta.axes]
        shape = tuple(meta.shape)
        second = self.registered(other)[0] if other is not None else None
        masks = []
        for f in features:
            feature = self.inputs[f] if isinstance(f, str) else f
            if not isinstance(feature, ex.Feature) or feature.mask.shape != shape:
                raise ValueError(f"{f!r} is not a feature on a {shape} grid")
            masks.append(feature.mask)
        panels, frames, outlines = [], [], []
        for d, at, selected, step, sampling in plane_windows(
            names,
            shape,
            mode,
            axis,
            index,
            roi if roi is not None else [(0, n) for n in shape],
            tuple(stride) if stride is not None else (1,) * len(shape),
        ):
            window = [(lo, hi, s) for (lo, hi), s in zip(selected, step, strict=True)]
            ranges = tuple(range(*w) for w in window)

            def flat(
                values: Any, d: int | None = d, ranges: tuple[range, ...] = ranges
            ) -> np.ndarray:
                plane = pixels_of(values, ranges)
                return plane if d is None else np.take(plane, 0, axis=d)

            pixels = flat(data).astype(float)
            title = "Python view"
            if second is not None:
                pixels = pixels - flat(second)
                title += " (difference)"
            where = f"{names[d]}[{at}]" if d is not None else ""
            span = ", ".join(
                f"{n} [{lo}, {hi}) step {s}"
                for n, (lo, hi, s) in zip(names, window, strict=True)
                if n != (names[d] if d is not None else None)
            )
            panels.append(
                Panel(title, pixels, (where, span, *sampling), meta.unit, 1.0, meta.domain)
            )
            frames.append(frame_of(names, window, d))
            colours = ("cyan", "magenta", "green", "orange", "red")
            outlines.append(
                [(flat(mask).astype(bool), c) for mask, c in zip(masks, colours, strict=False)]
            )
        return self.overlaid(panels, frames, marks, outlines, ticks)

    def overlaid(
        self,
        panels: list[Panel],
        frames: list[Frame],
        marks: Sequence[object],
        outlines: list[list[tuple[np.ndarray, Colour]]],
        ticks: bool | None,
    ) -> list[dict[str, object]]:
        """Stage the annotated montage and return where each mark landed."""
        parsed = parse(marks)
        check(parsed, frames[0].axes)
        scale = limits(panels)
        overlays = [
            Overlay(frame, parsed, tuple(found), ticks is not False)
            for frame, found in zip(frames, outlines, strict=True)
        ]
        images, reports = annotate(panels, [scale] * len(panels), overlays, parsed)
        for png in images:
            pixels = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
            self.images.append(stage_array(self.context, pixels))
        return reports


class HandleInputs(Mapping):
    """The session's handles and arrays by reference, read when indexed."""

    def __init__(
        self, aliases: Mapping[str, Member | Npy], leases: Mapping[Member | Npy, ex.Lease]
    ) -> None:
        self._aliases = aliases
        self._leases = leases

    def reference(self, name: str) -> Member | Npy:
        """The pinned reference behind a handle or array ID."""
        if name not in self._aliases:
            raise KeyError(f"{name} is not a live handle; status lists them")
        return self._aliases[name]

    def __getitem__(self, name: str) -> ex.HandleValue | Npy | np.ndarray:
        ref = self.reference(name)
        if isinstance(ref, Member):
            return self._leases[ref].value
        return np.load(ref.path, mmap_mode="r", allow_pickle=False)

    def __iter__(self) -> Iterator[str]:
        return iter(self._aliases)

    def __len__(self) -> int:
        return len(self._aliases)


def execute_in_namespace(
    cmd: op.Python, context: ex.ExecutionContext
) -> tuple[Registrations, object]:
    namespace = context.namespace
    if cmd.reset:
        namespace.clear()
    inputs = HandleInputs(cmd.aliases, context.leases)
    registrations = Registrations(context, inputs)
    namespace.update(
        tj=tj,
        np=np,
        handles=inputs,
        register=registrations.register,
        register_feature=registrations.register_feature,
        show=registrations.show,
        view=registrations.view,
    )
    try:
        tree = ast.parse(cmd.code, filename="<python>")
        last = tree.body[-1] if tree.body else None
        expression = None
        if isinstance(last, ast.Expr):
            tree.body.pop()
            expression = ast.Expression(last.value)
        exec(compile(tree, "<python>", "exec"), namespace)
        value = None
        if expression is not None:
            value = eval(compile(expression, "<python>", "eval"), namespace)
        if isinstance(value, np.ndarray | jax.Array):
            value = registrations.register(value)
        value = msgspec.to_builtins(value.item() if isinstance(value, np.generic) else value)
        return registrations, value
    finally:
        for key in ("handles", "register", "register_feature", "show", "view"):
            namespace.pop(key, None)
        kept = {k: v for k, v in namespace.items() if k != "__builtins__"}
        context.measure(array_bytes(kept, set()))


def _clip(text: str) -> str:
    missing = len(text) - 2 * _SHOWN
    if missing <= 0:
        return text
    return f"{text[:_SHOWN]}\n[{missing} characters omitted]\n{text[-_SHOWN:]}"


def run_python(cmd: op.Python, context: ex.ExecutionContext) -> ex.Success:
    out, err = StringIO(), StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            registrations, value = execute_in_namespace(cmd, context)
    except BaseException as error:
        frames = traceback.extract_tb(error.__traceback__)
        start = next((i for i, f in enumerate(frames) if f.filename == "<python>"), len(frames))
        message = "".join(
            traceback.format_list(frames[start:]) + traceback.format_exception_only(error)
        )
        detail = _clip(out.getvalue()) + _clip(err.getvalue()) + message
        if len(msgspec.json.encode(detail)) > INLINE_LIMIT:
            raise ValueError("Python exception too large for the worker pipe") from error
        raise ValueError(detail) from error
    return ex.Success(
        p.PythonResult(
            _clip(out.getvalue()),
            _clip(err.getvalue()),
            value,
            registrations.products,
            tuple(registrations.images),
        ),
        Info({}, {}),
        p.Provenance("python", {"code": cmd.code, "reset": cmd.reset}, ()),
        tuple(registrations.roots),
    )
