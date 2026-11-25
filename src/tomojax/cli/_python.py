"""Execute a trusted Python call against the session's handles."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass
from io import BytesIO, StringIO
import traceback
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import msgspec
import numpy as np
from PIL import Image

import tomojax as tj
from tomojax.cli import execution as ex, products as p
from tomojax.cli._browse import indexed, pixels as pixels_of
from tomojax.cli._features import feature_handle, owned, tight
from tomojax.cli._handles import handle, stage_array
from tomojax.cli._marks import Overlay, annotate, check, frame_of, parse
from tomojax.cli._planes import plane_windows
from tomojax.cli.envelope import INLINE_LIMIT
from tomojax.cli.images import Panel, limits, raster
from tomojax.cli.mcp.ledger import new_reference
from tomojax.wire.arrays import Member, Npy
from tomojax.wire.progress import Info

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    from tomojax.cli import operations as op
    from tomojax.cli._handles import Root
    from tomojax.cli._marks import Colour, Frame
    from tomojax.wire.arrays import Array, Storage

_KINDS = {
    tj.Scan: "scan",
    tj.Frames: "frames",
    tj.Simulation: "sim",
    tj.Reconstruction: "recon",
    tj.Alignment: "align",
}
_SHOWN = 10000  # characters kept from each end of a captured stream
_LISTED = 16  # arrays up to this size are returned as lists


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
        # What this call registered, usable by name before the call publishes it.
        self.made: dict[str, object] = {}

    def register(self, item: Root | jax.Array | np.ndarray) -> str:
        if isinstance(item, np.ndarray | jax.Array):
            name = new_reference("array")
            a = np.asarray(item)
            self.products[name] = p.ArrayResult(indexed(stage_array(self.context, a), a))
            self.made[name] = a
            return name
        kind = next((kind for cls, kind in _KINDS.items() if isinstance(item, cls)), None)
        if not kind:
            raise ValueError("register expects a TomoJAX object or array")
        name = new_reference(kind)
        self.roots.append((name, handle(item)))
        self.made[name] = item
        return name

    def registered(self, item: object) -> tuple[Any, Array, Callable[[Storage], np.ndarray]]:
        """An object's primary array or named member, its registration and coordinate reader.

        ``item`` is an ID (``ID.member`` selects a member), an object or an array. The
        reader returns the values of a coordinate vector the registration refers to.
        """
        member = ""
        identifier = "python"  # objects made in this call are bound under this name
        if isinstance(item, str) and item in self.made:
            item = self.made[item]
        if isinstance(item, str):
            base, _, member = item.partition(".")
            ref = self.inputs.reference(base)
            if isinstance(ref, Npy):
                if member:
                    raise KeyError(f"{base} is an array and has no member {member}")
                data = np.load(ref.path, mmap_mode="r", allow_pickle=False)
                return data, self.context.leases[ref].metadata or indexed(ref, data), _load
            identifier, item = ref.handle, self.context.leases[ref].value
        if isinstance(item, ex.Feature):
            return item, item.registration, _load
        if isinstance(item, ex.RegisteredArray):
            return item.data, item.metadata, _load
        if isinstance(item, tj.Scan | tj.Frames | tj.Simulation | tj.Reconstruction | tj.Alignment):
            key = member or next(key for cls, key in _PRIMARY.items() if isinstance(item, cls))
            bound = handle(item).bind(identifier)
            value = bound.members.get(key)
            if isinstance(value, ex.RegisteredArray):

                def coordinate(ref: Storage) -> np.ndarray:
                    if isinstance(ref, Member):
                        return np.asarray(
                            cast("ex.RegisteredArray", bound.members[ref.member]).data
                        )
                    return _load(ref)

                return value.data, value.metadata, coordinate
            if value is None or value is item:
                raise KeyError(f"this {type(item).__name__} has no array {key}")
            return self.registered(value)
        if member:
            raise KeyError(f"{type(item).__name__} has no member {member}")
        data = np.asarray(item)
        return data, indexed(Member("python", "array"), data), _load

    def register_feature(
        self, mask: ArrayLike | jax.Array, *, like: object, name: str = "python"
    ) -> str:
        values, registration, coordinate = self.registered(like)
        whole = np.asarray(mask, dtype=bool)
        if whole.shape != tuple(registration.shape):
            raise ValueError(f"mask shape {whole.shape} differs from the grid {registration.shape}")
        inside, box = tight(whole, tuple(slice(0, n) for n in whole.shape))
        grid = owned(registration, coordinate, self.context)
        feature = ex.Feature(name, inside, box, grid, {"type": "python"})
        named = new_reference("feature")
        self.roots.append((named, feature_handle(feature, values)))
        self.made[named] = feature
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
        data, meta, _ = self.registered(source)
        names = [a.name for a in meta.axes]
        shape = tuple(meta.shape)
        second = self.registered(other)[0] if other is not None else None
        masks = []
        for f in features:
            feature = f
            if isinstance(f, str):
                feature = self.made[f] if f in self.made else self.inputs[f]
            if not isinstance(feature, ex.Feature) or feature.shape != shape:
                raise ValueError(f"{f!r} is not a feature on a {shape} grid")
            masks.append(feature)
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

    def __getitem__(self, name: str) -> Any:
        """The object or array behind an ID, or one of its members with ``ID.member``."""
        base, _, member = name.partition(".")
        ref = self.reference(base)
        value = (
            self._leases[ref].value
            if isinstance(ref, Member)
            else np.load(ref.path, mmap_mode="r", allow_pickle=False)
        )
        if not member:
            return value
        if not hasattr(value, member):
            raise KeyError(f"{base} has no member {member}")
        return getattr(value, member)

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
        return registrations, msgspec.to_builtins(value, enc_hook=summarise)
    finally:
        for key in ("handles", "register", "register_feature", "show", "view"):
            namespace.pop(key, None)
        kept = {k: v for k, v in namespace.items() if k != "__builtins__"}
        context.measure(array_bytes(kept, set()))


def _load(ref: Storage) -> np.ndarray:
    """A coordinate vector stored in a NumPy file."""
    if not isinstance(ref, Npy):
        raise ValueError(f"coordinate vector {ref} is not a published array")
    return np.load(ref.path, mmap_mode="r", allow_pickle=False)


def summarise(value: object) -> object:
    """Keep returned values small: short arrays as lists, longer ones as a summary."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray | jax.Array):
        a = np.asarray(value)
        if a.size <= _LISTED:
            return a.tolist()
        finite = a[np.isfinite(a)] if a.dtype.kind in "fc" else a.ravel()
        moments = (
            {"min": finite.min().item(), "max": finite.max().item(), "mean": finite.mean().item()}
            if finite.size and a.dtype.kind in "biuf"
            else {}
        )
        return {"shape": list(a.shape), "dtype": str(a.dtype), **moments}
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "..."


def _clip(text: str) -> str:
    missing = len(text) - 2 * _SHOWN
    if missing <= 0:
        return text
    note = f"[{missing} characters omitted; write long output to a file and read it in parts]"
    return f"{text[:_SHOWN]}\n{note}\n{text[-_SHOWN:]}"


def error_report(error: BaseException, code: str, *, full: bool) -> str:
    """The exception with the lines of the submitted code it passed through.

    ``full`` gives the traceback from the first submitted line down into the library.
    """
    frames = traceback.extract_tb(error.__traceback__)
    start = next((i for i, f in enumerate(frames) if f.filename == "<python>"), len(frames))
    if full:
        trail = traceback.format_list(frames[start:])
    else:
        lines = code.splitlines()
        trail = [
            f"line {f.lineno}: {lines[f.lineno - 1].strip()}\n"
            for f in frames[start:]
            if f.filename == "<python>" and f.lineno and f.lineno <= len(lines)
        ]
    return "".join(trail + traceback.format_exception_only(error))


def run_python(cmd: op.Python, context: ex.ExecutionContext) -> ex.Success:
    out, err = StringIO(), StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            registrations, value = execute_in_namespace(cmd, context)
    except BaseException as error:
        message = error_report(error, cmd.code, full=cmd.traceback)
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
