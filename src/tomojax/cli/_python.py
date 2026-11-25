"""Execute a trusted Python call against the session's handles."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass
from io import StringIO
import traceback
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING

import jax
import msgspec
import numpy as np

import tomojax as tj
from tomojax.cli import execution as ex, products as p
from tomojax.cli._browse import indexed
from tomojax.cli._handles import handle, stage_array
from tomojax.cli.envelope import INLINE_LIMIT
from tomojax.cli.images import Panel, limits, raster
from tomojax.cli.mcp.ledger import new_reference
from tomojax.wire.arrays import Member
from tomojax.wire.progress import Info

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from tomojax.cli import operations as op
    from tomojax.cli._handles import Root
    from tomojax.wire.arrays import Npy

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


class Registrations:
    """The handles, arrays and images one Python call publishes when it succeeds."""

    def __init__(self, context: ex.ExecutionContext) -> None:
        self.context = context
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

    def show(self, array: ArrayLike | jax.Array) -> None:
        panel = Panel("Python", np.asarray(array), (), "unspecified")
        self.images.append(stage_array(self.context, raster(panel, *limits([panel]))))


class HandleInputs(Mapping):
    """The session's handles and arrays by reference, read when indexed."""

    def __init__(
        self, aliases: Mapping[str, Member | Npy], leases: Mapping[Member | Npy, ex.Lease]
    ) -> None:
        self._aliases = aliases
        self._leases = leases

    def __getitem__(self, name: str) -> ex.HandleValue | Npy | np.ndarray:
        ref = self._aliases[name]
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
    registrations = Registrations(context)
    namespace.update(
        tj=tj,
        np=np,
        handles=HandleInputs(cmd.aliases, context.leases),
        register=registrations.register,
        show=registrations.show,
    )
    try:
        tree = ast.parse(cmd.code, filename="<python>")
        expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, "<python>", "exec"), namespace)
        value = None
        if expr is not None:
            if TYPE_CHECKING:
                assert isinstance(expr, ast.Expr)
            value = eval(compile(ast.Expression(expr.value), "<python>", "eval"), namespace)
        if isinstance(value, np.ndarray | jax.Array):
            value = registrations.register(value)
        value = msgspec.to_builtins(value.item() if isinstance(value, np.generic) else value)
        return registrations, value
    finally:
        for key in ("handles", "register", "show"):
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
