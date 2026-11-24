"""Execute a trusted Python call against pinned, read-only session inputs."""

import ast
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import traceback
from types import FunctionType, MappingProxyType, ModuleType
from typing import Any
from weakref import ref as weakref

import h5py
import jax
import msgspec
import numpy as np

import tomojax as tj
from tomojax.cli import execution as ex, products as p
from tomojax.cli._browse import indexed
from tomojax.cli._capture import capture
from tomojax.cli._handles import handle, stage_array
from tomojax.cli._options import ancestor_paths
from tomojax.cli.envelope import INLINE_LIMIT
from tomojax.cli.images import Panel, limits, raster
from tomojax.cli.mcp.ledger import reserve
from tomojax.core.files import read_dataset, record_inputs
from tomojax.wire.arrays import Member
from tomojax.wire.progress import Info


def buffers(value, seen, *, metadata=False):
    """Walk array storage without descending into imported modules or user functions."""
    if id(value) in seen or isinstance(value, type | ModuleType | FunctionType):
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray | jax.Array | jax.Device | h5py.Dataset) or hasattr(
        type(value), "frames_at"
    ):
        yield value
        return
    if isinstance(value, HandleInputs):
        children = value.values_by_name.values()
    elif isinstance(value, dict | MappingProxyType):
        children = value.values()
    elif isinstance(value, list | tuple | set | frozenset):
        children = value
    elif is_dataclass(value):
        children = (getattr(value, f.name) for f in fields(value))
    elif metadata and hasattr(value, "__dict__"):
        children = vars(value).values()
    else:
        return
    for child in children:
        yield from buffers(child, seen, metadata=metadata)
    if isinstance(value, MappingProxyType):
        yield value


def owner_of(buffer, issued):
    # NumPy views may point straight at a base a few levels below the array
    # passed to the user. Only trace owners imported from this job's leases.
    while buffer is not None:
        if entry := issued.get(id(buffer)):
            return entry[1]
        buffer = buffer.base if isinstance(buffer, np.ndarray) else None
    return None


def snapshot(value, *, issued, dependencies, source=None):
    memo = {}
    for buf in buffers(value, set(), metadata=True):
        if isinstance(buf, MappingProxyType):
            memo[id(buf)] = MappingProxyType(deepcopy(dict(buf), memo))
            continue
        if isinstance(buf, jax.Device):
            memo[id(buf)] = buf
            continue
        owner = owner_of(buf, issued)
        if source is None and owner is not None:
            dependencies.add(owner)
        shared = source is not None or owner is not None
        if source is not None and hasattr(type(buf), "frames_at"):
            # Frame stacks hold only a storage descriptor and a frame selection.
            # Copy that metadata; indexing the private descriptor reads owned pixels.
            copy = deepcopy(buf)
        elif isinstance(buf, jax.Array):
            copy = buf if shared else buf.copy().block_until_ready()
        else:
            if isinstance(buf, np.ndarray):
                copy = (
                    np.asarray(memoryview(buf).toreadonly()) if shared else np.array(buf, copy=True)
                )
            else:
                copy = np.array(
                    (read_dataset(buf) if isinstance(buf, h5py.Dataset) else buf)[:], copy=True
                )
            copy.flags.writeable = False
        memo[id(buf)] = copy
        if source is not None:
            issued[id(copy)] = (copy, source)
    return deepcopy(value, memo)


class Registrations:
    """All Python-created products share the job's final publication decision."""

    def __init__(self, context, ledger, issued) -> None:
        self.context = context
        self.ledger = ledger
        self.issued = issued
        self.roots = []
        self.products = {}
        self.images = []

    def register(self, item):
        if isinstance(item, np.ndarray | jax.Array):
            name = reserve("array", self.ledger)
            a = np.asarray(item)
            self.products[name] = p.ArrayResult(indexed(stage_array(self.context, a), a))
        else:
            kinds = {
                tj.Scan: "scan",
                tj.Frames: "frames",
                tj.Simulation: "sim",
                tj.Reconstruction: "recon",
                tj.Alignment: "align",
            }
            kind = next((kind for cls, kind in kinds.items() if isinstance(item, cls)), None)
            if not kind:
                raise ValueError("register expects a TomoJAX object or array")
            name = reserve(kind, self.ledger)
            scan = item if isinstance(item, tj.Scan | tj.Frames) else item.scan
            provenance = getattr(item, "info", {}).get("source")
            paths = (
                scan.source.source_path if scan is not None and scan.source else None,
                getattr(item, "source_path", None),
                provenance.get("path") if isinstance(provenance, Mapping) else provenance,
            )
            self.context.namespace.sources.update(
                str(Path(p).resolve()) for p in paths if isinstance(p, str)
            )
            dependencies = set()
            frozen = snapshot(item, issued=self.issued, dependencies=dependencies)
            self.roots.append((name, replace(handle(frozen), dependencies=tuple(dependencies))))
        return name

    def show(self, array):
        panel = Panel("Python", np.asarray(array), (), "unspecified")
        self.images.append(stage_array(self.context, raster(panel, *limits([panel]))))


_active: ContextVar[Registrations] = ContextVar("python_registrations")


def register(item):
    """Resolve at call time so aliases in the persistent namespace remain usable."""
    return _active.get().register(item)


def show(array):
    _active.get().show(array)


class HandleInputs(Mapping):
    """Remember accessed source files across arbitrary transformations in Python."""

    def __init__(self, aliases, job) -> None:
        self._aliases = MappingProxyType(aliases)
        self._job = weakref(job)
        self.values_by_name = {}

    def __getitem__(self, name) -> Any:
        ref = self._aliases[name]
        if name not in self.values_by_name:
            job = self._job() or _active.get()
            lease = job.context.leases[ref]
            job.context.namespace.sources.update(map(str, ancestor_paths((lease.provenance,))))
            if isinstance(ref, Member):
                item = snapshot(lease.value, issued=job.issued, dependencies=set(), source=ref)
            else:
                item = np.load(ref.path, mmap_mode="r", allow_pickle=False)
                job.issued[id(item)] = (item, ref)
            self.values_by_name[name] = item
        return self.values_by_name[name]

    def __contains__(self, name) -> bool:
        return name in self._aliases

    def copy(self) -> dict[str, Any]:
        return dict(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self._aliases)

    def __len__(self) -> int:
        return len(self._aliases)


def execute_in_namespace(cmd, context):
    namespace = context.namespace
    if cmd.reset:
        namespace.clear()
        namespace.sources.clear()
    registrations = Registrations(context, cmd.ledger, {})
    token = _active.set(registrations)
    namespace.update(
        tj=tj,
        np=np,
        handles=HandleInputs(cmd.aliases, registrations),
        register=register,
        show=show,
    )
    try:
        tree = ast.parse(cmd.code, filename="<python>")
        expr = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, "<python>", "exec"), namespace)
        value = None
        if expr is not None:
            value = eval(compile(ast.Expression(expr.value), "<python>", "eval"), namespace)
        if isinstance(value, np.ndarray | jax.Array):
            value = register(value)
        value = msgspec.to_builtins(value.item() if isinstance(value, np.generic) else value)
        return registrations, value
    finally:
        _active.reset(token)
        for key in ("handles", "register", "show"):
            namespace.pop(key, None)
        context.measure(
            sum(
                getattr(a, "nbytes", 0)
                for a in buffers({k: v for k, v in namespace.items() if k != "__builtins__"}, set())
            )
        )


def run_python(cmd, context):
    output = []
    try:
        with capture() as output, record_inputs(context.namespace.sources):
            registrations, value = execute_in_namespace(cmd, context)
    except BaseException as err:
        frames = traceback.extract_tb(err.__traceback__)
        start = next((i for i, f in enumerate(frames) if f.filename == "<python>"), len(frames))
        message = "".join(
            traceback.format_list(frames[start:]) + traceback.format_exception_only(err)
        )
        detail = "".join(output) + message
        if len(msgspec.json.encode(detail)) > INLINE_LIMIT:
            raise ValueError("Python exception too large for the worker pipe") from err
        raise ValueError(detail) from err
    # Persistent globals can retain copies or calculated values even after close.
    # Flatten to unique paths so repeated Python registrations have bounded lineage.
    return ex.Success(
        p.PythonResult(
            output[0], output[1], value, registrations.products, tuple(registrations.images)
        ),
        Info({}, {}),
        p.Provenance(
            "python", {"code": cmd.code, "reset": cmd.reset}, (), context.namespace.sources.copy()
        ),
        tuple(registrations.roots),
    )
