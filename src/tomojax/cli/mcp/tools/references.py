"""Resolve model-facing references at data fields, including nested array fields."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from msgspec import inspect as mi

from tomojax.cli.operations import PathInput
from tomojax.cli.products import VOLUME_DATASET
from tomojax.wire.arrays import Member, Npy, Record

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tomojax.cli.mcp.ledger import Ledger
    from tomojax.cli.mcp.supervisor import Object

PRIMARY = {
    "scan": "projections",
    "frames": "counts",
    "reconstruction": "volume",
    "alignment": "image",
    "simulation": "phantom",
}


def resolve(
    value: str,
    *,
    base: Path,
    ledger: Ledger,
    field: Literal["array", "source", "calibration"] = "array",
    default_dataset: str | None = None,
) -> Object | str:
    """Convert one typed reference to its worker representation."""
    if value.startswith("@"):
        ref, _, member = value.partition(".")
        record = ledger.lookup(ref)
        if "artifact" in record:
            if member or field == "source":
                raise ValueError(f"{ref} names an array, not an object with members")
            return {"type": "npy", "path": record["path"], "artifact": record["artifact"]}
        if not member:
            member = "self" if field == "source" else PRIMARY.get(record["kind"])
            if member is None:
                raise ValueError(f"{ref} has no primary array")
        return {"type": "member", "handle": record["worker"], "member": member}
    filename, separator, dataset = value.partition("::/")
    path = str((base / Path(filename).expanduser()).resolve())
    if separator:
        return {"type": "hdf5", "path": path, "dataset": "/" + dataset}
    if default_dataset and Path(path).suffix.lower() in {".nx", ".nxs", ".h5", ".hdf", ".hdf5"}:
        return {"type": "hdf5", "path": path, "dataset": default_dataset}
    if field == "calibration" and Path(path).suffix.lower() != ".npy":
        return path
    return {"type": "path" if field == "source" else "npy", "path": path}


def choices(info: mi.Type) -> Iterator[mi.Type]:
    """Flatten named and anonymous union branches in a msgspec field."""
    if isinstance(info, mi.UnionType):
        for branch in info.types:
            yield from choices(branch)
    else:
        yield info


def wire_arguments(command: Object, model: type[Record], *, base: Path, ledger: Ledger) -> Object:
    """Translate references only where the worker schema expects storage or an object.

    Walk tagged choices as well as collections so nested support masks and pose
    tables use exactly the same reference syntax as top-level calibration arrays.
    """

    def translate(value: Any, info: mi.Type, name: str = "") -> Any:
        options = list(choices(info))
        structs = {t.cls: t for t in options if isinstance(t, mi.StructType)}
        if isinstance(value, str):
            if Npy in structs:
                field = (
                    "calibration" if any(isinstance(t, mi.StrType) for t in options) else "array"
                )
                return resolve(
                    value,
                    base=base,
                    ledger=ledger,
                    field=field,
                    default_dataset=VOLUME_DATASET if name == "volume" else None,
                )
            if PathInput in structs or (Member in structs and value.startswith("@")):
                return resolve(value, base=base, ledger=ledger, field="source")
        if isinstance(value, dict):
            for struct in structs.values():
                if struct.tag is None or struct.tag == value.get(struct.tag_field):
                    fields = {f.encode_name: f.type for f in struct.fields}
                    return {
                        k: translate(v, fields[k], k) if k in fields else v
                        for k, v in value.items()
                    }
        if isinstance(value, list | tuple):
            collection = next((t for t in options if isinstance(t, mi.CollectionType)), None)
            if collection:
                return [translate(item, collection.item_type) for item in value]
        return value

    return translate(command, mi.type_info(model))
