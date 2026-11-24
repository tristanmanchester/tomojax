"""Model-facing forms derived from the worker's workflows and reference fields."""

from __future__ import annotations

from functools import reduce
from operator import or_
from typing import Any, Literal, cast

import msgspec

from tomojax.cli import operations as op, wire_options as w
from tomojax.wire.arrays import Id, Record


class Rings(Record, tag="Rings"):
    """Correct rings using optional geometry from a scan reference."""

    geometry: str | None = None
    views: int | None = 512


type Steps = tuple[
    w.BeamHardening
    | w.Paganin
    | w.RadialDistortion
    | w.RejectViews
    | Rings
    | w.Stripes
    | w.Zingers,
    ...,
]
DETAIL = ("detail", Literal["summary", "full"], "summary")
DRY = ("dry_run", bool, False)
EDIT = (op.Select, op.Crop, op.Bin, op.Poses, op.Combine)
MODELS = {
    str(model.__struct_config__.tag): model
    for model in (
        op.Correct,
        op.Reconstruct,
        op.ReconstructSlabs,
        op.Align,
        op.Simulate,
        op.Project,
        op.Backproject,
        op.Save,
        op.Export,
        *EDIT,
    )
}
REFERENCES = {
    "source": Id,
    "sources": tuple[Id, ...],
    "region": str | None,
    "volume": Id,
    "projections": Id,
    "poses": str | None,
    "corrections": Steps,
    "config": dict[str, Any] | None,
}


def form(model: type[Record], *, tagged: bool = False) -> type[Record]:
    """Keep wire defaults and types except where the server owns the representation."""
    fields = []
    for field in msgspec.structs.fields(model):
        entry = (field.name, REFERENCES.get(field.name, field.type))
        fields.append(entry if field.required else (*entry, field.default))
    if not tagged:
        fields.extend([DRY, DETAIL])
    return cast(
        "type[Record]",
        msgspec.defstruct(
            model.__name__,
            fields,
            bases=(Record,),
            kw_only=True,
            tag=model.__struct_config__.tag if tagged else None,
            namespace={"__doc__": model.__doc__},
        ),
    )


def forms() -> dict[str, type[Record]]:
    """Expose one tool per workflow and a tagged operation for scan edits."""
    models = {name: form(cls) for name, cls in MODELS.items() if cls not in EDIT}
    edit = reduce(or_, (form(cls, tagged=True) for cls in EDIT))
    models["edit_scan"] = cast(
        "type[Record]",
        msgspec.defstruct(
            "EditScan",
            [("operation", edit), DRY, DETAIL],
            bases=(Record,),
            namespace={
                "__doc__": "Select, crop, bin, replace poses or combine scans into a new handle."
            },
        ),
    )
    return models
