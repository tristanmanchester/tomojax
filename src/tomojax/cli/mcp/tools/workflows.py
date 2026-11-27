"""Model-facing forms derived from the worker's workflows and reference fields."""

from __future__ import annotations

from functools import cache, reduce
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
    w.BeamHardening | w.Paganin | w.RadialDistortion | w.RejectViews | Rings | w.Zingers,
    ...,
]
DETAIL = ("detail", Literal["summary", "full"], "summary")
DRY = ("dry_run", bool, False)
EDIT = (op.Select, op.Crop, op.Bin, op.Poses, op.Combine)
PREVIEWS = (op.Slice, op.CenterSweep, op.CorrectionPreview, op.RegionPreview)
WORKFLOW_HELP = {
    op.Correct: (
        "Convert raw frames to line integrals or apply ordered corrections to an existing "
        "scan. Use the corrected scan for alignment and reconstruction."
    ),
    op.Reconstruct: (
        "Reconstruct a volume from a scan and inspect three orthogonal slices. Use dry_run "
        "to see the selected method, grid and memory placement."
    ),
    op.ReconstructSlabs: (
        "Reconstruct raw frames in slabs and write the volume directly to disk. Use this "
        "when the full acquisition or volume exceeds memory."
    ),
    op.Align: (
        "Fit the rotation center, detector geometry or per-view motion. Use the aligned scan "
        "for reconstruction, inspect the fitted poses, and check disturbed_views for objects "
        "crossing the field."
    ),
    op.Evaluate: (
        "Compare reconstruction or alignment settings on held-out projections. Use the "
        "scores alongside slice comparisons to choose settings."
    ),
    op.Simulate: (
        "Generate a synthetic scan and its ground-truth phantom to check reconstruction "
        "accuracy or test acquisition artifacts."
    ),
    op.Project: (
        "Forward-project a volume through the source scan geometry. Supply a handle member, "
        "session array, .npy file, file.h5::/dataset or saved TomoJAX volume in (z, y, x) order."
    ),
    op.Backproject: (
        "Backproject detector values onto a volume grid with the exact transpose of the "
        "forward projector. Supply an opened scan's .projections, a session array, .npy file or "
        "file.h5::/dataset."
    ),
    op.Feature: (
        "Make a named feature: a mask on the grid of a volume or projection stack, from a box, "
        "sphere, cylinder or prism in array indices (the coordinates view marks use), a region "
        "grown from seed points, or a band around another feature. Outline it with view "
        "features on any source with the same grid; in python it is handles[id].mask."
    ),
    op.Save: (
        "Save a TomoJAX dataset with its provenance manifest so it can be reopened, and return "
        "the written paths. Set include_scan to keep the source scan with a reconstruction."
    ),
    op.Export: (
        "Export projections, a sinogram or a volume in the format selected by the suffix for other "
        "software. Use dry_run to check destinations before writing."
    ),
}
MODELS = {str(model.__struct_config__.tag): model for model in (*WORKFLOW_HELP, *EDIT, *PREVIEWS)}
REFERENCES = {
    "source": Id,
    "sources": tuple[Id, ...],
    "region": str | None,
    "volume": Id,
    "projections": Id,
    "poses": str | None,
    "corrections": Steps,
    "config": dict[str, Any] | None,
    "feature": Id,
}


@cache
def form(model: type[Record], *, tagged: bool = False) -> type[Record]:
    """Keep wire defaults and types except where the server owns the representation."""
    fields = []
    for field in msgspec.structs.fields(model):
        entry = (field.name, REFERENCES.get(field.name, field.type))
        default = field.default
        if model is op.Evaluate:
            recon = form(op.ReconstructSettings, tagged=True)
            if field.name == "candidates":
                choices = recon | form(op.AlignSettings, tagged=True)
                entry = (field.name, tuple[choices, ...])
            if field.name == "reference":
                entry = (field.name, recon)
                default = msgspec.convert(msgspec.to_builtins(default), type=recon)
        if model is op.Feature and field.name == "recipe":
            shapes = op.Box | op.Sphere | op.Cylinder | op.Prism | op.Seeded
            entry = (field.name, shapes | form(op.Band, tagged=True))
        fields.append(entry if field.required else (*entry, default))
    if model in (op.Slice, op.CenterSweep):
        interval = tuple[int, int]
        fields.append(("roi", tuple[interval, interval, interval] | None, None))
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
            namespace={"__doc__": model.__doc__ if tagged else WORKFLOW_HELP[model]},
        ),
    )


def forms() -> dict[str, type[Record]]:
    """Expose one tool per workflow and a tagged operation for scan edits."""
    models = {name: form(cls) for name, cls in MODELS.items() if cls not in (*EDIT, *PREVIEWS)}
    edit = reduce(or_, (form(cls, tagged=True) for cls in EDIT))
    models["edit_scan"] = cast(
        "type[Record]",
        msgspec.defstruct(
            "EditScan",
            [("operation", edit), DRY, DETAIL],
            bases=(Record,),
            namespace={
                "__doc__": (
                    "Select views, crop or bin a detector, replace poses, or combine scans. "
                    "Pass the new handle to the next workflow."
                )
            },
        ),
    )
    preview = reduce(or_, (form(cls, tagged=True) for cls in PREVIEWS))
    models["preview"] = cast(
        "type[Record]",
        msgspec.defstruct(
            "Preview",
            [("operation", preview)],
            bases=(Record,),
            namespace={
                "__doc__": (
                    "Preview a slice, sweep the center or compare corrections. "
                    "For a slice or sweep, zoom in on detail with operation.roi: three half-open "
                    "volume index intervals in (z, y, x) order containing the slice index. "
                    "A region preview returns a handle for reconstruction."
                )
            },
        ),
    )
    return models
