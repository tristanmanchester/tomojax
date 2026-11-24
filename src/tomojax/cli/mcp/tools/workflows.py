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
PREVIEWS = (op.Slice, op.CentreSweep, op.CorrectionPreview, op.RegionPreview)
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
        "Fit the rotation centre, detector geometry or per-view motion. Use the aligned scan"
        " for reconstruction and inspect the fitted poses."
    ),
    op.Evaluate: (
        "Compare reconstruction or alignment settings on held-out projections. Use the "
        "scores alongside slice comparisons to choose settings."
    ),
    op.Simulate: (
        "Generate a synthetic scan and its ground-truth phantom to check reconstruction "
        "accuracy or test acquisition artefacts."
    ),
    op.Project: (
        "Forward-project a volume through the source scan geometry. Compare the predicted "
        "detector images with measured projections."
    ),
    op.Backproject: (
        "Backproject detector values onto a volume grid with the exact transpose of the "
        "forward projector. Use it to map projection residuals into object space."
    ),
    op.Save: (
        "Save a TomoJAX dataset with its provenance manifest so it can be reopened. Set "
        "include_scan to keep the source scan with a reconstruction."
    ),
    op.Export: (
        "Export a volume as a TIFF slice stack or a raw file with sidecars for use in other "
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
        fields.append(entry if field.required else (*entry, default))
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
                    "Preview a slice, sweep the centre or compare corrections. "
                    "A region preview returns a handle for reconstruction."
                )
            },
        ),
    )
    return models
