"""Tool input schemas: built from the operation records, and rendered short for errors."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from tomojax.cli.mcp.supervisor import Object
    from tomojax.wire.arrays import Record

DRY_HELP = {
    "Evaluate": "Validate settings and report split sizes before data-dependent view rejection.",
    "Open": "Validate the file and report the object kind, path and geometry.",
    "Simulate": "Report the resolved view count, phantom shape and projection shape.",
    "Align": "Resolve the fit method, working grids, budgets and placement.",
    "Reconstruct": "Resolve the reconstruction method, grid, iterations and placement.",
    "Save": "Report dataset and manifest paths and whether the destination exists.",
    "Export": "Report the format, files, axes, value mapping and resource estimates.",
}


def schema(cls: type[Record], definitions: Object) -> Object:
    """Inline the root and enrich matching fields from the worker's parameter document."""
    generated = msgspec.json.schema(cls)
    nested = generated["$defs"]
    root = nested.pop(cls.__name__)
    for name, struct in [(cls.__name__, root), *nested.items()]:
        properties = definitions.get(name, {}).get("properties", {})
        for key, entry in struct["properties"].items():
            entry.update(
                {
                    k: v
                    for k, v in properties.get(key, {}).items()
                    if k in {"description", "x-unit", "x-choices", "x-formats"}
                }
            )
    for field, help_text in {
        "config": "Expert settings. Use describe for the method's keys.",
        "dry_run": DRY_HELP.get(
            cls.__name__,
            f"Validate the proposed {cls.__name__} operation and report its parameters.",
        ),
        "detail": "Use full for the worker info and provenance when available.",
    }.items():
        if field in root["properties"]:
            root["properties"][field]["description"] = help_text
    return {**root, **({"$defs": nested} if nested else {})}


def outline(node: Object, defs: Object, depth: int = 0) -> str:
    """A compact rendering of a JSON schema, for an error that says what to send."""
    node = defs[node["$ref"].rsplit("/", 1)[1]] if "$ref" in node else node
    kind = node.get("type")
    if "enum" in node:
        text = " | ".join(json.dumps(v) for v in node["enum"])
    elif "anyOf" in node:
        options = [o for o in node["anyOf"] if o.get("type") != "null"]
        text = " | ".join(outline(o, defs, depth) for o in options)
    elif kind == "object" and "properties" in node and depth > 1:
        text = "{...}"
    elif kind == "object" and "properties" in node:
        required = node.get("required", [])
        fields = [
            f"{k}{'' if k in required else '?'}: {outline(v, defs, depth + 1)}"
            for k, v in node["properties"].items()
        ]
        text = "{" + ", ".join(fields) + "}"
    elif kind == "array" and "prefixItems" in node:
        text = "[" + ", ".join(outline(i, defs, depth) for i in node["prefixItems"]) + "]"
    elif kind == "array":
        text = "[" + outline(node.get("items", {}), defs, depth) + ", ...]"
    else:
        text = str(kind or "any")
    return text


def expected(spec: Object, field: str, given: object) -> str | None:
    """The shape of one top-level argument.

    A discriminated union narrows to the variant the caller named, if any.
    """
    node = spec["properties"].get(field)
    if node is None:
        return None
    defs = spec.get("$defs", {})
    tag = node.get("discriminator", {})
    if tag and isinstance(given, dict):
        ref = tag["mapping"].get(given.get(tag["propertyName"]))
        node = {"$ref": ref} if ref else node
    return f"{field} as {outline(node, defs)}"
