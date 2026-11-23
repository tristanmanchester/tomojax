"""Generate linked Draft 2020-12 schemas from the exact ingress and output types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import msgspec

from tomojax.cli.envelope import Request, Response
from tomojax.cli.operations import ExecutionInput
from tomojax.cli.products import Output
from tomojax.wire.arrays import Array, PoseColumn
from tomojax.wire.progress import ProgressRecord

DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "protocol"
BASE = "https://tomojax.dev/protocol/"


def documents() -> dict[str, dict[str, Any]]:
    """Define each shared model in its owner's document and link users to it."""
    documents = {}
    owners = {}
    for name, types in (
        ("arrays", (Array,)),
        ("progress", (ProgressRecord,)),
        ("operations", (ExecutionInput,)),
        ("products", (Output,)),
        ("worker", (Request, Response)),
    ):
        filename = f"{name}.schema.json"
        roots, defs = msgspec.json.schema_components(types)
        if name == "arrays":
            # Cross-field constraints on a pose descriptor supplement the field types.
            columns = [
                {
                    "properties": {
                        "name": {"const": column},
                        **({"unit": {"const": "degree"}} if index < 3 else {}),
                    }
                }
                for index, column in enumerate(get_args(PoseColumn))
            ]
            defs["Array"]["if"] = {"properties": {"domain": {"const": "poses"}}}
            defs["Array"]["then"] = {
                "required": ["columns"],
                "properties": {
                    "columns": {
                        "minItems": len(columns),
                        "maxItems": len(columns),
                        "prefixItems": columns,
                    },
                    "shape": {
                        "minItems": 2,
                        "maxItems": 2,
                        "prefixItems": [{}, {"const": len(columns)}],
                    },
                },
            }
        own = {key: value for key, value in defs.items() if key not in owners}
        document = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": BASE + filename,
            **(roots[0] if len(roots) == 1 else {"oneOf": roots}),
            "$defs": own,
        }
        # Discriminator mappings as well as $ref fields contain component pointers.
        encoded = json.dumps(document)
        for key, owner in owners.items():
            encoded = encoded.replace(f'"#/$defs/{key}"', f'"{BASE}{owner}#/$defs/{key}"')
        documents[filename] = json.loads(encoded)
        owners.update(dict.fromkeys(own, filename))
    return documents


if __name__ == "__main__":
    for filename, schema in documents().items():
        _ = (DIRECTORY / filename).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
