"""Generate linked Draft 2020-12 schemas from the exact ingress and output types."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, get_args

import msgspec

from tomojax.cli.envelope import Credit, Request, Response
from tomojax.cli.operations import ExecutionInput
from tomojax.cli.products import Output
from tomojax.wire.arrays import Array, PoseColumn
from tomojax.wire.progress import ProgressRecord

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs" / "protocol"
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
        ("worker", (Credit, Request, Response)),
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
        if name == "operations":
            from tomojax.io.api import format_descriptions

            defs["Export"]["properties"]["path"]["x-formats"] = format_descriptions()
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


class _Options(argparse.Namespace):
    check: bool = False


def main() -> int:
    """Write generated schemas or verify the checked-in copies."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--check", action="store_true", help="Check for drift without writing.")
    check = parser.parse_args(namespace=_Options()).check

    drifted: list[str] = []
    schemas = documents()
    for filename, schema in schemas.items():
        path = DIRECTORY / filename
        content = json.dumps(schema, indent=2) + "\n"
        if not check:
            _ = path.write_text(content, encoding="utf-8")
        elif not path.is_file() or path.read_bytes() != content.encode("utf-8"):
            drifted.append(path.relative_to(ROOT).as_posix())

    if check:
        obsolete = sorted(
            path.relative_to(ROOT).as_posix()
            for path in DIRECTORY.glob("*.schema.json")
            if path.name not in schemas
        )
        if obsolete:
            print(f"Remove obsolete schemas: {', '.join(obsolete)}", file=sys.stderr)
        if drifted:
            print(f"Generated files are out of date: {', '.join(drifted)}", file=sys.stderr)
        if obsolete or drifted:
            print("Regenerate with: uv run python tools/build_wire_schemas.py", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
