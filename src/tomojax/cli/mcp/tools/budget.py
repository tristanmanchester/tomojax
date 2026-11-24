"""Budget all model-visible fields together and retain complete replies for paging."""

from __future__ import annotations

import base64
import json
from math import ceil
import struct
from typing import TYPE_CHECKING, Any

from mcp_types import CallToolResult, ImageContent, TextContent
import msgspec

if TYPE_CHECKING:
    from tomojax.cli.mcp.parts import Parts

LIMIT = 20_000
PNG_BYTES = 512 * 1024


def cost(reply: CallToolResult) -> int:
    """Bound text tokens by UTF-8 bytes, charging PNGs separately by pixels and bytes."""
    payload = reply.model_dump(mode="json", by_alias=True, exclude={"content"})
    contents = []
    pixels = encoded = 0
    for item in reply.content:
        if isinstance(item, ImageContent):
            header = base64.b64decode(item.data[:32])
            width, height = struct.unpack(">II", header[16:24])
            encoded += len(item.data)
            if max(width, height) > 1024 or encoded > 4 * ceil(PNG_BYTES / 3):
                return LIMIT + 1
            pixels += ceil(width * height / 750)
            contents.append(item.model_dump(mode="json", by_alias=True, exclude={"data"}))
        else:
            contents.append(item.model_dump(mode="json", by_alias=True))
    payload["content"] = contents
    # Leave room for the SDK's JSON-RPC envelope and server identification.
    return len(msgspec.json.encode(payload)) + pixels + 256


def bounded(reply: CallToolResult, parts: Parts) -> CallToolResult:
    """Retain high-priority fields and link to a lossless copy of the original result."""
    if cost(reply) <= LIMIT:
        return reply
    raw = json.dumps(
        reply.model_dump(mode="json", by_alias=True, exclude_none=True), separators=(",", ":")
    )
    uri = parts.add(raw)
    message = (
        f"Full result continues at {uri}"
        if uri
        else "Result exceeds the paging limit. Repeat the call with a smaller result."
    )
    data: dict[str, Any] = {"continuation": uri} if uri else {"message": message}
    fields = reply.structured_content or {}
    headline = " ".join(
        str(fields[k])[:160] for k in ("handle", "kind", "method", "next") if fields.get(k)
    )
    clipped = CallToolResult(
        content=[TextContent(type="text", text=f"{headline}\n{message}".strip())],
        structured_content=data,
        is_error=reply.is_error,
    )
    retain_fields(clipped, reply.structured_content or {})
    # Fill remaining room with the human summary, file links, then panels.
    for item in sorted(reply.content, key=lambda c: isinstance(c, ImageContent)):
        clipped.content.insert(len(clipped.content) - 1, item)
        if cost(clipped) > LIMIT:
            clipped.content.pop(-2)
    return clipped


def retain_fields(clipped: CallToolResult, fields: dict[str, Any]) -> None:
    """Keep identifiers, affected files and next steps ahead of verbose summaries."""
    data = clipped.structured_content
    assert data is not None
    priority = (
        "handle",
        "members",
        "handles",
        "paths",
        "next",
        "kind",
        "shape",
        "unit",
        "method",
        "message",
    )
    for key in dict.fromkeys((*priority, *fields)):
        if key not in fields:
            continue
        value = fields[key]
        data[key] = value
        if cost(clipped) <= LIMIT:
            continue
        del data[key]
        if key not in priority or not isinstance(value, str | list | dict):
            continue
        # Find a prefix that fits, including the escaping and framing of the field.
        pairs = list(value.items()) if isinstance(value, dict) else None
        lo, hi = 0, len(value)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            data[key] = dict(pairs[:mid]) if pairs is not None else value[:mid]
            if cost(clipped) <= LIMIT:
                lo = mid
            else:
                hi = mid - 1
        if lo:
            data[key] = dict(pairs[:lo]) if pairs is not None else value[:lo]
        else:
            data.pop(key, None)
