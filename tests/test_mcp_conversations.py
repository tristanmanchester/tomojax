"""Replay complete conversations against the installed stdio launcher in both MCP eras."""

import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image
import pytest

from tests.mcp_client import connection
import tomojax as tj

pytestmark = pytest.mark.surface


def conversation(directory, *, legacy, executable=None):
    """Return decoded replies in order, following this session's actual IDs."""
    transcript = []
    names = {}

    def record(value):
        def visit(item):
            if isinstance(item, dict):
                if "io.modelcontextprotocol/serverInfo" in item:
                    identity = item["io.modelcontextprotocol/serverInfo"]
                    assert identity == {"name": "tomojax", "version": tj.__version__}
                    item = {
                        **item,
                        "io.modelcontextprotocol/serverInfo": {**identity, "version": "VERSION"},
                    }
                if item.get("type") == "image":
                    with Image.open(BytesIO(base64.b64decode(item["data"]))) as image:
                        # Compare every decoded pixel, independently of the PNG compressor.
                        item = {
                            **item,
                            "data": [image.mode, image.size, sha256(image.tobytes()).hexdigest()],
                        }
                return {visit(k): visit(v) for k, v in item.items()}
            if isinstance(item, list):
                return [visit(v) for v in item]
            if isinstance(item, str):
                for actual, placeholder in names.items():
                    item = item.replace(actual, placeholder)
            return item

        transcript.append(visit(value))
        return value

    with connection(directory, legacy=legacy, executable=executable) as c:
        registered = c.call("python", code="register(np.arange(12).reshape(3, 4))")
        assert not registered["isError"], registered
        handle = registered["structuredContent"]["value"]
        assert re.fullmatch(r"@array-[a-f0-9]{8}", handle)
        names[handle] = "@array-HANDLE"
        record(registered)
        mark = {"type": "point", "at": [1, 2]}
        code = (
            f"reports = show(handles[{handle!r}], marks=[{mark!r}])\n"
            "reports[0]['panels'][0]['value'] + 1"
        )
        reply = record(c.call("python", code=code))
        assert reply["structuredContent"]["value"] == 7
        assert [part["type"] for part in reply["content"]] == ["text", "image"]
        record(c.call("control", action="close", target=handle))
        assert record(c.call("control", action="close", target=handle))["isError"]
        result = c.call("python", code="print('q' * 18000)", tasks=True)
        if not legacy:
            task = result["taskId"]
            names[task] = "TASK"
            for key in ("createdAt", "lastUpdatedAt"):
                names[result[key]] = "TIMESTAMP"
            record(result)
            record(c.rpc("tasks/get", {"taskId": task})["error"])
            terminal = c.terminal(task)
            names[terminal["lastUpdatedAt"]] = "TIMESTAMP"
            result = terminal["result"]
        page = result["structuredContent"]["continuation"]
        names[page.rsplit("/", 2)[1]] = "PAGE"
        record(result if legacy else terminal)
        while page:
            reply = record(c.rpc("resources/read", {"uri": page})["result"])
            page = json.loads(reply["contents"][0]["text"])["next"]
        record(c.rpc("resources/read", {"uri": "tomojax://result/absent/0"})["error"])
    return transcript


@pytest.mark.parametrize("legacy", [True, False], ids=["2025-11-25", "2026-07-28"])
def test_recorded_conversation(tmp_path, legacy):
    replies = conversation(tmp_path, legacy=legacy)
    fingerprints = [
        sha256(json.dumps(r, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        for r in replies
    ]
    fixture = Path(__file__).with_name("mcp_conversations.json")
    assert fingerprints == json.loads(fixture.read_text())["legacy" if legacy else "modern"]
