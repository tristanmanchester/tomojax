"""Keep the worked Python routes reachable in an installed server."""

import json

import numpy as np
import pytest

from tests.test_mcp_python import python
from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface


def guide(c, topic):
    row = c.rpc("resources/read", {"uri": "tomojax://guide/" + topic})
    return row["result"]["contents"][0]["text"]


def example(text, heading):
    return text.split("## " + heading + "\n", 1)[1].split("```python\n", 1)[1].split("\n```", 1)[0]


def test_workflow_prompts_and_live_resources(tmp_path):
    topics = {
        "reconstruct-scan",
        "lab-ct-start-to-finish",
        "find-the-centre",
        "check-a-reconstruction",
    }
    with connection(tmp_path) as c:
        listing = c.rpc("prompts/list", {})["result"]["prompts"]
        assert {p["name"] for p in listing} == topics
        for topic in topics:
            reply = c.rpc("prompts/get", {"name": topic, "arguments": {"source": "scan.nxs"}})
            (message,) = reply["result"]["messages"]
            assert message["role"] == "user"
            assert guide(c, topic) in message["content"]["text"]
            assert "scan.nxs" in message["content"]["text"]
        templates = c.rpc("resources/templates/list", {})["result"]["resourceTemplates"]
        assert {t["uriTemplate"] for t in templates} == {
            "tomojax://guide/{topic}",
            "tomojax://handle/{id}",
            "tomojax://describe/{operation}",
            "tomojax://result/{call}/{n}",
        }
        ref = c.call("simulate", size=8)["structuredContent"]["handle"]

        def read(uri):
            return c.rpc("resources/read", {"uri": uri})

        uri = f"tomojax://handle/{ref}"
        handle = json.loads(read(uri)["result"]["contents"][0]["text"])
        assert handle["kind"] == "simulation" and handle["shape"] == [8, 8, 8]
        assert (
            handle["manifest"]
            == c.call("status", detail="full")["structuredContent"]["handles"][ref]["manifest"]
        )
        info = json.loads(read("tomojax://describe/reconstruct")["result"]["contents"][0]["text"])
        assert info == c.call("describe", operation="reconstruct")["structuredContent"]
        c.call("control", action="close", target=ref)
        assert "Unknown reference" in read(uri)["error"]["message"]


def test_worked_python_routes(tmp_path):
    with connection(tmp_path) as c:
        text = guide(c, "python")
        for topic in ("solvers", "geometry", "calibration"):
            assert guide(c, topic) in text
        simulated = c.call("simulate", size=12, n_views=20)["structuredContent"]
        python(c, f"sim=handles[{simulated['handle']!r}]\nscan=sim.scan")
        direct = python(c, example(text, "Direct solver"))["value"]
        custom = python(c, example(text, "Custom geometry"))["value"]
        projected = c.call("project", source=custom, volume=direct)["structuredContent"]["handle"]
        assert python(c, f"np.allclose(handles[{projected!r}],tj.project(geometry,volume))")[
            "value"
        ]
        assert not python(
            c, f"np.allclose(handles[{projected!r}],tj.project(scan.geometry,volume))"
        )["value"]
        assert python(c, example(text, "Memory-mapped reconstruction"))["value"] > 0
        assert np.load(tmp_path / "reconstruction.npy").shape == (12, 12, 12)
        assert not c.call("save", source=simulated["members"]["scan"], path="scan.nxs")["isError"]
        metadata = python(c, example(text, "Acquisition metadata"))["value"]
        assert metadata["views"] == 20 and metadata["shape"] == [12, 12, 12]


@pytest.mark.gpu
def test_torch_guide(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("tomojax.torch requires CUDA")
    with connection(tmp_path) as c:
        python(c, "scan=tj.simulate(size=8,n_views=11).scan")
        assert python(c, example(guide(c, "python"), "Torch"))["value"]
