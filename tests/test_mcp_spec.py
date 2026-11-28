"""Tool annotations and declared result schemas match what the tools do and return."""

from jsonschema import Draft202012Validator
import pytest

from tests.test_mcp_features import feature
from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface

WRITES = {"save", "export", "reconstruct_slabs", "control", "python", "cancel_job"}
READS = {"status", "inspect", "describe", "view", "preview", "jobs"}


def test_annotations_and_results_conform(tmp_path):
    with connection(tmp_path) as c:
        listing = {t["name"]: t for t in c.rpc("tools/list", {})["result"]["tools"]}
        for name, tool in listing.items():
            hints = tool["annotations"]
            assert hints["openWorldHint"] is False, name
            assert hints["readOnlyHint"] is (name in READS), name
            if name not in READS:
                # Unset, the spec reads destructive; only tools that replace or discard say so.
                assert hints["destructiveHint"] is (name in WRITES), name
            assert tool["outputSchema"]["type"] == "object" and tool["outputSchema"]["properties"]

        def conforms(name, reply):
            assert not reply["isError"], reply["content"][0]["text"]
            Draft202012Validator(listing[name]["outputSchema"]).validate(reply["structuredContent"])
            return reply["structuredContent"]

        sim = conforms("simulate", c.call("simulate", size=12, n_views=10))
        volume = sim["members"]["phantom"]
        conforms("status", c.call("status"))
        conforms("describe", c.call("describe", operation="reconstruct", method="fista"))
        made = feature(c, source=volume, recipe={"type": "sphere", "at": [6, 6, 6], "radius": 3})
        Draft202012Validator(listing["feature"]["outputSchema"]).validate(made)
        marks = [{"type": "point", "at": [6, 6, 6]}]
        conforms("view", c.call("view", source=volume, marks=marks, features=[made["handle"]]))
        conforms("view", c.call("view", source=volume, mode="statistics"))
        conforms(
            "view", c.call("view", source=volume, mode="profile", roi=[[6, 7], [6, 7], [0, 12]])
        )
        conforms("python", c.call("python", code="np.arange(40)"))
        conforms(
            "preview",
            c.call(
                "preview",
                operation={"type": "slice", "source": sim["members"]["scan"], "axis": "z"}
                | {"index": 6, "method": "fbp"},
            ),
        )
        conforms("control", c.call("control", action="close", target=made["handle"]))


def test_describe_replies_stand_on_their_own(tmp_path):
    import json

    with connection(tmp_path) as c:
        for operation in ("evaluate", "align", "correct", "feature"):
            reply = c.call("describe", operation=operation)
            assert not reply["isError"], reply["content"][0]["text"]
            text = json.dumps(reply["structuredContent"])
            assert "$ref" not in text, operation
        candidates = json.dumps(c.call("describe", operation="evaluate")["structuredContent"])
        assert "tv_weight" in candidates and "image_iterations" in candidates
