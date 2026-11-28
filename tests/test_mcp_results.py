"""Check result budgets, request-scoped progress and wire coverage independently."""

from io import BytesIO
import json
from typing import get_args

import msgspec
import numpy as np
from PIL import Image
import pytest

from tests.mcp_client import connection
from tomojax.cli import operations as op

pytestmark = pytest.mark.surface


def test_each_wire_field_has_a_tool_owner(tmp_path):
    with connection(tmp_path) as c:
        schemas = {
            tool["name"]: tool["inputSchema"] for tool in c.rpc("tools/list", {})["result"]["tools"]
        }
    interactive = {
        op.Plane: "view",
        op.CenterSweep: "M3 preview",
        op.Slice: "M3 preview",
        op.CorrectionPreview: "M3 preview",
        op.RegionPreview: "M3 preview",
    }
    for cls in (op.Slice, op.CenterSweep, op.CorrectionPreview, op.RegionPreview):
        assert (
            set(cls.__struct_fields__)
            <= schemas["preview"]["$defs"][cls.__name__]["properties"].keys()
        )
    for cls in (op.Read, op.Statistics):
        assert (
            set(cls.__struct_fields__) - {"metadata_only"} <= schemas["view"]["properties"].keys()
        )
    assert set(interactive) == set(get_args(op.Interactive.__value__))
    assert set(get_args(op.ExecutionInput.__value__)) == {
        op.Open,
        op.Python,
        op.Inspect,
        op.Plan,
        op.Validate,
        op.Run,
        op.Interactive,
        op.Read,
        op.Statistics,
    }
    assert (
        set(op.Python.__struct_fields__) - {"aliases", "ledger"}
        == schemas["python"]["properties"].keys()
    )
    for candidate in (op.ReconstructSettings, op.AlignSettings):
        fields = schemas["evaluate"]["$defs"][candidate.__name__]["properties"]
        assert set(candidate.__struct_fields__) <= fields.keys()
    assert op.Run.__struct_fields__ == ("operation", "previews")  # previews owned by M3 panels
    # Studio consumes Inspect.previews during the job. MCP's preview option
    # attaches a picture to the completed tool result.
    assert set(op.Inspect.__struct_fields__) == set(schemas["inspect"]["required"]) | {"previews"}
    for wrapper in (op.Plan, op.Validate):
        assert wrapper.__struct_fields__ == ("operation",)  # dry_run
    for cls in (op.Open, *get_args(op.Workflow.__value__)):
        name = cls.__struct_config__.tag
        schema = schemas.get(name) or schemas["edit_scan"]["$defs"][cls.__name__]
        assert set(cls.__struct_fields__) <= schema["properties"].keys()
        assert "dry_run" in schemas.get(name, schemas["edit_scan"])["properties"]
        for key in ("config", "corrections"):
            if key in cls.__struct_fields__:
                wire = msgspec.json.schema(cls)["$defs"]
                for nested in wire:
                    if nested.endswith("Config") and cls is not op.Evaluate:
                        assert '"type": "object"' in json.dumps(schema["properties"]["config"])
                        assert nested not in schema.get("$defs", {})  # keys belong to describe
                    elif nested in {step.__name__ for step in get_args(op.Step.__value__)}:
                        assert (
                            wire[nested]["properties"].keys()
                            == schema["$defs"][nested]["properties"].keys()
                        )


def test_comparison_panels_use_one_pixel_mapping_and_label_each_plane():
    from tomojax.cli.images import Panel, limits, montage

    panels = [
        Panel("Before", np.array([[0.0, 1.0], [0.0, 1.0]]), ("z[1] = 2 mm",), "count"),
        Panel("After", np.array([[1.0, 2.0], [1.0, 2.0]]), ("z[1] = 2 mm",), "1"),
    ]
    lo, hi = limits(panels)
    assert (lo, hi) == (0.0, 2.0)
    picture = np.asarray(Image.open(BytesIO(montage(panels, [(lo, hi)] * len(panels)))))
    assert max(picture.shape) <= 1024 and picture.dtype == np.uint8
    assert picture[80, 220] == picture[80, 328 + 100] == 127
    assert picture[80, 100] == 0 and picture[80, 328 + 220] == 255
    assert (picture[240:, :328] > 128).any() and (picture[240:, 328:] > 128).any()


def test_roundoff_in_panel_labels():
    from tomojax.cli.images import label_number

    assert label_number(-3.5527e-15) == "0"
    assert label_number(-0.000025) == "-2.5e-05"
