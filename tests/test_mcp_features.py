"""Feature handles: masks made from shapes, seeds and bands, outlined and reused."""

import pytest

from tests.test_mcp_marks import images
from tests.test_mcp_python import python
from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface

CYAN = (0, 229, 255)
BLOCK = (slice(8, 14), slice(10, 18), slice(12, 22))  # value 1 plus noise, background 0


def feature(c, **arguments):
    reply = c.rpc("tools/call", {"name": "feature", "arguments": arguments})["result"]
    assert not reply["isError"], reply
    return reply["structuredContent"]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    with connection(tmp_path_factory.mktemp("features")) as c:
        made = python(
            c,
            """sim = tj.simulate(size=24, n_views=12)
rng = np.random.default_rng(3)
volume = rng.normal(0, 0.05, (24, 24, 24)).astype(np.float32)
volume[8:14, 10:18, 12:22] += 1
other = volume[::-1].copy()
small = tj.simulate(size=20, n_views=12).scan.geometry
[register(tj.Reconstruction(volume, sim.scan.geometry, "fbp", {})),
 register(tj.Reconstruction(other, sim.scan.geometry, "fbp", {})),
 register(tj.Reconstruction(volume[:20, :20, :20], small, "fbp", {}))]""",
        )["value"]
        yield c, made


@pytest.mark.parametrize(
    ("recipe", "voxels"),
    [
        ({"type": "box", "corners": [[2, 3, 4], [4, 6, 8]]}, 3 * 4 * 5),
        # Lattice points within radius 2: 1 + 6 + 12 + 8 + 6.
        ({"type": "sphere", "at": [12, 12, 12], "radius": 2}, 33),
        # A five-pixel disc across z over four planes.
        ({"type": "cylinder", "at": [0, 12, 12], "radius": 1, "axis": "z", "extent": [2, 5]}, 20),
        # A 4 x 4 square across z (edges included) over two planes.
        (
            {"type": "prism", "points": [[0, 2, 2], [0, 2, 5], [0, 5, 5], [0, 5, 2]], "axis": "z"}
            | {"extent": [0, 1]},
            32,
        ),
    ],
)
def test_shapes_cover_the_voxels_they_describe(session, recipe, voxels):
    c, (volume, _, _) = session
    made = feature(c, source=volume, recipe=recipe)
    assert made["kind"] == "feature" and made["voxels"] == voxels
    assert made["name"] == recipe["type"] and made["members"] == {"mask": made["handle"] + ".mask"}
    count = python(c, f"int(handles[{made['handle']!r}].mask.sum())")["value"]
    assert count == voxels


def test_a_seed_grows_into_the_block_and_a_band_surrounds_it(session):
    c, (volume, other, _) = session
    block = feature(
        c, source=volume, recipe={"type": "seeded", "seeds": [[10, 14, 16]]}, name="block"
    )
    assert block["name"] == "block" and block["bounds"] == [[8, 14], [10, 18], [12, 22]]
    # A three-sigma band leaves out about 0.3% of the block's noisy voxels.
    assert 0.99 * 480 <= block["voxels"] <= 480
    assert abs(block["mean"] - 1) < 0.02 and 0.03 < block["std"] < 0.07
    assert 0.03 < block["recipe"]["noise"] < 0.07  # the noise the volume was made with
    # Bands around an exact box, so the counts follow from its faces.
    box = feature(c, source=volume, recipe={"type": "box", "corners": [[8, 10, 12], [13, 17, 21]]})
    shell = feature(c, source=volume, recipe={"type": "band", "feature": box["handle"], "outer": 1})
    # Voxels one step from a face: 2 (8 x 10) + 2 (6 x 10) + 2 (6 x 8).
    assert shell["voxels"] == 2 * (80 + 60 + 48) and abs(shell["mean"]) < 0.02
    rim = feature(
        c, source=volume, recipe={"type": "band", "feature": box["handle"], "inner": -1, "outer": 0}
    )
    assert rim["voxels"] == 6 * 8 * 10 - 4 * 6 * 8 and rim["recipe"]["feature"] == "box"
    deep = feature(
        c, source=volume, recipe={"type": "band", "feature": box["handle"], "inner": -2, "outer": 0}
    )
    assert deep["voxels"] == 6 * 8 * 10 - 2 * 4 * 6

    # Outline the block on two planes of the volume, and on a second volume on the same grid.
    for source, axis in ((volume, "z"), (volume, "x"), (other, "z")):
        index = {"z": 11, "x": 16}[axis] if source == volume else 12
        reply = c.call("view", source=source, axis=axis, index=index, features=[block["handle"]])
        assert not reply["isError"], reply
        assert reply["structuredContent"]["features"] == [
            {"handle": block["handle"], "colour": "cyan"}
        ]
        cyan = (images(reply)[0] == CYAN).all(axis=2)
        assert cyan.sum() > 40, (source, axis)
    off = c.call("view", source=volume, axis="z", index=20, features=[block["handle"]])
    assert not (images(off)[0] == CYAN).all(axis=2).any()  # the plane misses the block


def test_python_round_trip_and_grid_checks(session):
    c, (volume, other, small) = session
    box = feature(c, source=volume, recipe={"type": "box", "corners": [[8, 10, 12], [13, 17, 21]]})
    out = python(
        c,
        f"""mask = handles[{box["handle"]!r}].mask
again = register_feature(mask, like={other!r}, name="copy")
report = view({volume!r}, axis="y", index=14, features=[{box["handle"]!r}],
              marks=[{{"type": "point", "at": [10, 14, 16]}}])
# A feature registered in this call can be shown before the call publishes it.
same_call = view({other!r}, axis="y", index=14, features=[again])
[again, report]""",
    )
    again, report = out["value"]
    assert again in out["handles"] and out["handles"][again]["voxels"] == 6 * 8 * 10
    assert out["handles"][again]["name"] == "copy"
    assert report[0]["panels"][0]["drawn"] and abs(report[0]["panels"][0]["value"] - 1) < 0.2
    same = python(
        c, f"bool(np.array_equal(handles[{again!r}].mask, handles[{box['handle']!r}].mask))"
    )
    assert same["value"] is True
    reply = c.call("view", source=small, features=[box["handle"]])
    assert reply["isError"] and "grid" in reply["content"][0]["text"]
    not_feature = c.call("view", source=volume, features=[other])
    assert not_feature["isError"] and "not a feature" in not_feature["content"][0]["text"]
    wrong = c.rpc(
        "tools/call",
        {
            "name": "feature",
            "arguments": {
                "source": volume,
                "recipe": {"type": "sphere", "at": [1, 2], "radius": 1},
            },
        },
    )["result"]
    assert wrong["isError"] and "3 array indices (z, y, x)" in wrong["content"][0]["text"]
