"""Marks drawn by view land on the voxel they name, in every plane, ROI and stride."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from tests.test_mcp_python import python
from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface

# The documented mark colours.
PALETTE = {
    "yellow": (255, 214, 0),
    "cyan": (0, 229, 255),
    "magenta": (255, 64, 245),
    "orange": (255, 140, 0),
    "green": (110, 255, 60),
    "red": (255, 48, 48),
}
VOXEL = (10, 12, 26)  # bright voxel, value 5
BLOCK = ((4, 8), (20, 26), (6, 14))  # half-open block, value 2


def images(reply):
    return [
        np.asarray(Image.open(BytesIO(base64.b64decode(item["data"]))).convert("RGB"))
        for item in reply["content"]
        if item["type"] == "image"
    ]


def white_centre(picture):
    """Centre of the saturated pixels: the only full-scale sample in the plane."""
    rows, cols = np.nonzero((picture == 255).all(axis=2))
    assert rows.size, "no saturated sample in the rendered plane"
    return (cols.min() + cols.max() + 1) / 2, (rows.min() + rows.max() + 1) / 2


def coloured_near(picture, pixel, colour, reach=7):
    x, y = (round(v) for v in pixel)
    patch = picture[max(y - reach, 0) : y + reach, max(x - reach, 0) : x + reach]
    # Text is antialiased, so accept close shades of the mark's colour.
    return (np.abs(patch.astype(int) - colour).max(axis=2) <= 40).any()


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    with connection(tmp_path_factory.mktemp("marks")) as c:
        made = python(
            c,
            f"""sim = tj.simulate(size=32, n_views=12)
volume = np.zeros((32, 32, 32), np.float32)
volume[{VOXEL}] = 5
volume[{BLOCK[0][0]}:{BLOCK[0][1]}, {BLOCK[1][0]}:{BLOCK[1][1]}, {BLOCK[2][0]}:{BLOCK[2][1]}] = 2
projections = np.zeros(sim.scan.projections.shape, np.float32)
projections[5, 9, 20] = 3
[register(tj.Reconstruction(volume, sim.scan.geometry, "fbp", {{}})),
 register(tj.Scan(projections, sim.scan.geometry)), list(projections.shape)]""",
        )["value"]
        yield c, made


@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_a_point_lands_on_its_voxel_in_each_plane_with_roi_and_stride(session, axis):
    c, (volume, _, _) = session
    d = "zyx".index(axis)
    window = dict(
        source=volume, axis=axis, index=VOXEL[d], roi=[[2, 30], [4, 32], [0, 32]], stride=[2] * 3
    )
    plain = c.call("view", ticks=True, **window)
    assert not plain["isError"], plain
    expected = white_centre(images(plain)[0])

    marks = [
        {"type": "point", "at": list(VOXEL), "label": "voxel"},
        {"type": "point", "at": [v + (4 if i == d else 0) for i, v in enumerate(VOXEL)]},
        {"type": "point", "at": [v + (0 if i == d else 40) for i, v in enumerate(VOXEL)]},
    ]
    reply = c.call("view", marks=marks, **window)
    assert not reply["isError"], reply
    on, off, outside = (m["panels"][0] for m in reply["structuredContent"]["marks"])
    assert on["drawn"] and on["value"] == 5
    assert abs(on["pixel"][0] - expected[0]) <= 1.5 and abs(on["pixel"][1] - expected[1]) <= 1.5
    assert coloured_near(images(reply)[0], on["pixel"], PALETTE["yellow"])
    assert off == {"panel": 0, "drawn": False, "reason": "off_plane", "distance": 4}
    assert outside == {"panel": 0, "drawn": False, "reason": "outside_roi"}


def test_shapes_report_means_and_follow_into_orthogonal_planes(session):
    c, (volume, _, _) = session
    (z0, z1), (y0, y1), (x0, x1) = BLOCK
    centre = [(z0 + z1 - 1) / 2, (y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2]
    marks = [
        {"type": "box", "corners": [[z0, y0, x0], [z1 - 1, y1 - 1, x1 - 1]], "colour": "magenta"},
        {"type": "circle", "at": centre, "radius": 1.5, "colour": "cyan"},
        {"type": "arrow", "from": [5, 2, 2], "to": [5, 20, 6], "colour": "orange"},
        {"type": "polygon", "points": [[5, 20, 6], [5, 25, 6], [5, 25, 13]], "colour": "green"},
        {"type": "text", "at": [5, 28, 28], "label": "note", "colour": "red"},
    ]
    reply = c.call("view", source=volume, axis="z", index=5, marks=marks)
    assert not reply["isError"], reply
    picture = images(reply)[0]
    found = [m["panels"][0] for m in reply["structuredContent"]["marks"]]
    assert all(f["drawn"] for f in found), found
    for f, spec in zip(found, marks, strict=True):
        assert coloured_near(picture, f["pixel"], PALETTE[spec["colour"]], reach=70), spec
    box, circle, arrow, polygon, _ = found
    assert box["mean"] == circle["mean"] == polygon["mean"] == 2
    assert arrow["value"] == 2  # the head sits on the block's corner
    # The same box and sphere through a vertical plane across the block, and a plane past it.
    vertical = c.call("view", source=volume, axis="x", index=round(centre[2]), marks=marks[:2])
    box, circle = (m["panels"][0] for m in vertical["structuredContent"]["marks"])
    assert box["drawn"] and box["mean"] == 2 and circle["drawn"] and circle["mean"] == 2
    past = c.call("view", source=volume, axis="y", index=16, marks=marks[:1])
    assert past["structuredContent"]["marks"][0]["panels"][0] == {
        "panel": 0,
        "drawn": False,
        "reason": "off_plane",
        "distance": 4,
    }


@pytest.mark.parametrize(("mode", "extra"), [("projection", {"index": 5}), ("sinogram", {})])
def test_marks_on_projection_planes(session, mode, extra):
    c, (_, scan, shape) = session
    window = dict(source=scan, mode=mode, roi=[[0, shape[0]], [1, shape[1]], [0, shape[2]]])
    if mode == "sinogram":
        window["index"] = 9
    window.update(extra)
    expected = white_centre(images(c.call("view", ticks=True, **window))[0])
    reply = c.call("view", marks=[{"type": "point", "at": [5, 9, 20]}], **window)
    (found,) = reply["structuredContent"]["marks"][0]["panels"]
    assert found["drawn"] and found["value"] == 3
    assert abs(found["pixel"][0] - expected[0]) <= 1.5
    assert abs(found["pixel"][1] - expected[1]) <= 1.5


def test_marks_reject_wrong_coordinates_and_non_plane_modes(session):
    c, (volume, _, _) = session
    wrong = c.call("view", source=volume, marks=[{"type": "point", "at": [1, 2]}])
    assert wrong["isError"] and "one per source axis (z, y, x)" in wrong["content"][0]["text"]
    stats = c.call(
        "view", source=volume, mode="statistics", marks=[{"type": "point", "at": [1] * 3}]
    )
    assert stats["isError"] and "rendered plane" in stats["content"][0]["text"]


def test_a_pixel_in_a_returned_image_becomes_array_indices(session):
    c, (volume, _, _) = session
    for decorated in (False, True):
        # Find the bright voxel in a plain or ticked axial render, as a model would by eye.
        shown = c.call("view", source=volume, axis="z", index=VOXEL[0], ticks=decorated or None)
        render = shown["structuredContent"]["images"][0]["image"]
        x, y = white_centre(images(shown)[0])
        mark = {"type": "point", "at": {"pixel": [x, y], "image": render}}
        # Shown on the vertical plane through it, it lands on the same voxel.
        across = c.call("view", source=volume, axis="x", index=VOXEL[2], marks=[mark])
        found = across["structuredContent"]["marks"][0]
        assert found["positions"][0] == pytest.approx(VOXEL, abs=0.5)
        on = found["panels"][0]
        assert on["drawn"] and on["value"] == 5 and on["percentile"] > 99 and on["display"] == 1
        assert on["image"] == across["structuredContent"]["images"][0]["image"]
    margin = c.call(
        "view", source=volume, marks=[{"type": "point", "at": {"pixel": [1, 1], "image": render}}]
    )
    assert margin["isError"] and "not on a plane" in margin["content"][0]["text"]
    gone = c.call(
        "view",
        source=volume,
        marks=[{"type": "point", "at": {"pixel": [1, 1], "image": "render-0"}}],
    )
    assert gone["isError"] and "view images" in gone["content"][0]["text"]


def test_a_mark_on_background_reads_as_a_miss(session):
    c, (volume, _, _) = session
    miss = c.call(
        "view",
        source=volume,
        axis="z",
        index=VOXEL[0],
        marks=[{"type": "point", "at": [VOXEL[0], 3, 3]}],
    )
    found = miss["structuredContent"]["marks"][0]["panels"][0]
    assert found["value"] == 0 and found["percentile"] == 0 and found["display"] == 0


def test_images_fit_the_model_budget_and_take_limits_sources_and_a_grid(session):
    c, (volume, _, _) = session
    twice = f"r = handles[{volume!r}]\nregister(tj.Reconstruction(r.volume * 2, r.geometry, 'fbp', {{}}))"
    other = python(c, twice)["value"]
    marks = [{"type": "box", "corners": [[5, 20, 6], [5, 25, 13]], "colour": "magenta"}]
    for count, sources in ((1, volume), (2, [volume, other]), (3, [volume, other, volume])):
        reply = c.call(
            "view", source=sources, axis="z", index=5, marks=marks, limits=[0, 4], grid=True
        )
        assert not reply["isError"], reply["content"][0]["text"]
        data = reply["structuredContent"]
        assert data["image_limits"] == {"edge": 1568, "pixels": 1_150_000}
        pictures = images(reply)
        assert len(pictures) == 1 and len(data["panels"]) == count
        h, w = pictures[0].shape[:2]
        assert (w, h) == (data["images"][0]["width"], data["images"][0]["height"])
        assert max(w, h) <= 1568 and w * h <= 1_150_000
        assert all(p["intensity_limits"] == [0, 4] for p in data["panels"])
        means = [p["mean"] for p in data["marks"][0]["panels"]]
        assert means == [2, 4, 2][:count]  # the same box on each source
        assert (pictures[0][:, 100] == (90, 170, 255)).all(axis=1).sum() > 20  # the grid line
    bad = c.call("view", source=volume, limits=[1, 1])
    assert bad["isError"] and "low < high" in bad["content"][0]["text"]
    small = python(
        c,
        "g = tj.simulate(size=8, n_views=4).scan.geometry\nregister(tj.Reconstruction(np.zeros((8, 8, 8), np.float32), g, 'fbp', {}))",
    )["value"]
    mixed = c.call("view", source=[volume, small], axis="z")
    assert mixed["isError"] and "shape" in mixed["content"][0]["text"]
