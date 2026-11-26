"""A view mark that misses a structure says so, and says where the structure is."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.test_mcp_python import python
from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface

# A sphere on a 64-voxel grid centred at 31.5: a bright shell 18 <= r < 20 (value 1) around
# a core (0.4), in air (0), with seeded noise of 5 % of the shell's contrast.
CENTRE, INNER, OUTER = 31.5, 18, 20


def radius(at):
    z, y, x = at
    return float(np.hypot(np.hypot(z - CENTRE, y - CENTRE), x - CENTRE))


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    with connection(tmp_path_factory.mktemp("affordances")) as c:
        made = python(
            c,
            f"""g = tj.simulate(size=64, n_views=4).scan.geometry
z, y, x = np.mgrid[:64, :64, :64] - {CENTRE}
r = np.sqrt(z**2 + y**2 + x**2)
shell = np.where(r < {INNER}, 0.4, 0.0) + ((r >= {INNER}) & (r < {OUTER}))
noisy = shell + 0.05 * np.random.default_rng(1).standard_normal(shell.shape)
[register(tj.Reconstruction(noisy.astype(np.float32), g, "fbp", {{}})),
 register(tj.Reconstruction(np.zeros((64, 64, 64), np.float32), g, "fbp", {{}}))]""",
        )["value"]
        yield c, made


def marked(c, source, marks, **window):
    reply = c.call("view", source=source, marks=marks, **window)
    assert not reply["isError"], reply["content"][0]["text"]
    return [m["panels"][0] for m in reply["structuredContent"]["marks"]]


@pytest.mark.parametrize("stride", [1, 2])
def test_a_mark_beside_a_thin_shell_names_the_shell(session, stride):
    c, (volume, _) = session
    # About 11 voxels outside the shell's outer face, in air: the walkthrough's miss.
    window = dict(axis="z", index=32, stride=[stride] * 3)
    (miss,) = marked(c, volume, [{"type": "point", "at": [32, 32, 2]}], **window)
    assert miss["drawn"] and miss["value"] < 0.2
    assert "Flat" in miss["warning"] and "nearest.at" in miss["warning"]
    nearest = miss["nearest"]
    assert nearest["at"][0] == 32
    assert INNER - 1 <= radius(nearest["at"]) <= OUTER + 1
    # The shell's outer face is 29.5 - 20 = 9.5 voxels from the mark, its middle 10.5.
    assert 9 <= nearest["distance"] <= 13
    assert nearest["value"] > 0.6 and nearest["percentile"] > 90
    # Pointing at the reported structure, and snapping onto it, lands on the shell.
    (hit,) = marked(c, volume, [{"type": "point", "at": nearest["at"], "snap": "peak"}], **window)
    assert "warning" not in hit and "nearest" not in hit
    assert hit["value"] > 0.8 and INNER - 0.5 <= radius(hit["snapped"]) <= OUTER


def test_marks_on_structure_carry_no_warning(session):
    c, (volume, _) = session
    found = marked(
        c,
        volume,
        [
            {"type": "point", "at": [32, 32, 12.5]},  # r = 19, in the shell
            {"type": "point", "at": [32, 32, 14.5]},  # r = 17, the core one voxel inside it
            {"type": "point", "at": [32, 32, 9.5]},  # r = 22, air two voxels outside it
            {"type": "box", "corners": [[32, 2, 2], [32, 4, 4]]},  # regions are not judged
            {"type": "text", "at": [32, 3, 3], "label": "air"},
        ],
        axis="z",
        index=32,
    )
    assert all(f["drawn"] for f in found)
    assert not any("warning" in f or "nearest" in f for f in found), found


def test_flat_marks_inside_the_core_and_at_an_arrow_head_are_reported(session):
    c, (volume, _) = session
    core, arrow = marked(
        c,
        volume,
        [
            {"type": "point", "at": [32, 32, 32]},
            # The tail sits on the shell; only the head is judged.
            {"type": "arrow", "from": [32, 32, 12.5], "to": [32, 32, 2]},
        ],
        axis="z",
        index=32,
    )
    # From the centre of the core the closest sample that differs is on the shell's inner side.
    assert INNER - 1 <= radius(core["nearest"]["at"]) <= OUTER and core["nearest"]["value"] > 0.6
    assert 16 <= core["nearest"]["distance"] <= 20
    assert "nearest" in arrow and INNER - 1 <= radius(arrow["nearest"]["at"]) <= OUTER + 1


def test_a_plane_with_nothing_on_it_is_flat_without_a_nearest_structure(session):
    c, (volume, empty) = session
    # z = 2 passes 29.5 voxels from the centre, outside the sphere: air and noise only.
    (air,) = marked(c, volume, [{"type": "point", "at": [2, 32, 32]}], axis="z", index=2)
    (blank,) = marked(c, empty, [{"type": "point", "at": [32, 10, 10]}], axis="z", index=32)
    for found in (air, blank):
        assert "nothing on this plane does" in found["warning"] and "nearest" not in found


@pytest.mark.slow
def test_the_imaging_eval_meets_its_bars(tmp_path):
    script = Path(__file__).parents[1] / "tools" / "eval_mcp_imaging.py"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "TMPDIR": str(tmp_path)}
    run = subprocess.run(
        [sys.executable, str(script), "--size", "32"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    scores = {row["scenario"]: row for row in map(json.loads, run.stdout.splitlines())}
    assert set(scores) == {"shell", "centre", "pose"}
