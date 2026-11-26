"""The python tool as the path for compound work: sweeps, small results, short errors."""

from pathlib import Path
import time

import pytest

from tests.test_mcp_features import CYAN, feature
from tests.test_mcp_guides import example
from tests.test_mcp_marks import images
from tests.test_mcp_python import python
from tests.test_mcp_stdio import connection
import tomojax

pytestmark = pytest.mark.surface

TUNING = (
    Path(tomojax.__file__).parent / "skill" / "tomojax" / "references" / "tuning-without-truth.md"
)


def test_the_documented_sweep_scores_candidates_and_leaves_viewable_handles(tmp_path):
    with connection(tmp_path) as c:
        sim = c.call("simulate", size=32, n_views=48, poisson_scale=3000)["structuredContent"]
        scan = sim["members"]["scan"]
        first = c.call("reconstruct", source=scan, method="fbp")["structuredContent"]["handle"]
        # The bright interior of the Shepp-Logan head, then a band around it.
        inside = feature(c, source=first, recipe={"type": "seeded", "seeds": [[16, 10, 16]]})
        around = feature(
            c, source=first, recipe={"type": "band", "feature": inside["handle"]} | {"outer": 4}
        )
        setup = f"scan_id, feature_id, surround_id = {scan!r}, {inside['handle']!r}, {around['handle']!r}"
        out = python(c, setup + "\n" + example(TUNING.read_text(), "Sweep and score"))
        rows = out["value"]
        assert {r["handle"] for r in rows} == set(out["handles"])
        assert [r["tv_weight"] for r in rows[1:]] == pytest.approx([1e-5, 1e-4, 1e-3])
        assert rows[0]["tv_weight"] > 0  # the simulated measurements include shot noise
        for row in rows:
            assert row["contrast_to_noise"] > 1 and row["noise"] > 0 and row["edge_width"] > 0
            shown = c.call("view", source=row["handle"], axis="z", features=[inside["handle"]])
            assert not shown["isError"], shown
            assert (images(shown)[0] == CYAN).all(axis=2).sum() > 40
        # More total variation blurs the boundary more.
        ordered = sorted(rows, key=lambda r: r["tv_weight"])
        widths = [r["edge_width"] for r in ordered]
        assert widths == sorted(widths) and widths[-1] > widths[0]


def test_results_stay_small_and_errors_name_the_line(tmp_path):
    with connection(tmp_path) as c:
        out = python(
            c, "{'long': np.arange(100.0), 'short': np.arange(3), 'scalar': np.float32(2)}"
        )
        assert out["value"] == {
            "long": {"shape": [100], "dtype": "float64", "min": 0.0, "max": 99.0, "mean": 49.5},
            "short": [0, 1, 2],
            "scalar": 2.0,
        }
        code = "def scale(x):\n    return x / 0\n\nscale(np.float64(2.0)) + scale(1)"
        short = c.call("python", code=code)
        assert short["isError"]
        text = short["structuredContent"]["message"]
        assert "line 4: scale(np.float64(2.0)) + scale(1)" in text
        assert "line 2: return x / 0" in text and "ZeroDivisionError" in text
        assert 'File "' not in text
        full = c.call("python", code=code, traceback=True)["structuredContent"]["message"]
        assert 'File "<python>", line 2' in full


def test_a_long_python_sweep_runs_as_a_task_and_cancels(tmp_path):
    with connection(tmp_path) as c:
        scan = c.call("simulate", size=16)["structuredContent"]["members"]["scan"]
        code = f"""scan = handles[{scan.split(".")[0]!r}].scan
while True:
    tj.reconstruct(scan, "sirt", iterations=10000)"""
        task = c.call("python", tasks=True, code=code)
        assert task["resultType"] == "task"
        deadline = time.monotonic() + 30
        while (
            c.task("get", task["taskId"])["status"] == "working"
            and not c.call("status")["structuredContent"]["jobs"]
        ):
            assert time.monotonic() < deadline
        c.task("cancel", task["taskId"])
        assert c.terminal(task["taskId"])["status"] == "cancelled"
        assert python(c, "1 + 1")["value"] == 2  # the worker is free again


SHELL = TUNING.with_name("thin-shell.md")


def test_the_thin_shell_example_segments_and_scores_the_shell(tmp_path):
    with connection(tmp_path) as c:
        sim = c.call("simulate", size=128, n_views=180, poisson_scale=5000)["structuredContent"]
        recon = c.call("reconstruct", source=sim["members"]["scan"])["structuredContent"]["handle"]
        # Measure first: the shell is the peak of a profile along the middle row.
        row = c.call("view", source=recon, mode="profile", roi=[[64, 65], [64, 65], [0, 128]])
        values = row["structuredContent"]["profile"]
        x = max(range(64), key=lambda i: values[i])
        setup = (
            f"recon_id = {recon!r}\nshell_at = (64, 64, {x})\n"
            "core_at = (64, 64, 40)\nair_at = (64, 64, 4)\n"
        )
        text = SHELL.read_text()
        made = python(c, setup + example(text, "2. Segment by threshold, and check it is stable"))
        found = made["value"]
        assert min(found["stability"]) > 0.9 and found["shell"] > found["core"]
        truth = python(
            c,
            f"""phantom = np.asarray(handles[{sim["handle"]!r}].phantom)
true = np.isclose(phantom, phantom[64, 64, {x}]) & (phantom > phantom[64, 64, 40])
mask = handles[shell_id].mask
float(2 * (mask & true).sum() / (mask.sum() + true.sum()))""",
        )["value"]
        assert truth > 0.8  # Dice against the phantom's shell
        scored = python(c, example(text, "3. Score the shell against each side"))["value"]
        core, air = scored["against_core"], scored["against_air"]
        assert (
            core["contrast_to_noise"] > 1 and air["contrast_to_noise"] > core["contrast_to_noise"]
        )
