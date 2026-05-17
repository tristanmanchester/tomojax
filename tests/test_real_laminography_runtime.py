from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tomojax.bench.real_laminography_runtime import (
    append_real_lamino_csv,
    real_lamino_json_safe,
    relative_l2,
    select_real_lamino_views,
    timed_repeats,
    update_real_lamino_status,
    write_real_lamino_json,
)


def test_real_lamino_runtime_writes_json_csv_and_status(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "payload.json"
    csv_path = tmp_path / "summary.csv"
    status_path = tmp_path / "status.json"

    write_real_lamino_json(json_path, {"value": np.float32(1.25)})
    append_real_lamino_csv(csv_path, {"stage": "probe", "loss": np.float32(2.5)}, ["stage", "loss"])
    update_real_lamino_status(status_path, state="running", error="previous")
    update_real_lamino_status(status_path, state="running", stage="baseline", message="baseline")
    update_real_lamino_status(status_path, state="running", stage="setup")
    update_real_lamino_status(status_path, state="completed", stage="done")

    assert json.loads(json_path.read_text()) == {"value": 1.25}
    assert csv_path.read_text().splitlines() == ["stage,loss", "probe,2.5"]
    status = json.loads(status_path.read_text())
    assert status["state"] == "completed"
    assert status["stage"] == "done"
    assert "error" not in status
    assert "message" not in status
    assert isinstance(status["updated_at"], float)


def test_real_lamino_runtime_view_selection_and_norms_are_deterministic() -> None:
    projections = np.arange(10 * 2 * 2, dtype=np.float32).reshape(10, 2, 2)
    thetas = np.linspace(0.0, 180.0, 10, endpoint=False, dtype=np.float32)

    selected, selected_thetas, indices = select_real_lamino_views(
        projections,
        thetas,
        max_views=4,
    )

    np.testing.assert_array_equal(indices, np.asarray([0, 3, 6, 9], dtype=np.int32))
    np.testing.assert_array_equal(selected, projections[indices])
    np.testing.assert_array_equal(selected_thetas, thetas[indices])
    assert relative_l2(np.asarray([2.0, 0.0]), np.asarray([1.0, 0.0])) == 1.0
    assert real_lamino_json_safe({"x": np.float32(3.5)}) == {"x": 3.5}


def test_real_lamino_timed_repeats_reports_shape() -> None:
    result, timing = timed_repeats(
        name="constant",
        fn=lambda: jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        repeats=2,
        warmups=1,
    )

    np.testing.assert_array_equal(np.asarray(result), np.asarray([1.0, 2.0], dtype=np.float32))
    assert timing["name"] == "constant"
    assert timing["warmup_repeats"] == 1
    assert timing["measured_repeats"] == 2
    assert len(timing["times_seconds"]) == 2
    assert timing["median_seconds"] >= 0.0
