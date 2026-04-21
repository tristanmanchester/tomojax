from __future__ import annotations

import csv
import json

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align.gauge import apply_alignment_gauge
from tomojax.align.params_export import (
    ALIGNMENT_PARAMS_SCHEMA,
    CSV_FIELDNAMES,
    PARAMETER_ORDER,
    save_alignment_params_csv,
    save_alignment_params_json,
)


def test_alignment_params_export_writes_named_json_and_csv(tmp_path):
    params5 = np.asarray(
        [
            [0.1, 0.2, 0.3, 1.0, 4.0],
            [-0.1, -0.2, -0.3, -2.0, -6.0],
        ],
        dtype=np.float32,
    )
    json_path = tmp_path / "nested" / "params.json"
    csv_path = tmp_path / "nested" / "params.csv"

    gauge_metadata = {
        "mode": "mean_translation",
        "dofs": ["dx", "dz"],
        "final": {"dx_mean_after": 0.0, "dz_mean_after": 0.0},
    }
    save_alignment_params_json(
        json_path,
        params5,
        du=0.5,
        dv=2.0,
        gauge_metadata=gauge_metadata,
    )
    save_alignment_params_csv(csv_path, params5, du=0.5, dv=2.0)

    assert json_path.exists()
    assert csv_path.exists()

    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].split(",") == list(CSV_FIELDNAMES)
    rows = list(csv.DictReader(csv_lines))
    assert len(rows) == 2
    assert rows[0]["view_index"] == "0"
    assert float(rows[0]["alpha_rad"]) == pytest.approx(0.1)
    assert float(rows[0]["dx_world"]) == pytest.approx(1.0)
    assert float(rows[0]["dz_world"]) == pytest.approx(4.0)
    assert float(rows[0]["dx_px"]) == pytest.approx(2.0)
    assert float(rows[0]["dz_px"]) == pytest.approx(2.0)
    assert rows[1]["view_index"] == "1"
    assert float(rows[1]["dx_px"]) == pytest.approx(-4.0)
    assert float(rows[1]["dz_px"]) == pytest.approx(-3.0)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == ALIGNMENT_PARAMS_SCHEMA
    assert payload["parameter_order"] == list(PARAMETER_ORDER)
    assert payload["units"]["alpha"] == "rad"
    assert payload["units"]["dx"] == "world"
    assert payload["units"]["dx_px"] == "pixel"
    assert payload["detector_spacing"] == {"du": 0.5, "dv": 2.0}
    assert payload["gauge_fix"] == gauge_metadata
    assert len(payload["views"]) == 2
    assert set(payload["views"][0]) == set(CSV_FIELDNAMES)
    assert payload["views"][0]["view_index"] == 0
    assert payload["views"][0]["dx_px"] == pytest.approx(2.0)
    assert payload["views"][1]["dz_px"] == pytest.approx(-3.0)


def test_alignment_params_json_serializes_raw_jax_gauge_metadata(tmp_path):
    params5 = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 1.5, -0.5],
            [0.0, 0.0, 0.0, 2.5, 1.5],
        ],
        dtype=jnp.float32,
    )
    bounds_lower = jnp.full((5,), -jnp.inf, dtype=jnp.float32)
    bounds_upper = jnp.full((5,), jnp.inf, dtype=jnp.float32)
    _, gauge_metadata = apply_alignment_gauge(
        params5,
        mode="mean_translation",
        active_mask=(True, True, True, True, True),
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
    )
    json_path = tmp_path / "params.json"

    save_alignment_params_json(
        json_path,
        params5,
        du=1.0,
        dv=1.0,
        gauge_metadata=dict(gauge_metadata),
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["gauge_fix"]["mode"] == "mean_translation"
    assert payload["gauge_fix"]["dofs"] == ["dx", "dz"]
    assert payload["gauge_fix"]["dx_mean_before"] == pytest.approx(2.0)
    assert payload["gauge_fix"]["dz_mean_before"] == pytest.approx(0.5)
    assert payload["gauge_fix"]["dx_mean_after"] == pytest.approx(0.0)
    assert payload["gauge_fix"]["dz_mean_after"] == pytest.approx(0.0)
