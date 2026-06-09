from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import (
    CalibrationState,
    CalibrationVariable,
    build_calibrated_geometry_metadata_patch,
)


def test_calibration_variable_normalizes_numpy_scalars() -> None:
    variable = CalibrationVariable(
        name="det_u_px",
        value=np.int64(42),
        unit="px",
        status="estimated",
        frame="detector",
    )

    payload = variable.to_dict()
    restored = CalibrationVariable.from_dict(payload)

    assert payload["value"] == 42
    assert isinstance(payload["value"], int)
    assert restored.value == 42
    assert isinstance(restored.value, int)


def test_calibration_state_normalizes_numpy_arrays() -> None:
    state = CalibrationState(
        detector=(
            CalibrationVariable(
                name="axis_unit_lab",
                value=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                unit="unit",
                status="estimated",
                frame="detector",
            ),
        )
    )

    payload = state.to_dict()

    assert payload["detector"][0]["value"] == [0.0, 0.0, 1.0]


def test_calibration_state_normalizes_jax_arrays() -> None:
    state = CalibrationState(
        detector=(
            CalibrationVariable(
                name="det_u_px",
                value=jnp.asarray(2.5, dtype=jnp.float32),
                unit="px",
                status="estimated",
                frame="detector",
            ),
            CalibrationVariable(
                name="axis_unit_lab",
                value=jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32),
                unit="unit",
                status="estimated",
                frame="detector",
            ),
        )
    )

    payload = state.to_dict()
    restored = CalibrationState.from_dict(payload)

    assert payload["detector"][0]["value"] == pytest.approx(2.5)
    assert isinstance(payload["detector"][0]["value"], float)
    assert payload["detector"][1]["value"] == [0.0, 0.0, 1.0]
    assert restored.detector[0].value == pytest.approx(2.5)
    assert restored.detector[1].value == [0.0, 0.0, 1.0]


def test_calibrated_geometry_patch_requires_detector_pixel_spacing() -> None:
    state = CalibrationState(
        detector=(
            CalibrationVariable(
                name="det_u_px",
                value=2.0,
                unit="px",
                status="estimated",
                frame="detector",
            ),
        )
    )

    with pytest.raises(ValueError, match="requires du, dv"):
        build_calibrated_geometry_metadata_patch(
            calibration_state=state,
            detector={"nu": 4, "nv": 4},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "learned", "Unknown calibration variable status"),
        ("frame", "gantry", "Unknown calibration variable frame"),
    ],
)
def test_calibration_variable_from_dict_rejects_unknown_literals(
    field: str,
    value: str,
    message: str,
) -> None:
    payload = {
        "name": "det_u_px",
        "value": 1.0,
        "unit": "px",
        "status": "estimated",
        "frame": "detector",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        CalibrationVariable.from_dict(payload)
