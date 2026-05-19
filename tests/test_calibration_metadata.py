from __future__ import annotations

import numpy as np

from tomojax.geometry import CalibrationState, CalibrationVariable


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
