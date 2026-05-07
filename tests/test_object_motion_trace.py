from pathlib import Path

import numpy as np
import pytest

from tomojax.motion import ObjectMotionTrace, read_object_motion_csv, write_object_motion_csv


def test_object_motion_trace_round_trips_csv_and_reports_tx_rmse(tmp_path: Path) -> None:
    trace = ObjectMotionTrace(
        tx_obj_px=np.array([0.0, -1.0, -2.0], dtype=np.float64),
        ty_obj_px=np.array([0.0, 0.5, 0.0], dtype=np.float64),
        tz_obj_px=np.array([0.0, 0.2, 0.4], dtype=np.float64),
        rot_obj_z_deg=np.array([0.0, 0.1, 0.2], dtype=np.float64),
    )

    path = tmp_path / "true_motion.csv"
    write_object_motion_csv(path, trace)
    loaded = read_object_motion_csv(path)

    assert loaded.n_views == 3
    np.testing.assert_allclose(loaded.tx_obj_px, trace.tx_obj_px)
    np.testing.assert_allclose(loaded.ty_obj_px, trace.ty_obj_px)
    np.testing.assert_allclose(loaded.tz_obj_px, trace.tz_obj_px)
    np.testing.assert_allclose(loaded.rot_obj_z_deg, trace.rot_obj_z_deg)
    assert loaded.tx_rmse_px(trace) == 0.0


def test_object_motion_trace_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="same shape"):
        _ = ObjectMotionTrace(
            tx_obj_px=np.zeros(2, dtype=np.float64),
            ty_obj_px=np.zeros(3, dtype=np.float64),
            tz_obj_px=np.zeros(2, dtype=np.float64),
            rot_obj_z_deg=np.zeros(2, dtype=np.float64),
        )
