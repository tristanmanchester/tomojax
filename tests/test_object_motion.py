"""Object-motion sidecars keep each displacement on the view it names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tomojax.motion import read_object_motion_csv

HEADER = "view,tx_obj_px,ty_obj_px,tz_obj_px,rot_obj_z_deg\n"


def test_motion_rows_follow_their_view_ids_not_their_order(tmp_path: Path) -> None:
    path = tmp_path / "true_motion.csv"
    path.write_text(HEADER + "1,10,0,0,0\n0,0,0,0,0\n")
    np.testing.assert_array_equal(read_object_motion_csv(path).tx_obj_px, [0.0, 10.0])


@pytest.mark.parametrize("ids", [(0, 0), (0, 2)])
def test_duplicate_or_missing_view_ids_are_rejected(tmp_path: Path, ids) -> None:
    path = tmp_path / "true_motion.csv"
    path.write_text(HEADER + "".join(f"{i},1,0,0,0\n" for i in ids))
    with pytest.raises(ValueError, match="each view"):
        read_object_motion_csv(path)
