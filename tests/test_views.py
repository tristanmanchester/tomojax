from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from tomojax.core.geometry.views import stack_view_poses


class DummyGeometry:
    def pose_for_view(self, i: int) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(i)
        pose[2, 3] = -float(i)
        return pose


def test_stack_view_poses_preserves_order_and_dtype():
    poses = stack_view_poses(DummyGeometry(), 3, dtype=jnp.float16)

    assert poses.shape == (3, 4, 4)
    assert poses.dtype == jnp.float16
    np.testing.assert_allclose(np.asarray(poses[:, 0, 3]), [0.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(poses[:, 2, 3]), [0.0, -1.0, -2.0], atol=1e-6)
