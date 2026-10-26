"""View-packet scheduling preserves every ray and gives each output one owner."""

import numpy as np
import pytest

import tomojax as tj
from tomojax.core import cone
from tomojax.geometry import stack_view_poses


@pytest.mark.gpu
@pytest.mark.parametrize("views", [1, 2, 3, 4, 5, 6, 7, 17, 33])
@pytest.mark.parametrize("suffix", ["", "64"])
@pytest.mark.parametrize("rolled", [False, True])
def test_packet_paths_partition_pixels_and_match_individual_views(views, suffix, rolled):
    import cupy as cp

    grid = tj.Grid(32, 29, 35, 0.9, 1.1, 1.0)
    detector = tj.Detector(43, 37, 1.2, 1.3, (0.4, -0.3))
    # Dense, nonmonotone sparse, repeated dense, and partial final packets.
    angles = np.concatenate(
        [np.arange(8) * 0.2, [0, 90, 180, 270, 0, 270, 180, 90], np.zeros(8), np.arange(9) * 0.2]
    )[:views]
    geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(201, 301.5))
    frames = cone.cone_model(geometry).frames(views).copy()
    if rolled:
        horizontal, vertical = frames[:, 1, :].copy(), frames[:, 2, :].copy()
        frames[1::2, 1, :], frames[1::2, 2, :] = vertical[1::2], -horizontal[1::2]
    coeff = cp.asarray(
        cone.frame_coefficients(stack_view_poses(geometry, views), frames, grid, detector)
    )
    volume = cp.asarray(
        np.random.default_rng(55).uniform(-1, 1, (grid.nz, grid.ny, grid.nx)).astype(np.float32)
    )
    saved_coeff, saved_volume = coeff.copy(), volume.copy()
    expected = cp.empty((views, detector.nv, detector.nu), cp.float32)
    for view in range(views):
        cone.forward_into(
            expected[view : view + 1], coeff[view : view + 1], volume, grid=grid, det=detector
        )
    args = (
        *(np.int32(v) for v in (grid.nx, grid.ny, grid.nz, detector.nu, detector.nv)),
        *(np.float32(v) for v in (grid.vx, grid.vy, grid.vz)),
        np.int32(views),
    )
    outputs = []
    for name, gu, gv, group in [("dense", 8, 4, 8), ("sparse", 16, 16, 1)]:
        output = cp.full_like(expected, cp.nan)
        cone.cuda_module().get_function("cone_forward_" + name + suffix)(
            (-(-views // group), -(-detector.nu // gu), -(-detector.nv // gv)),
            (256,),
            (coeff, volume, output, *args),
        )
        outputs.append(output)
    dense, sparse = outputs
    dense_mask, sparse_mask = cp.isfinite(dense), cp.isfinite(sparse)
    assert bool(cp.all(dense_mask ^ sparse_mask))
    if views >= 17:
        assert bool(cp.any(dense_mask))
        assert bool(cp.any(sparse_mask))
    combined = cp.where(dense_mask, dense, sparse)
    np.testing.assert_array_equal(combined.view(cp.uint32).get(), expected.view(cp.uint32).get())
    assert bool(cp.array_equal(coeff, saved_coeff))
    assert bool(cp.array_equal(volume, saved_volume))
