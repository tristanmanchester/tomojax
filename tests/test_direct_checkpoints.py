"""Direct-method checkpoints count synchronized numerical batches across device shares."""

from importlib import import_module
import itertools
from threading import enumerate as threads, get_ident

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax import progress
from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid, ParallelGeometry
from tomojax.recon import FBPConfig, FDKConfig, fbp, fdk


@pytest.mark.parametrize("method", ["fbp", "fdk"])
@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("devices", [1, 2])
@pytest.mark.parametrize("cancel", [False, True])
def test_nondivisible_view_batches(method, host, devices, cancel, monkeypatch):
    if len(jax.devices()) < devices:
        pytest.skip("needs two devices")
    grid = Grid(7, 6, 5, 0.92, 1.13, 0.79)
    detector = Detector(11, 8, 1.04, 1.07, (0.37, -0.19))
    placement = tuple(jax.devices()[:devices]) if devices > 1 else None
    values = np.random.default_rng(419).normal(size=(19, 8, 11)).astype(np.float32)
    data = values if host else jnp.asarray(values)
    # The native FBP batch size normally varies with scan size and workspace.
    if method == "fbp":
        monkeypatch.setattr(
            import_module("tomojax.recon.fbp"), "parallel_filter_batch_size", lambda *_: 3
        )
        geometry = ParallelGeometry(grid, detector, np.linspace(0, 180, 19, endpoint=False))
        solve, config = fbp, FBPConfig(devices=placement, backend="jax")
        sample = [0.02850862, 0.11065112, 0.03739797, -0.22521386, -0.00283955]
    else:
        geometry = ConeGeometry(
            grid, detector, np.linspace(0, 360, 19, endpoint=False), ConeBeam(19, 34)
        )
        solve, config = fdk, FDKConfig(devices=placement, views_per_batch=3, backend="jax")
        sample = [0.00213336, 0.18044995, -0.03095098, 0.28804696, 0.08832941]
    reference = np.asarray(solve(geometry, grid, detector, data, config=config))
    # Voxel samples from the whole stack in one batch on the CPU; the rows of this random
    # data end above zero, so they include its continuation past the detector.
    np.testing.assert_allclose(reference.ravel()[[0, 17, 67, 137, 209]], sample, rtol=0, atol=3e-6)
    owner, before = get_ident(), set(threads())
    updates, control = [], progress.Control()

    def observe(event):
        assert get_ident() == owner
        if isinstance(event, progress.Advanced):
            updates.append((event.done, event.total))
            if cancel:
                control.cancel()

    with progress.watch(observe, control=control):
        if cancel:
            with pytest.raises(progress.Canceled):
                solve(geometry, grid, detector, data, config=config)
        else:
            actual = solve(geometry, grid, detector, data, config=config)
            np.testing.assert_array_equal(actual, reference)
    # 19 views in one share, or 10 and 9 in two. Batch widths are always three.
    assert updates[0] == (devices, 7)
    if cancel:
        assert len(updates) == 1
    else:
        assert updates[-1] == (7, 7)
        assert all(first[0] < second[0] for first, second in itertools.pairwise(updates))
    assert set(threads()) <= before
