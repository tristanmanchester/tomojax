"""Resident corrections respect producer headroom, explicit batches and old host fallbacks."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.backends import api as backend_runtime
from tomojax.corrections import BeamHardening


def _scan(device=False):
    data = np.linspace(0.1, 2, 13 * 9 * 11, dtype=np.float32).reshape(13, 9, 11)
    geometry = tj.ParallelGeometry(
        tj.Grid(11, 11, 9, 1, 1, 1), tj.Detector(11, 9, 1, 1), np.arange(13)
    )
    return tj.Scan(jax.device_put(data) if device else data, geometry)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_automatic_resident_batches_follow_the_actual_source_device_budget(cap, monkeypatch):
    original = _scan(True)
    expected = _scan().corrected(BeamHardening((1, 0.05)))
    queried = []

    def free(*, device=None):
        queried.append(device)
        return 2 * cap * 9 * 11 * 4

    monkeypatch.setattr(backend_runtime, "device_free_memory_bytes", free)
    asarray = np.asarray
    downloads = []

    def tracked(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            downloads.append(tuple(value.shape))
        return asarray(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "asarray", tracked)
        result = original.corrected(BeamHardening((1, 0.05)))
    assert set(queried) == original.projections.devices()
    assert downloads == [(min(cap, 13 - start), 9, 11) for start in range(0, 13, cap)]
    np.testing.assert_array_equal(result.projections, expected.projections)


@pytest.mark.parametrize("free", [None, 0, 2 * 9 * 11 * 4 - 1])
def test_unknown_or_no_single_view_budget_keeps_the_old_host_route(free, monkeypatch):
    original = _scan(True)
    expected = _scan().corrected(BeamHardening((1, 0.05)), batch_views=3)
    monkeypatch.setattr(backend_runtime, "device_free_memory_bytes", lambda *, device=None: free)
    scheduled = []
    asynchronous = type(original.projections).copy_to_host_async

    def tracked(value):
        scheduled.append(tuple(value.shape))
        return asynchronous(value)

    monkeypatch.setattr(type(original.projections), "copy_to_host_async", tracked)
    result = original.corrected(BeamHardening((1, 0.05)), batch_views=3)
    assert scheduled == []
    np.testing.assert_array_equal(result.projections, expected.projections)


@pytest.mark.parametrize("requested", [2, 3, 20])
def test_explicit_batches_are_not_silently_reduced_to_fit(requested, monkeypatch):
    original = _scan(True)
    monkeypatch.setattr(
        backend_runtime, "device_free_memory_bytes", lambda *, device=None: 2 * 2 * 9 * 11 * 4
    )
    scheduled = []
    asynchronous = type(original.projections).copy_to_host_async

    def tracked(value):
        scheduled.append(tuple(value.shape))
        return asynchronous(value)

    with monkeypatch.context() as patch:
        patch.setattr(type(original.projections), "copy_to_host_async", tracked)
        result = original.corrected(BeamHardening((1, 0.05)), batch_views=requested)
    assert scheduled == (
        [(min(requested, 13 - start), 9, 11) for start in range(0, 13, requested)]
        if requested <= 2
        else []
    )
    expected = _scan().corrected(BeamHardening((1, 0.05)), batch_views=requested)
    np.testing.assert_array_equal(result.projections, expected.projections)
