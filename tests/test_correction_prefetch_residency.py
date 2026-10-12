"""Prefetched device reads stay on the source even without its creation context."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("platform", ["cpu", None])
def test_prefetched_slices_keep_input_residency(
    monkeypatch: pytest.MonkeyPatch, raw: bool, committed: bool, platform: str | None
) -> None:
    host = np.linspace(100 if raw else 0.1, 800 if raw else 0.9, 17 * 4 * 7, dtype=np.float32)
    host = host.reshape(17, 4, 7)
    device = jax.devices(platform)[-1]
    flats, darks = np.full((3, 4, 7), 1000, np.float32), np.full((5, 4, 7), 10, np.float32)
    with jax.default_device(device):
        source = jnp.asarray(host)
        device_flats, device_darks = jnp.asarray(flats), jnp.asarray(darks)
    assert not source.committed
    if committed:
        source = jax.device_put(source, device)
        device_flats = jax.device_put(device_flats, device)
        device_darks = jax.device_put(device_darks, device)
    pointer = source.unsafe_buffer_pointer()
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 4, 1, 1, 1), tj.Detector(7, 4, 1, 1), np.arange(17) * 180 / 17
    )

    def corrected(data: np.ndarray | jax.Array) -> tj.Scan:
        out = np.empty(host.shape, np.float32)
        if raw:
            return tj.Frames(
                data,
                geometry,
                flats=device_flats if isinstance(data, jax.Array) else flats,
                darks=device_darks if isinstance(data, jax.Array) else darks,
                flat_positions=np.array([0, 17, 0]),
            ).corrected(batch_views=3, out=out)
        return tj.Scan(data, geometry).corrected(batch_views=3, out=out)

    slices = []
    field_slices = []
    original = type(source).__getitem__

    def get(array: jax.Array, key: object) -> jax.Array:
        result = original(array, key)
        if array.shape == source.shape:
            slices.append((result.shape[0], result.devices()))
        if raw and array.shape in (flats.shape, darks.shape):
            field_slices.append((result.shape[0], result.devices()))
        return result

    monkeypatch.setattr(type(source), "__getitem__", get)
    # The main thread's context need not match either the source or the worker.
    with jax.default_device(jax.devices("cpu")[0]):
        expected, actual = corrected(host), corrected(source)
    assert slices == [(3, {device})] * 5 + [(2, {device})]
    if raw:
        assert field_slices == [(5, {device}), (2, {device}), (1, {device})]
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections
    assert source.devices() == {device}
    assert source.committed == committed
    assert source.unsafe_buffer_pointer() == pointer


def test_prefetch_preserves_residency_on_a_nondefault_cpu() -> None:
    # A fresh process makes this regression run even on a one-GPU/one-CPU host.
    code = """
import itertools, runpy
import pytest
checks = runpy.run_path(TEST_PATH)
for raw, committed, platform in itertools.product([False, True], [False, True], ['cpu', None]):
    with pytest.MonkeyPatch.context() as patch:
        checks['test_prefetched_slices_keep_input_residency'](patch, raw, committed, platform)
""".replace("TEST_PATH", repr(str(Path(__file__).resolve())))
    env = os.environ | {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
