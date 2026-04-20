from __future__ import annotations

import json
import sys

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.core.geometry import Detector, Grid
from tomojax.data.artefacts import (
    SimulationArtefacts,
    apply_simulation_artefacts,
)
from tomojax.data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo, validate_nxtomo


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def _base(value: float = 1.0) -> jnp.ndarray:
    return jnp.full((5, 4, 6), value, dtype=jnp.float32)


def _assert_common(
    source: jnp.ndarray,
    artefacts: SimulationArtefacts,
    *,
    seed: int = 11,
) -> tuple[np.ndarray, dict]:
    out_a, meta_a = apply_simulation_artefacts(source, artefacts, seed=seed)
    out_b, meta_b = apply_simulation_artefacts(source, artefacts, seed=seed)
    arr_a = np.asarray(out_a)
    arr_b = np.asarray(out_b)

    assert arr_a.shape == np.asarray(source).shape
    assert arr_a.dtype == np.asarray(source).dtype
    assert np.array_equal(arr_a, arr_b)
    assert meta_a == meta_b
    assert meta_a["enabled"] is True
    return arr_a, meta_a


def test_default_artefacts_are_exact_noop() -> None:
    source = _base()

    out_none, meta_none = apply_simulation_artefacts(source, None, seed=1)
    out_default, meta_default = apply_simulation_artefacts(
        source,
        SimulationArtefacts(),
        seed=1,
    )

    assert np.array_equal(np.asarray(out_none), np.asarray(source))
    assert np.array_equal(np.asarray(out_default), np.asarray(source))
    assert meta_none["enabled"] is False
    assert meta_default["enabled"] is False


def test_poisson_noise_shape_dtype_determinism_and_effect() -> None:
    source = _base(0.5)
    out, meta = _assert_common(source, SimulationArtefacts(poisson_scale=20.0))

    assert np.all(out >= 0.0)
    assert not np.array_equal(out, np.asarray(source))
    assert meta["poisson_scale"] == 20.0


def test_gaussian_noise_shape_dtype_determinism_and_effect() -> None:
    source = _base(0.5)
    out, meta = _assert_common(source, SimulationArtefacts(gaussian_sigma=0.1))

    assert not np.array_equal(out, np.asarray(source))
    assert float(np.var(out - np.asarray(source))) > 0.0
    assert meta["gaussian_sigma"] == 0.1


def test_dead_pixels_are_persistent_detector_locations() -> None:
    source = _base(1.0)
    out, meta = _assert_common(
        source,
        SimulationArtefacts(dead_pixel_fraction=0.1, dead_pixel_value=-3.0),
    )

    assert meta["dead_pixel_count"] > 0
    for flat_index in meta["dead_pixel_indices"]:
        row, col = np.unravel_index(flat_index, source.shape[1:])
        assert np.all(out[:, row, col] == -3.0)


def test_hot_pixels_are_persistent_detector_locations() -> None:
    source = _base(1.0)
    out, meta = _assert_common(
        source,
        SimulationArtefacts(hot_pixel_fraction=0.1, hot_pixel_value=7.0),
    )

    assert meta["hot_pixel_count"] > 0
    for flat_index in meta["hot_pixel_indices"]:
        row, col = np.unravel_index(flat_index, source.shape[1:])
        assert np.all(out[:, row, col] == 7.0)


def test_zingers_create_sparse_bright_impulses() -> None:
    source = _base(0.0)
    out, meta = _assert_common(
        source,
        SimulationArtefacts(zinger_fraction=0.05, zinger_value=5.0),
    )

    assert meta["zinger_count"] > 0
    assert np.count_nonzero(out == 5.0) == meta["zinger_count"]
    assert np.max(out) == 5.0


def test_stripes_create_column_correlated_gain_changes() -> None:
    source = _base(1.0)
    out, meta = _assert_common(
        source,
        SimulationArtefacts(stripe_fraction=0.5, stripe_gain_sigma=0.25),
    )

    assert meta["stripe_count"] > 0
    changed_columns = 0
    for col, gain in zip(meta["stripe_columns"], meta["stripe_gains"], strict=True):
        assert np.allclose(out[:, :, col], gain)
        changed_columns += int(not np.isclose(gain, 1.0))
    assert changed_columns > 0


def test_dropped_views_preserve_shape_and_replace_whole_views() -> None:
    source = _base(1.0)
    out, meta = _assert_common(
        source,
        SimulationArtefacts(dropped_view_fraction=0.4, dropped_view_fill=-2.0),
    )

    assert meta["dropped_view_count"] > 0
    for view_index in meta["dropped_view_indices"]:
        assert np.all(out[view_index] == -2.0)
    assert out.shape[0] == source.shape[0]


def test_detector_blur_spreads_detector_impulse_without_view_blur() -> None:
    source_np = np.zeros((3, 7, 7), dtype=np.float32)
    source_np[1, 3, 3] = 1.0
    source = jnp.asarray(source_np)

    out, meta = _assert_common(
        source,
        SimulationArtefacts(detector_blur_sigma=0.8),
    )

    assert meta["detector_blur_sigma"] == 0.8
    assert 0.0 < out[1, 3, 3] < 1.0
    assert out[1, 3, 4] > 0.0
    assert np.all(out[0] == 0.0)
    assert np.all(out[2] == 0.0)


def test_intensity_drift_changes_per_view_means() -> None:
    source = _base(1.0)
    out, meta = _assert_common(
        source,
        SimulationArtefacts(intensity_drift_mode="linear", intensity_drift_amplitude=0.2),
    )

    means = out.mean(axis=(1, 2))
    assert np.allclose(means, np.asarray(meta["intensity_drift_factors"]))
    assert means[0] == pytest.approx(0.8)
    assert means[-1] == pytest.approx(1.2)


def test_simulation_artefact_metadata_roundtrips_through_nxtomo(tmp_path) -> None:
    projections = _base(1.0)
    out, artefact_meta = apply_simulation_artefacts(
        projections,
        SimulationArtefacts(dropped_view_fraction=0.4, dropped_view_fill=0.0),
        seed=5,
    )
    path = tmp_path / "artefacts.nxs"
    metadata = NXTomoMetadata(
        thetas_deg=np.linspace(0.0, 180.0, projections.shape[0], endpoint=False),
        grid=Grid(4, 4, 4, 1.0, 1.0, 1.0).to_dict(),
        detector=Detector(6, 4, 1.0, 1.0).to_dict(),
        simulation_artefacts=artefact_meta,
    )

    save_nxtomo(str(path), np.asarray(out), metadata=metadata)

    assert validate_nxtomo(str(path))["issues"] == []
    with h5py.File(path, "r") as handle:
        attrs = handle["entry/processing/tomojax/simulation"].attrs
        persisted = json.loads(attrs["artefacts_json"])
        assert persisted["dropped_view_indices"] == artefact_meta["dropped_view_indices"]

    loaded = load_nxtomo(str(path))
    assert loaded.simulation_artefacts is not None
    assert loaded.simulation_artefacts["dropped_view_count"] == artefact_meta[
        "dropped_view_count"
    ]
