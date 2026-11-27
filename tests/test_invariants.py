"""Guarantees the source once stated only in comments, checked through the public API."""

from __future__ import annotations

import importlib
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.io import load_dataset, save_dataset
from tomojax.recon import CGLSConfig, cgls

from ._helpers import make_projection_dataset, write_raw_nxtomo

if TYPE_CHECKING:
    from pathlib import Path


def test_frames_opened_by_a_relative_path_still_read_after_a_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    monkeypatch.chdir(tmp_path)
    frames = tj.load_frames("raw.nxs")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    # Frames are read lazily. Dark 1 and flat 11 leave samples 5 and 9 at 4/10 and 8/10.
    expected = -np.log(np.array([0.4, 0.8]))[:, None, None] * np.ones((2, 2, 2))
    np.testing.assert_allclose(frames.corrected().projections, expected, rtol=1e-6)


@pytest.mark.parametrize(("geometry_type", "axis"), [("parallel", [0, 0, 1]), ("cone", None)])
def test_nxtomo_records_a_rotation_axis_only_when_the_geometry_determines_it(
    tmp_path: Path, geometry_type: str, axis: list[float] | None
) -> None:
    path = tmp_path / "scan.nxs"
    # A cone scan without a tilt or an axis vector leaves the rotation axis unknown.
    save_dataset(path, make_projection_dataset(geometry_type=geometry_type))
    with h5py.File(path) as file:
        rotation = file["entry/sample/rotation_angle"]
        recorded = rotation.attrs.get("vector")
        if axis is None:
            assert recorded is None
        else:
            assert rotation.attrs["transformation_type"] == "rotation"
            np.testing.assert_array_equal(recorded, axis)
    assert load_dataset(path).geometry_inputs()["geometry_type"] == geometry_type


def test_importing_a_solver_module_keeps_the_solver_of_the_same_name() -> None:
    recon = importlib.import_module("tomojax.recon")
    shared = [name for name in recon.__all__ if find_spec(f"tomojax.recon.{name}") is not None]
    assert {"fbp", "fdk", "sirt", "cgls"} <= set(shared)
    for name in shared:
        importlib.import_module(f"tomojax.recon.{name}")
        solver = getattr(recon, name)
        assert callable(solver) and not isinstance(solver, ModuleType), name


@pytest.mark.numerical
@pytest.mark.parametrize(("tolerance", "absolute_tolerance"), [(0.0, 0.0), (1e-3, 0.05)])
def test_cgls_judges_convergence_by_the_requested_tolerance(
    tolerance: float, absolute_tolerance: float
) -> None:
    grid = Grid(3, 2, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 4, 0.7, 1.2, (0.17, -0.23))
    geometry = LaminographyGeometry(grid, detector, [13, 49, 83, 122, 161], tilt=23)
    # Damped and inconsistent, so the normal residual stays above zero: a zero
    # tolerance can only end at the FP32 roundoff limit, never as converged.
    data = np.random.default_rng(842).normal(size=(5, 4, 5)).astype(np.float32)
    config = CGLSConfig(
        iterations=160,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        damping=0.7,
        backend="jax",
    )
    _, info = cgls(geometry, grid, detector, data, config=config)
    requested = absolute_tolerance + tolerance * info["initial_normal_residual_norm"]
    assert info["normal_residual_tolerance"] == pytest.approx(requested, rel=1e-6)
    met = info["normal_residual_norm"] <= requested
    assert info["converged"] is met
    assert info["termination"] == ("converged" if met else "roundoff_limit")
    assert met is (tolerance > 0)
