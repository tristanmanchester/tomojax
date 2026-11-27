"""Scans whose object extends outside the field of view, through the public API.

A uniform cylinder about the rotation axis, wider than the detector, has exact
line integrals 2 mu sqrt(R^2 - s^2) at distance s from the axis, and the analytic
methods continue rows as such a cylinder: its attenuation inside the field of view
is the analytic answer for each of them.
"""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FourierConfig

MU, RADIUS = 0.03, 1.6  # attenuation per length; cylinder radius over the field-of-view radius
NU, VIEWS = 64, 96


def _columns(nu: int = NU) -> np.ndarray:
    """Detector coordinates spanning the field of view [-1, 1]."""
    return (np.arange(nu) - (nu - 1) / 2) * 2.0 / nu


def _radius() -> np.ndarray:
    x = _columns()
    return np.hypot(x[None, :], x[:, None])


def _parallel(radius: float = RADIUS, *, rows: int = 2) -> tj.Scan:
    pitch = 2.0 / NU
    grid = tj.Grid(NU, NU, rows, pitch, pitch, pitch)
    detector = tj.Detector(NU, rows, pitch, pitch)
    angles = np.linspace(0.0, 180.0, VIEWS, endpoint=False)
    profile = 2 * MU * np.sqrt(np.maximum(radius**2 - _columns() ** 2, 0))
    data = np.broadcast_to(profile, (VIEWS, rows, NU)).astype(np.float32)
    return tj.Scan(data, tj.ParallelGeometry(grid, detector, angles))


def _cone(radius: float = RADIUS) -> tj.Scan:
    """One detector row in the source's plane: each ray's distance from the axis is exact."""
    source_to_axis, magnification = 12.0, 1.5
    pitch = 2.0 / NU
    grid = tj.Grid(NU, NU, 1, pitch, pitch, pitch)
    detector = tj.Detector(NU, 1, pitch * magnification, pitch * magnification)
    angles = np.linspace(0.0, 360.0, 2 * VIEWS, endpoint=False)
    beam = tj.ConeBeam(source_to_axis, source_to_axis * magnification)
    fan = np.arctan(_columns() * magnification / beam.source_to_detector)
    distance = source_to_axis * np.sin(fan)
    profile = 2 * MU * np.sqrt(np.maximum(radius**2 - distance**2, 0))
    data = np.broadcast_to(profile, (len(angles), 1, NU)).astype(np.float32)
    return tj.Scan(data, tj.ConeGeometry(grid, detector, angles, beam))


def _level_and_bowl(volume: np.ndarray) -> tuple[float, float]:
    """Mean over the inner half of the field of view, and the outer ring's excess over the center."""
    r = _radius()
    plane = np.asarray(volume)[np.asarray(volume).shape[0] // 2]
    return float(plane[r < 0.5].mean()), float(
        plane[(r > 0.8) & (r < 0.92)].mean() - plane[r < 0.3].mean()
    )


@pytest.mark.parametrize(
    ("method", "config"),
    [
        ("fbp", None),
        ("fourier", FourierConfig(backend="numpy")),
        ("fourier", FourierConfig(backend="numpy", filter="ramp")),
    ],
)
def test_parallel_methods_recover_a_cylinder_wider_than_the_detector(method, config) -> None:
    # Zero beyond the detector would raise the level by a quarter here and bend it by more.
    level, bowl = _level_and_bowl(tj.reconstruct(_parallel(), method, config=config).volume)
    assert level == pytest.approx(MU, rel=0.03)
    assert abs(bowl) < 0.03 * MU


def test_fdk_recovers_a_cylinder_wider_than_the_detector() -> None:
    level, bowl = _level_and_bowl(tj.reconstruct(_cone(), "fbp").volume)
    assert level == pytest.approx(MU, rel=0.04)
    assert abs(bowl) < 0.04 * MU


@pytest.mark.parametrize("method", ["fbp", "fourier"])
def test_a_contained_object_reconstructs_as_on_a_wider_detector_of_zeros(method) -> None:
    # Rows that end at zero continue as zero: more zero columns change nothing.
    scan = _parallel(0.7, rows=1)
    wide = NU + 32
    pitch = scan.detector.du
    padded = np.pad(np.asarray(scan.projections), ((0, 0), (0, 0), (16, 16)))
    geometry = tj.ParallelGeometry(scan.grid, tj.Detector(wide, 1, pitch, pitch), scan.angles)
    config = FourierConfig(backend="numpy") if method == "fourier" else None

    narrow = np.asarray(tj.reconstruct(scan, method, config=config).volume)
    wider = np.asarray(tj.reconstruct(tj.Scan(padded, geometry), method, config=config).volume)

    assert np.linalg.norm(narrow - wider) < 2e-3 * np.linalg.norm(wider)
    level, _ = _level_and_bowl(narrow)
    assert level == pytest.approx(MU, rel=0.01)


def test_the_record_says_how_much_lies_beyond_the_detector() -> None:
    record = tj.reconstruct(_parallel(), "fbp").info["truncation"]
    edge = (NU - 1) / NU  # distance of the outermost column's center from the axis
    s = np.linspace(-RADIUS, RADIUS, 200_001)
    profile = 2 * MU * np.sqrt(np.maximum(RADIUS**2 - s**2, 0))
    inside = np.abs(s) <= 1.0
    step = s[1] - s[0]
    # Independent of the library: numerical integrals of the cylinder's exact projection.
    beyond = profile[~inside].sum() / profile.sum()
    on_axis = -(profile[~inside] / s[~inside] ** 2).sum() * step / (2 * np.pi)
    at_edge = 2 * MU * np.sqrt(RADIUS**2 - edge**2)

    assert record["radius"] == pytest.approx(RADIUS, rel=1e-3)
    assert record["field_of_view_radius"] == pytest.approx(edge)
    assert record["outside_fraction"] == pytest.approx(beyond, rel=0.02)
    assert record["interior_shift"] == pytest.approx(on_axis, rel=0.03)
    assert record["edge_ratio"] == pytest.approx(at_edge / profile[inside].mean(), rel=0.02)


def test_the_record_of_a_contained_object_is_zero() -> None:
    record = tj.reconstruct(_parallel(0.7), "fbp").info["truncation"]
    assert record["edge_ratio"] == 0 and record["outside_fraction"] == 0
    assert record["interior_shift"] == 0


def test_every_method_carries_the_record() -> None:
    # The methods that fit the grid take the data as measured, and say what the rows say.
    scan = _parallel()
    expected = tj.reconstruct(scan, "fbp").info["truncation"]
    assert tj.reconstruct(scan, "cgls", iterations=2).info["truncation"] == expected
    assert tj.reconstruct(_cone(), "fbp").info["truncation"]["radius"] == pytest.approx(
        RADIUS, rel=1e-3
    )


@pytest.mark.parametrize(("radius", "beyond"), [(RADIUS, True), (0.7, False)])
def test_the_command_line_and_the_mcp_summary_say_what_lies_beyond_the_detector(
    radius, beyond, tmp_path, capsys
) -> None:
    from tomojax.cli.main import main
    from tomojax.cli.mcp.tools import summary_text

    tj.save(tmp_path / "scan.nxs", _parallel(radius))
    arguments = ["reconstruct", str(tmp_path / "scan.nxs"), "-o", str(tmp_path / "out.nxs")]
    assert main([*arguments, "--method", "fbp"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = tj.load_reconstruction(tmp_path / "out.nxs").info["truncation"]
    summary = summary_text({"kind": "volume", "truncation": record, "wall_seconds": 1.0})

    share = f"{record['outside_fraction']:.0%} of attenuation beyond the detector"
    assert (share in line) is beyond
    assert (share in summary) is beyond
    assert ("beyond the detector" in line + summary) is beyond
