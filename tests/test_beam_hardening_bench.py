"""Check the beam hardening bench's case split, geometry and scores without gVXR."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench" / "beam_hardening"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(BENCH))
    try:
        yield {
            name: importlib.import_module(name)
            for name in ("cases", "score", "simulate", "donoharm", "run")
        }
    finally:
        sys.path.remove(str(BENCH))


def test_split_is_deterministic_and_holds_out_titanium_and_high_voltage(bench):
    cases = bench["cases"]
    for index in range(12):
        for geometry in ("cone", "lamino"):
            dev, held = (cases.sample(split, index, geometry) for split in ("dev", "heldout"))
            assert dev == cases.sample("dev", index, geometry)
            assert not cases.held_out(dev) and cases.held_out(held)
            for case in (dev, held):
                for shape in case["shapes"]:
                    if "parent" in shape:  # rods lie inside their parent
                        parent = case["shapes"][shape["parent"]]
                        gap = np.hypot(*np.subtract(shape["centre"], parent["centre"]))
                        assert gap + shape["radius"] < parent["radius"]


def test_cylinder_chords_match_the_disc_geometry(bench):
    end = np.array([[0.0, 100.0, 0.0], [3.0, 100.0, 0.0], [0.0, 100.0, 50.0], [20.0, 100.0, 0.0]])
    start = np.array([0.0, -100.0, 0.0])
    chords = bench["simulate"].cylinder_chords(start, end, [0.0, 0.0], 3.0, None)
    # Through the axis; a ray 1.5 mm from it; an oblique ray; a miss.
    np.testing.assert_allclose(chords[0], 6.0)
    np.testing.assert_allclose(
        chords[1], 2 * np.sqrt(9 - (300 / np.hypot(200, 3)) ** 2), rtol=1e-12
    )
    np.testing.assert_allclose(chords[2], 6.0 * np.hypot(200, 50) / 200)
    assert chords[3] == 0.0
    # A plate 1 mm thick, crossed at 45 degrees, flat-on, edge-on inside and outside its faces.
    start = np.array(
        [[-10.0, -10.0, -10.0], [0.0, 0.0, -10.0], [-10.0, 0.0, 0.2], [-10.0, 0.0, 0.7]]
    )
    end = start + np.array(
        [[20.0, 20.0, 20.0], [0.0, 0.0, 20.0], [20.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    )
    plate = bench["simulate"].cylinder_chords(start, end, [0.0, 0.0], 5.0, 0.5)
    np.testing.assert_allclose(plate, [np.sqrt(3), 1.0, 10.0, 0.0], rtol=1e-12)


def _phantom(bench, values, rod=2.0):
    case = {
        "shapes": [
            {"material": "water", "centre": [0.0, 0.0], "radius": 12.0},
            {"material": "iron", "centre": [-5.0, 0.0], "radius": rod, "parent": 0},
            {"material": "iron", "centre": [5.0, 0.0], "radius": rod, "parent": 0},
        ]
    }
    fractions = bench["score"].fractions(case)
    image = sum(f * values[s["material"]] for f, s in zip(fractions, case["shapes"], strict=True))
    return case, np.repeat(image[None], 4, axis=0)


def test_true_image_scores_zero_and_errors_are_measured(bench):
    score, truth = bench["score"], {"water": 0.02, "iron": 0.5}
    case, volume = _phantom(bench, truth)
    exact = score.score_volume(volume, case, truth)
    assert all(abs(v["bias"]) < 1e-6 for v in exact["mu"].values())
    assert abs(exact["cupping"]) < 1e-6 and abs(exact["dark_band"]) < 1e-6
    assert all(abs(r) < 0.01 for r in exact["radius"].values())
    x, y = score.coordinates(case)
    owner = volume[0] == truth["water"]
    cupped = volume - 0.002 * (1 - (x**2 + y**2) / 144)[None] * owner[None]
    streak = cupped - 0.004 * ((np.abs(y) < 1) & (np.abs(x) < 3))[None] * owner[None]
    measured = score.score_volume(streak, case, truth)
    assert measured["cupping"] > 0.05 and measured["mu"]["water"]["bias"] < 0
    assert -0.25 < measured["dark_band"] < -0.15  # the streak is 20 % of water
    grown = score.score_volume(_phantom(bench, truth, rod=2.1)[1], case, truth)
    assert abs(grown["radius"]["1:iron"] - 0.1) < 0.01


def test_line_integral_bias_ignores_noise(bench):
    rng = np.random.default_rng(0)
    paths = {"water": rng.uniform(0, 30, (50, 4, 40))}
    truth = 0.02 * paths["water"]
    noisy = truth + rng.normal(0, 0.02, truth.shape)
    result = bench["score"].score_line_integrals(noisy, paths, {"water": 0.02})
    assert result["all"] < 0.01 < result["rms"]
    hardened = bench["score"].score_line_integrals(
        truth * (1 - 0.1 * truth), paths, {"water": 0.02}
    )
    assert hardened["all"] > 0.02
    scaled = bench["score"].score_line_integrals(1.3 * truth, paths, {"water": 0.02})
    assert abs(scaled["all"] - 0.3) < 1e-9 and scaled["shape"] < 1e-9  # a gauge, not a shape


def test_curve_finds_nonlinearity(bench, tmp_path):
    q = np.linspace(0.0, 3.0, 10000)
    assert np.abs(bench["donoharm"].curve(q, 2 * q)[1]).max() < 1e-12
    centres, deviation, _ = bench["donoharm"].curve(q, q + 0.1 * q**2)
    assert np.abs(deviation).max() > 0.02
    pytest.importorskip("matplotlib")
    bench["donoharm"].plots.curves(tmp_path / "c.png", {"all": (centres, deviation)}, 0.005, "t")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_gauges_read_the_spectrum(bench):
    # Two energies with attenuation 0.4 and 0.1 mm^-1, equal detected weight.
    arrays = {"energies": np.array([30.0, 60.0]), "weights": np.array([0.5, 0.5])}
    arrays["mu_grid"] = np.interp(np.arange(1.0, 301.0), [30.0, 60.0], [0.4, 0.1])[None]
    gauge = bench["score"].gauge_mu
    assert gauge(["m"], arrays, ("slope",))["m"] == pytest.approx(0.25)
    assert gauge(["m"], arrays, ("energy", 60.0))["m"] == pytest.approx(0.1)


def test_laminography_images_are_scored_against_the_exact_control(bench):
    record = {"exact": {"mu": {"glass": {"bias": 0.067}}, "cupping": 0.02}}
    record["spectrum"] = {"mu": {"glass": {"bias": 0.067}}, "cupping": 0.05}
    bench["run"]._against_exact(record)
    assert record["spectrum"]["mu"]["glass"]["bias"] == pytest.approx(0.0)
    assert record["spectrum"]["cupping"] == pytest.approx(0.03)
    assert record["spectrum"]["vs_truth"] == {"mu": {"glass": 0.067}, "cupping": 0.05}
