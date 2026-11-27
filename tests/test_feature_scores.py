"""The Skill's feature scores measure known contrast, noise and edge blur."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy import ndimage

import tomojax

pytestmark = pytest.mark.surface

SCRIPT = Path(tomojax.__file__).parent / "skill" / "tomojax" / "scripts" / "feature_scores.py"
scores = runpy.run_path(str(SCRIPT))["scores"]


def ball(radius=10, size=48):
    z, y, x = np.ogrid[:size, :size, :size]
    c = (size - 1) / 2
    return (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2


def shell(feature, width=3):
    grown = ndimage.binary_dilation(feature, iterations=width)
    return grown & ~ndimage.binary_dilation(feature)  # leave a voxel between them


def test_contrast_and_noise_follow_the_noise_added():
    feature = ball()
    rng = np.random.default_rng(1)
    for sigma in (0.05, 0.2):
        volume = feature.astype(float) + rng.normal(0, sigma, feature.shape)
        found = scores(volume, feature, shell(feature))
        assert found["noise"] == pytest.approx(sigma, rel=0.05)
        assert found["contrast_to_noise"] == pytest.approx(1 / sigma, rel=0.05)


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
def test_edge_width_of_a_blurred_step_is_its_10_90_rise(sigma):
    feature = ball(radius=14, size=56)
    volume = ndimage.gaussian_filter(feature.astype(float), sigma)
    # A Gaussian blur's 10-90% rise is 2 * 1.2816 sigma.
    assert scores(volume, feature, shell(feature))["edge_width"] == pytest.approx(
        2.563 * sigma, rel=0.12
    )


def test_a_sharp_step_measures_under_a_voxel_and_the_command_line_agrees(tmp_path):
    feature = ball()
    surround = shell(feature)
    volume = feature + np.random.default_rng(2).normal(0, 0.01, feature.shape)
    direct = scores(volume, feature, surround)
    assert direct["edge_width"] < 1
    for name, array in (("v", volume), ("f", feature), ("s", surround)):
        np.save(tmp_path / f"{name}.npy", array)
    # A copied skill can score bare .npy files in a NumPy/SciPy environment.
    (tmp_path / "tomojax.py").write_text('raise ImportError("standalone scoring")\n')
    printed = subprocess.run(
        [sys.executable, str(SCRIPT), *(str(tmp_path / f"{n}.npy") for n in "vfs")],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert json.loads(printed) == pytest.approx(direct)
