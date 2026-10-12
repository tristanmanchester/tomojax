"""Log records describe the transmission the log actually receives."""

from __future__ import annotations

import numpy as np

import tomojax as tj
from tomojax.corrections import Paganin


def _phase(decay_pixels: float) -> Paganin:
    energy, pixel, delta_beta = 20.0, 1e-6, 300.0
    wavelength = 12.398419843320026e-10 / energy
    distance = (decay_pixels * pixel) ** 2 * 4 * np.pi / (wavelength * delta_beta)
    return Paganin(delta_beta, distance, energy, pixel)


def _geometry(columns: int) -> tj.ParallelGeometry:
    return tj.ParallelGeometry(
        tj.Grid(columns, columns, 1, 1.0, 1.0, 1.0),
        tj.Detector(columns, 1, 1.0, 1.0),
        [0.0],
    )


def test_log_records_nonpositive_transmission_created_by_phase_filter() -> None:
    # A resolved but near-opaque edge can undershoot with the sampled spectrum.
    # It is this filtered transmission, not the strictly positive input, that
    # is clamped at the log boundary.
    counts = np.where(np.arange(641) < 320, 0.001, 1.0)[None, None, :].astype(np.float32)
    phase = _phase(0.3)
    filtered = np.asarray(phase.apply(counts))
    assert (counts > 0).all()
    assert (filtered < 0).any()

    scan = tj.Frames(counts, _geometry(641), white_level=1.0).corrected(phase)

    assert scan.corrections[-1].name == "log"
    assert scan.corrections[-1].found["nonpositive_transmission"] == int((filtered <= 0).sum())
    np.testing.assert_allclose(
        scan.projections, -np.log(np.maximum(filtered, 1e-6)), rtol=3e-6, atol=3e-7
    )


def test_log_does_not_count_below_dark_counts_recovered_by_phase_filter() -> None:
    # Noise can put one positive raw count below the dark field. Spatial
    # filtering recovers positive transmission, so the log clamps no pixels.
    counts = np.full((1, 1, 33), 600.0, np.float32)
    counts[0, 0, 16] = 50.0
    darks = np.full((3, 1, 33), 100.0, np.float32)
    flats = np.full((3, 1, 33), 1100.0, np.float32)
    phase = _phase(3.0)
    transmission = (counts - 100.0) / 1000.0
    filtered = np.asarray(phase.apply(transmission))
    assert (transmission < 0).sum() == 1
    assert (filtered > 0).all()

    scan = tj.Frames(counts, _geometry(33), flats=flats, darks=darks).corrected(phase)

    assert scan.corrections[-1].name == "log"
    assert scan.corrections[-1].found["nonpositive_transmission"] == 0
    np.testing.assert_allclose(scan.projections, -np.log(filtered), rtol=3e-6, atol=3e-7)
