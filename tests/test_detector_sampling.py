"""Known post-log block means survive scan operations and reach the joint fit."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax import _smooth_alignment
from tomojax.io import ProjectionDataset


def _scan(*, metadata=None, cone=False):
    grid = tj.Grid(5, 4, 3, 0.7, 0.8, 0.9)
    detector = tj.Detector(18, 14, 0.25, 0.4, (0.3, -0.6))
    angles = np.arange(4, dtype=np.float32) * 45
    geometry = (
        tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(20, 50))
        if cone
        else tj.ParallelGeometry(grid, detector, angles)
    )
    projections = np.arange(4 * 14 * 18, dtype=np.float32).reshape(4, 14, 18) / 100
    source = (
        None
        if metadata is None
        else ProjectionDataset(
            projections,
            angles,
            detector=detector,
            grid=grid,
            geometry_type="cone" if cone else "parallel",
            geometry_metadata=metadata,
        )
    )
    return tj.Scan(projections, geometry, source=source)


def _sampling(u, v):
    return {"kind": "line_integral_block_mean", "factors_uv": [u, v]}


@pytest.mark.parametrize("with_source", [False, True])
def test_repeated_binning_composes_provenance_without_mutating_the_source(with_source):
    metadata = {"acquisition": {"camera_binning": [8, 8]}, "note": "preserve me"}
    original = _scan(metadata=metadata if with_source else None)
    before = deepcopy(metadata)
    first = original.binned(2)
    second = first.binned(3)
    assert first.source.geometry_metadata["detector_sampling"] == _sampling(2, 2)
    assert second.source.geometry_metadata["detector_sampling"] == _sampling(6, 6)
    assert metadata == before
    assert first.source is not original.source and second.source is not first.source
    assert first.source.detector == first.detector
    assert second.source.detector == second.detector
    np.testing.assert_array_equal(second.source.projections, second.projections)
    assert (second.detector.du / 6, second.detector.dv / 6) == pytest.approx(
        (original.detector.du, original.detector.dv)
    )
    expected = original.projections[:, :12, :18].reshape(4, 2, 6, 3, 6).mean((2, 4))
    np.testing.assert_allclose(second.projections, expected, atol=1e-6)


@pytest.mark.parametrize("suffix", [".npz", ".nxs"])
def test_sampling_survives_crop_selection_and_file_roundtrip(tmp_path, suffix):
    source = _scan(metadata={"detector_sampling": _sampling(2, 3), "note": "keep"})
    scan = source.binned(2).cropped(slice(1, 6), slice(2, 8)).selected([0, 2, 3])
    path = tmp_path / f"sampling{suffix}"
    tj.save(path, scan)
    loaded = tj.load(path)
    assert loaded.source.geometry_metadata["detector_sampling"] == _sampling(4, 6)
    assert loaded.source.geometry_metadata["note"] == "keep"
    assert loaded.detector == scan.detector
    np.testing.assert_array_equal(loaded.projections, scan.projections)
    np.testing.assert_array_equal(loaded.angles, scan.angles)
    assert source.source.geometry_metadata["detector_sampling"] == _sampling(2, 3)


@pytest.mark.parametrize("previous", [None, (2, 3)])
def test_corrected_frames_record_post_log_means_and_compose_existing_factors(previous):
    metadata = {"acquisition": {"camera_binning": [4, 4]}}
    if previous is not None:
        metadata["detector_sampling"] = _sampling(*previous)
    template = _scan(metadata=metadata)
    # Unequal transmission makes mean(-log transmission) differ from -log(mean transmission).
    counts = 1000 * np.exp(-0.2 - 0.03 * template.projections)
    source = replace(template.source, projections=counts)
    before = deepcopy(source.geometry_metadata)
    frames = tj.Frames(counts, template.geometry, white_level=1000, source=source)
    corrected = frames.corrected(binning=2, batch_views=2)
    expected = -np.log(counts / 1000)
    expected = expected.reshape(4, 7, 2, 9, 2).mean((2, 4))
    np.testing.assert_allclose(corrected.projections, expected, rtol=2e-6, atol=1e-7)
    factors = (1, 1) if previous is None else previous
    assert corrected.source.geometry_metadata["detector_sampling"] == _sampling(
        2 * factors[0], 2 * factors[1]
    )
    assert corrected.source is not source and source.geometry_metadata == before
    assert corrected.source.detector == corrected.detector
    np.testing.assert_array_equal(corrected.source.projections, corrected.projections)


@pytest.mark.parametrize("explicit", [False, True])
def test_public_alignment_passes_only_explicit_sampling_and_derived_native_pitch(
    monkeypatch, explicit
):
    metadata = {"acquisition": {"camera_binning": [7, 9], "native_pixel_size": [0.001, 0.002]}}
    if explicit:
        metadata["detector_sampling"] = _sampling(3, 2)
    scan = _scan(metadata=metadata)
    before = deepcopy(metadata)
    expected = (3, 2) if explicit else (1, 1)

    def fit(
        projections,
        nominal,
        grid,
        detector,
        *,
        keep_grid,
        ray_binning,
        native_pixel_size,
        fit,
        frames,
        detector_roll_deg,
    ):
        assert projections is scan.projections and nominal.shape == (4, 4, 4)
        assert ray_binning == expected
        assert native_pixel_size == (detector.du / expected[0], detector.dv / expected[1])
        assert not keep_grid
        assert fit == "translations" and frames is None and detector_roll_deg == 0
        record = {
            "corrections": {},
            "fit": ["translations"],
            "detector_binning": 1,
            "levels": [{"status": "budget_exhausted"}],
            "refinement": {"status": "converged"},
        }
        return np.zeros((grid.nz, grid.ny, grid.nx)), grid, np.zeros((4, 6)), record

    monkeypatch.setattr(_smooth_alignment, "fit_poses", fit)
    aligned = tj.align(scan)
    assert aligned.info["levels"][0]["status"] == "budget_exhausted"
    assert scan.source.geometry_metadata == before
    if explicit:
        assert aligned.scan.source.geometry_metadata["detector_sampling"] == _sampling(3, 2)


@pytest.mark.parametrize(
    "description",
    [
        None,
        {},
        "line_integral_block_mean",
        {"kind": "intensity_block_mean", "factors_uv": [2, 2]},
        {"kind": "line_integral_block_mean"},
        _sampling(2, 0),
        _sampling(2, -1),
        _sampling(2, 2.0),
        _sampling(2, True),
        {"kind": "line_integral_block_mean", "factors_uv": [2]},
    ],
)
def test_explicit_bad_sampling_fails_before_fitting_or_binning(monkeypatch, description):
    scan = _scan(metadata={"detector_sampling": description})

    def unexpected(*args, **kwargs):
        pytest.fail("malformed provenance reached the fitter")

    monkeypatch.setattr(_smooth_alignment, "fit_poses", unexpected)
    with pytest.raises(ValueError, match="detector_sampling"):
        tj.align(scan)
    with pytest.raises(ValueError, match="detector_sampling"):
        scan.binned(2)


def test_combine_preserves_common_sampling_and_rejects_heterogeneous_sampling():
    first = _scan(metadata={"detector_sampling": _sampling(2, 3)}, cone=True)
    second = _scan(metadata={"detector_sampling": _sampling(2, 3)}, cone=True)
    combined = tj.Scan.combine([first, second])
    assert combined.source.geometry_metadata["detector_sampling"] == _sampling(2, 3)
    assert combined.source is not first.source and combined.source is not second.source
    with pytest.raises(ValueError, match="identical detector_sampling"):
        tj.Scan.combine([first, _scan(cone=True)])
    with pytest.raises(ValueError, match="identical detector_sampling"):
        tj.Scan.combine([first, _scan(metadata={"detector_sampling": _sampling(3, 2)}, cone=True)])
    with pytest.raises(ValueError, match="detector_sampling"):
        tj.Scan.combine(
            [first, _scan(metadata={"detector_sampling": {"kind": "unknown"}}, cone=True)]
        )
