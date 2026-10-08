from __future__ import annotations

import numpy as np
import pytest

import tomojax.alignment.api as align_api
from tomojax.alignment.api import (
    AlignConfig,
    AlignmentCheckpoint,
    AlignmentRun,
    AlignMultiresResumeState,
    CheckpointError,
    alignment_checkpoint_metadata,
    alignment_params_payload,
    validate_alignment_checkpoint,
)

# check-public-imports: allow-private
from tomojax.alignment.io.resume import _schedule_resume_state_from_checkpoint
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry

pytestmark = pytest.mark.surface


def _run(**changes: object) -> AlignmentRun:
    grid = Grid(2, 3, 4, 1.0, 1.0, 1.0)
    detector = Detector(7, 6, 0.5, 0.75)
    fields: dict[str, object] = {
        "projections": np.zeros((5, 6, 7), np.float32),
        "geometry": ParallelGeometry(grid, detector, np.linspace(0, 180, 5)),
        "config": AlignConfig(reconstruction="fista"),
        "mode": "pose",
        "levels": (2, 1),
    }
    return AlignmentRun(**{**fields, **changes})  # pyright: ignore[reportArgumentType]


def _state(**changes: object) -> AlignMultiresResumeState:
    fields: dict[str, object] = {
        "x": np.zeros((2, 3, 4), np.float32),
        "pose_params": np.zeros((5, 6), np.float32),
        "level_index": 1,
        "level_factor": 1,
        "completed_outer_iterations_in_level": 1,
        "global_outer_iterations_completed": 3,
    }
    return AlignMultiresResumeState(**{**fields, **changes})  # pyright: ignore[reportArgumentType]


def test_checkpoint_metadata_records_the_run_and_its_progress() -> None:
    assert not hasattr(align_api, "build_alignment_checkpoint_metadata_from_input")

    metadata = alignment_checkpoint_metadata(_run(), _state(level_index=0, level_factor=2))

    assert metadata["checkpoint_kind"] == "tomojax.alignment.checkpoint"
    assert metadata["schema_version"] == 4
    assert metadata["projection_shape"] == [5, 6, 7]
    assert metadata["reconstruction_grid"]["nz"] == 4 and metadata["detector"]["nu"] == 7
    assert metadata["mode"] == "pose" and metadata["levels"] == [2, 1]
    assert metadata["state_grid"]["nx"] == 1  # the factor-2 level's grid
    assert "cli_options" not in metadata


def test_the_fingerprint_tells_scans_and_geometries_apart() -> None:
    run = _run()
    assert _run().fingerprint == run.fingerprint
    other_data = _run(projections=np.ones((5, 6, 7), np.float32))
    other_angles = _run(geometry=ParallelGeometry(run.grid, run.detector, np.linspace(0, 90, 5)))
    tilted = _run(geometry=LaminographyGeometry(run.grid, run.detector, np.linspace(0, 180, 5)))
    fingerprints = {r.fingerprint for r in (run, other_data, other_angles, tilted)}
    assert len(fingerprints) == 4


def test_checkpoint_schedule_resume_state_uses_public_schema() -> None:
    state = _state(stage_index=2, stage_name="calibrate_geometry")
    state.completed_outer_iterations_in_stage = 7
    metadata = alignment_checkpoint_metadata(_run(), state)

    assert metadata["schedule_state"] == {
        "stage_index": 2,
        "stage_name": "calibrate_geometry",
        "stage_completed": False,
        "completed_outer_iterations_in_stage": 7,
    }
    assert _schedule_resume_state_from_checkpoint(metadata) == metadata["schedule_state"]


def test_checkpoint_resume_rejects_non_mapping_schedule_state() -> None:
    metadata = alignment_checkpoint_metadata(_run(), _state())
    metadata["schedule_state"] = ["stage", "state"]  # type: ignore[typeddict-item]

    with pytest.raises(CheckpointError, match="invalid schedule_state"):
        _schedule_resume_state_from_checkpoint(metadata)


def test_checkpoint_validation_requires_exact_config_defaults() -> None:
    metadata = alignment_checkpoint_metadata(_run(), _state())
    checkpoint = AlignmentCheckpoint(
        x=np.zeros((2, 3, 4), dtype=np.float32),
        pose_params=np.zeros((5, 5), dtype=np.float32),
        motion_coeffs=None,
        loss_history=[],
        outer_stats=[],
        metadata=metadata,
    )
    expected_metadata = dict(metadata)
    expected_metadata["config"] = {**metadata["config"], "outer_iterations": 3}

    with pytest.raises(CheckpointError, match=r"config differs in outer_iterations \(checkpoint"):
        validate_alignment_checkpoint(checkpoint, expected_metadata)


def test_alignment_params_schema_uses_current_identifier() -> None:
    payload = alignment_params_payload(np.zeros((1, 5), dtype=np.float32), du=1.0, dv=1.0)

    assert payload["schema"] == "tomojax.alignment_params"


def test_checkpoints_written_before_the_settings_rename_do_not_resume() -> None:
    metadata = alignment_checkpoint_metadata(_run(), _state())
    checkpoint = AlignmentCheckpoint(
        x=np.zeros((2, 3, 4), dtype=np.float32),
        pose_params=np.zeros((5, 5), dtype=np.float32),
        motion_coeffs=None,
        loss_history=[],
        outer_stats=[],
        metadata={**metadata, "schema_version": 2},
    )
    with pytest.raises(CheckpointError, match="schema version 2 predates this version"):
        validate_alignment_checkpoint(checkpoint, metadata)
    checkpoint.metadata["schema_version"] = 99
    with pytest.raises(CheckpointError, match="unsupported schema version 99"):
        validate_alignment_checkpoint(checkpoint, metadata)


@pytest.mark.parametrize("multires", [False, True])
def test_legacy_checkpoint_sampling_is_rejected_only_for_multires(multires):
    metadata = alignment_checkpoint_metadata(_run(), _state())
    if not multires:
        metadata["levels"] = None
        metadata.pop("multires_geometry_version")
    checkpoint = AlignmentCheckpoint(
        x=np.zeros((2, 3, 4), dtype=np.float32),
        pose_params=np.zeros((5, 5), dtype=np.float32),
        motion_coeffs=None,
        loss_history=[],
        outer_stats=[],
        metadata=dict(metadata),
    )
    validate_alignment_checkpoint(checkpoint, metadata)
    checkpoint.metadata.pop("multires_geometry_version", None)
    if multires:
        with pytest.raises(CheckpointError, match="multires geometry version"):
            validate_alignment_checkpoint(checkpoint, metadata)
        # Supplying old metadata as the expected request cannot bypass the
        # current physical sampling convention.
        with pytest.raises(CheckpointError, match="multires geometry version"):
            validate_alignment_checkpoint(checkpoint, checkpoint.metadata)
    else:
        validate_alignment_checkpoint(checkpoint, metadata)
