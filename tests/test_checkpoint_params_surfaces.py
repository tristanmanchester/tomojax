from __future__ import annotations

import numpy as np
import pytest

import tomojax.align.api as align_api
from tomojax.align.api import (
    AlignmentCheckpoint,
    AlignmentCheckpointGeometrySnapshot,
    AlignmentCheckpointMetadataInput,
    AlignmentCheckpointProgress,
    AlignmentProjectionIdentity,
    CheckpointError,
    ScheduleResumeState,
    alignment_params_payload,
    build_alignment_checkpoint_metadata_from_input,
    normalize_schedule_resume_state,
    validate_alignment_checkpoint,
)

# check-public-imports: allow-private
from tomojax.cli.align.checkpoint import _schedule_resume_state_from_checkpoint


def _metadata_input() -> AlignmentCheckpointMetadataInput:
    grid = {"nx": 2, "ny": 3, "nz": 4, "vx": 1.0, "vy": 1.0, "vz": 1.0}
    detector = {"nu": 5, "nv": 6, "du": 0.5, "dv": 0.75}
    return AlignmentCheckpointMetadataInput(
        projection=AlignmentProjectionIdentity(shape=(5, 6, 7), dtype="float32"),
        geometry=AlignmentCheckpointGeometrySnapshot(
            geometry_type="parallel",
            geometry_meta={"tilt_deg": 55.0},
            reconstruction_grid=grid,
            detector=detector,
            state_grid=grid,
            state_detector=detector,
        ),
        progress=AlignmentCheckpointProgress(
            levels=[2, 1],
            level_index=0,
            level_factor=2,
            completed_outer_iters_in_level=1,
            global_outer_iters_completed=3,
        ),
        config={"recon_algo": "fista"},
        cli_options={"mode": "rigid"},
    )


def test_checkpoint_metadata_exposes_only_structured_builder() -> None:
    assert hasattr(align_api, "build_alignment_checkpoint_metadata_from_input")
    assert not hasattr(align_api, "build_alignment_checkpoint_metadata")

    metadata = build_alignment_checkpoint_metadata_from_input(_metadata_input())

    assert metadata["checkpoint_kind"] == "tomojax.align.checkpoint"
    assert metadata["projection_shape"] == [5, 6, 7]
    assert metadata["geometry_meta"] == {"tilt_deg": 55.0}


def test_checkpoint_schedule_resume_state_uses_public_schema() -> None:
    schedule_state: ScheduleResumeState = {
        "stage_index": 2,
        "stage_name": "calibrate_geometry",
        "stage_completed": False,
        "completed_outer_iters_in_stage": 7,
    }
    metadata_input = _metadata_input()
    metadata = build_alignment_checkpoint_metadata_from_input(
        AlignmentCheckpointMetadataInput(
            projection=metadata_input.projection,
            geometry=metadata_input.geometry,
            progress=metadata_input.progress,
            config=metadata_input.config,
            cli_options=metadata_input.cli_options,
            schedule_state=schedule_state,
        )
    )

    assert metadata["schedule_state"] == schedule_state
    assert normalize_schedule_resume_state(metadata["schedule_state"]) == schedule_state


def test_checkpoint_resume_rejects_non_mapping_schedule_state() -> None:
    metadata = build_alignment_checkpoint_metadata_from_input(_metadata_input())
    metadata["schedule_state"] = ["stage", "state"]  # type: ignore[typeddict-item]

    with pytest.raises(CheckpointError, match="invalid schedule_state"):
        _schedule_resume_state_from_checkpoint(metadata)


def test_checkpoint_validation_requires_exact_config_defaults() -> None:
    metadata = build_alignment_checkpoint_metadata_from_input(_metadata_input())
    checkpoint = AlignmentCheckpoint(
        x=np.zeros((2, 3, 4), dtype=np.float32),
        params5=np.zeros((5, 5), dtype=np.float32),
        motion_coeffs=None,
        loss_history=[],
        outer_stats=[],
        metadata=metadata,
    )
    expected_metadata = dict(metadata)
    expected_metadata["config"] = {
        "recon_algo": "fista",
        "gauge_fix": "mean_translation",
    }

    with pytest.raises(CheckpointError, match="config"):
        validate_alignment_checkpoint(checkpoint, expected_metadata)


def test_alignment_params_schema_uses_current_identifier() -> None:
    payload = alignment_params_payload(np.zeros((1, 5), dtype=np.float32), du=1.0, dv=1.0)

    assert payload["schema"] == "tomojax.alignment_params"
