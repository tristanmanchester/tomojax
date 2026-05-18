from __future__ import annotations

import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.align.io import checkpoint as checkpoint_io

# check-public-imports: allow-private
from tomojax.align.io.checkpoint import (
    AlignmentCheckpoint,
    AlignmentCheckpointGeometrySnapshot,
    AlignmentCheckpointMetadataInput,
    AlignmentCheckpointProgress,
    AlignmentProjectionIdentity,
    CheckpointError,
    build_alignment_checkpoint_metadata_from_input,
    validate_alignment_checkpoint,
)

# check-public-imports: allow-private
from tomojax.align.io.params_export import alignment_params_payload


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
    assert hasattr(checkpoint_io, "build_alignment_checkpoint_metadata_from_input")
    assert not hasattr(checkpoint_io, "build_alignment_checkpoint_metadata")

    metadata = build_alignment_checkpoint_metadata_from_input(_metadata_input())

    assert metadata["checkpoint_kind"] == "tomojax.align.checkpoint"
    assert metadata["projection_shape"] == [5, 6, 7]
    assert metadata["geometry_meta"] == {"tilt_deg": 55.0}


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
