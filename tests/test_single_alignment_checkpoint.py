"""Single-resolution checkpoints retain the ungauged iterate and completion."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment import api as alignment

# check-public-imports: allow-private
from tomojax.alignment.io.checkpoint import CheckpointError

# check-public-imports: allow-private
from tomojax.alignment.io.resume import AlignmentRun, alignment_checkpointing


@pytest.mark.parametrize("early_stop", [False, True])
def test_seeded_alignment_resumes_and_keeps_a_completed_run(tmp_path, early_stop):
    grid, detector = tj.Grid(4, 5, 3, 1, 1, 1), tj.Detector(9, 7, 1, 1)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(8) * 45, tilt_deg=32)
    truth = np.random.default_rng(43).uniform(0, 1, (3, 5, 4)).astype(np.float32)
    data = tj.project(geometry, truth)
    config = alignment.AlignConfig(
        quality="reference",
        backend="jax",
        gather_dtype="fp32",
        regulariser="tv",
        outer_iterations=5 if early_stop else 2,
        iterations=2,
        optimise_dofs=("dx", "dz"),
        gn_coupling="joint",
        gn_joint_iterations=3,
        tv_weight=0,
        early_stop=early_stop,
        early_stop_rel_impr=1e6,
        early_stop_patience=2,
    )
    run = AlignmentRun(
        projections=data,
        geometry=geometry,
        config=config,
        mode="pose",
        levels=None,
    )
    path = tmp_path / "native.npz"

    def solve(resume=None, callback=None):
        return alignment.align(
            geometry,
            grid,
            detector,
            data,
            config=config,
            init_x=truth * 0.7,
            resume_state=resume,
            checkpoint_callback=callback,
        )

    expected = solve()
    resume, write = alignment_checkpointing(path, run)
    assert resume is None and write is not None

    def interrupt(state):
        write(state)
        raise RuntimeError("interrupted after saved outer")

    with pytest.raises(RuntimeError, match="interrupted after saved outer"):
        solve(callback=interrupt)
    resume, write = alignment_checkpointing(path, run)
    assert isinstance(resume, alignment.AlignResumeState) and not resume.run_complete
    actual = solve(resume, write)
    resume, _ = alignment_checkpointing(path, run)
    assert resume.run_complete
    assert (resume.start_outer_iter < config.outer_iterations) == early_stop
    for result in (actual, solve(resume)):
        np.testing.assert_allclose(result[0], expected[0], atol=2e-5)
        np.testing.assert_allclose(result[1], expected[1], atol=2e-5)
        np.testing.assert_allclose(result[2]["loss"], expected[2]["loss"], rtol=2e-5)
    saved = path.read_bytes()
    other = replace(run, config=replace(config, gn_damping=config.gn_damping * 2))
    with pytest.raises(CheckpointError, match="cannot resume this alignment"):
        alignment_checkpointing(path, other)
    assert path.read_bytes() == saved
