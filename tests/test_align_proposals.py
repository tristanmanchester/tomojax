from __future__ import annotations

import jax.numpy as jnp

from tomojax.align.objectives.loss_adapters import build_loss_adapter
from tomojax.align.objectives.loss_specs import L2LossSpec
from tomojax.align.proposals import ProposalCandidate, score_pose_stack_candidates
from tomojax.core.geometry import Detector, Grid


def test_score_pose_stack_candidates_selects_lowest_loss(monkeypatch):
    calls = []

    def fake_score(**kwargs):
        calls.append(kwargs)
        return jnp.asarray(10.0 - float(kwargs["pose_stack"][0, 0, 3]), dtype=jnp.float32)

    monkeypatch.setattr("tomojax.align.proposals.project_and_score_stack", fake_score)
    monkeypatch.setattr(
        "tomojax.align.proposals.alignment_projector_backend_provenance",
        lambda **kwargs: type(
            "P",
            (),
            {"to_dict": lambda self: {"actual_backend": kwargs["projector_backend"]}},
        )(),
    )
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0)
    pose0 = jnp.eye(4, dtype=jnp.float32)[None, :, :]
    pose1 = pose0.at[0, 0, 3].set(3.0)
    targets = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    adapter = build_loss_adapter(L2LossSpec(), targets)

    result = score_pose_stack_candidates(
        candidates=(
            ProposalCandidate("baseline", pose0, {"kind": "baseline"}),
            ProposalCandidate("shifted", pose1, {"kind": "shifted"}),
        ),
        grid=grid,
        detector=detector,
        volume=jnp.zeros((4, 4, 4), dtype=jnp.float32),
        det_grid=(jnp.zeros((4, 4), dtype=jnp.float32), jnp.zeros((4, 4), dtype=jnp.float32)),
        targets=targets,
        loss_adapter=adapter,
        projector_backend="pallas",
    )

    assert result.best_index == 1
    assert result.best_name == "shifted"
    assert result.improved is True
    assert result.values == (10.0, 7.0)
    assert result.backend_provenance["actual_backend"] == "pallas"
    assert calls[0]["require_differentiable_projector"] is False
    assert result.to_dict()["candidate_metadata"][1]["kind"] == "shifted"


def test_score_pose_stack_candidates_rejects_empty_candidates():
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    targets = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    adapter = build_loss_adapter(L2LossSpec(), targets)

    try:
        score_pose_stack_candidates(
            candidates=(),
            grid=grid,
            detector=detector,
            volume=jnp.zeros((2, 2, 2), dtype=jnp.float32),
            det_grid=(
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.float32),
            ),
            targets=targets,
            loss_adapter=adapter,
        )
    except ValueError as exc:
        assert "at least one candidate" in str(exc)
    else:
        raise AssertionError("empty candidates should fail")
