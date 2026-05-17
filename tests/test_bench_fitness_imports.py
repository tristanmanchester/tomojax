from __future__ import annotations

from pathlib import Path


def test_fitness_harness_uses_alignment_owner_modules() -> None:
    source = (Path(__file__).resolve().parents[1] / "bench" / "fitness.py").read_text(
        encoding="utf-8"
    )

    assert "from tomojax.align.api import" in source
    assert "parse_loss_spec" in source
    assert "se3_from_5d" in source
    assert "tomojax.align._objectives.loss_specs" not in source
    assert "tomojax.align._objectives.losses" not in source
    assert "tomojax.align._geometry.parametrizations" not in source
    assert "tomojax.align.losses" not in source
    assert "tomojax.align.parametrizations" not in source
