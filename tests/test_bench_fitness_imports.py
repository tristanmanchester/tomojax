from __future__ import annotations

from pathlib import Path


def test_fitness_harness_uses_alignment_owner_modules() -> None:
    source = (Path(__file__).resolve().parents[1] / "bench" / "fitness.py").read_text(
        encoding="utf-8"
    )

    assert "tomojax.align.objectives.losses" in source
    assert "tomojax.align.geometry.parametrizations" in source
    assert "tomojax.align.losses" not in source
    assert "tomojax.align.parametrizations" not in source
