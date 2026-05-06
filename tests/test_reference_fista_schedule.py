from __future__ import annotations

import pytest

from tomojax.recon import reference_fista_schedule


def test_reference_preview_schedule_matches_phase3_levels_and_filters() -> None:
    schedule = reference_fista_schedule("preview")

    assert schedule.name == "preview"
    assert schedule.level_factors == (4, 2)
    assert [entry.role for entry in schedule.entries] == ["preview", "preview"]
    assert [entry.fista.iterations for entry in schedule.entries] == [4, 6]
    assert [entry.residual_filters[0].kind for entry in schedule.entries] == [
        "lowpass_gaussian",
        "lowpass_gaussian",
    ]
    assert schedule.entries[1].residual_filters[1].kind == "bandpass_difference_of_gaussians"
    assert schedule.entries[1].residual_filters[0].weight == 0.7
    assert schedule.entries[1].residual_filters[1].weight == 0.3


def test_reference_final_schedule_uses_raw_level_one() -> None:
    schedule = reference_fista_schedule("final")

    assert schedule.name == "final"
    assert schedule.level_factors == (1,)
    assert schedule.entries[0].role == "final"
    assert schedule.entries[0].fista.iterations > 0
    assert [config.kind for config in schedule.entries[0].residual_filters] == ["raw"]


def test_reference_fista_schedule_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown reference FISTA schedule"):
        reference_fista_schedule("unknown")  # type: ignore[arg-type]
