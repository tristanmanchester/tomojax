from __future__ import annotations

from typing import cast

import pytest

from tomojax.align.api import ContinuationScheduleName, reference_continuation_schedule


@pytest.mark.parametrize("name", ["smoke32", "lightning", "balanced", "reference"])
def test_continuation_profiles_share_phase7_level_order(name: str) -> None:
    schedule = reference_continuation_schedule(cast("ContinuationScheduleName", name))

    assert schedule.level_factors == (4, 2, 1)
    assert schedule.levels[0].role == "preview"
    assert schedule.levels[1].run_if_coarse_unverified
    assert schedule.levels[2].role == "final"
    assert not schedule.levels[2].skip_finer_if_verified


def test_continuation_profiles_increase_work_monotonically() -> None:
    lightning = reference_continuation_schedule("lightning")
    balanced = reference_continuation_schedule("balanced")
    reference = reference_continuation_schedule("reference")

    assert (
        lightning.levels[0].reconstruction_iterations < balanced.levels[0].reconstruction_iterations
    )
    assert (
        balanced.levels[0].reconstruction_iterations < reference.levels[0].reconstruction_iterations
    )
    assert lightning.levels[0].geometry_updates < balanced.levels[0].geometry_updates
    assert balanced.levels[0].geometry_updates < reference.levels[0].geometry_updates
    assert lightning.levels[2].geometry_updates <= balanced.levels[2].geometry_updates
    assert balanced.levels[2].geometry_updates < reference.levels[2].geometry_updates


def test_unknown_continuation_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown continuation schedule"):
        _ = reference_continuation_schedule(cast("ContinuationScheduleName", "slow"))
