from __future__ import annotations

import pytest

from tomojax.align.dof_specs import dof_spec
from tomojax.align.schedules import schedule_preset
from tomojax.bench.alignment_scenarios import (
    scenario_catalog,
    scenario_suite,
    validate_scenario_catalog,
)


def test_alignment_scenario_catalog_validates():
    validate_scenario_catalog()


def test_alignment_scenario_slugs_are_unique_and_deterministic():
    scenarios = scenario_catalog()
    slugs = [scenario.slug for scenario in scenarios]

    assert len(slugs) == len(set(slugs))
    assert slugs == [scenario.slug for scenario in scenario_catalog()]


def test_alignment_scenario_dofs_and_schedules_resolve():
    for scenario in scenario_catalog():
        for dof in scenario.active_dofs:
            dof_spec(dof)
        if scenario.schedule == "expert_coupled":
            schedule = schedule_preset(
                scenario.schedule,
                active_dofs=scenario.active_dofs,
                gauge_policy="prior_required",
            )
        else:
            schedule = schedule_preset(scenario.schedule)
        assert schedule.name == scenario.schedule


def test_headline_arbitrary_axis_scenarios_use_full_rotation():
    for scenario in scenario_catalog():
        if scenario.category not in {"capability", "stress"}:
            continue
        if scenario.family != "full_rotation_axis":
            continue
        assert scenario.acquisition_span_deg == 360.0
        assert scenario.headline_eligible is True


@pytest.mark.parametrize(
    "slug",
    [
        "diagnostic_parallel_axis_pitch_180",
        "diagnostic_parallel_axis_yaw_180",
    ],
)
def test_weak_180_axis_cases_are_diagnostic_not_headline(slug: str):
    diagnostic = {scenario.slug: scenario for scenario in scenario_suite("diagnostic_128").scenarios()}

    scenario = diagnostic[slug]
    assert scenario.acquisition_span_deg == 180.0
    assert scenario.expectation.kind == "weak"
    assert scenario.headline_eligible is False


def test_default_and_headline_suites_exclude_rejected_controls_and_duplicate_cor():
    headline_suites = ("default", "capability_128", "stress_128", "comprehensive_128")
    forbidden = {"known_det_u_control", "parallel_det_u_p004"}

    for suite_name in headline_suites:
        slugs = {scenario.slug for scenario in scenario_suite(suite_name).scenarios()}
        assert forbidden.isdisjoint(slugs)


def test_headline_suites_exclude_static_detector_v_shift_cases():
    headline_suites = ("default", "capability_128", "stress_128", "comprehensive_128")

    for suite_name in headline_suites:
        for scenario in scenario_suite(suite_name).scenarios():
            assert "det_v_px" not in scenario.active_dofs
            assert "det_v_px" not in scenario.hidden_setup


def test_comprehensive_suite_contains_all_canonical_categories():
    scenarios = scenario_suite("comprehensive_128").scenarios()
    categories = {scenario.category for scenario in scenarios}

    assert {"capability", "stress", "pose_parity", "diagnostic"}.issubset(categories)
    assert all(
        scenario.headline_eligible is False
        for scenario in scenarios
        if scenario.category == "diagnostic"
    )


def test_pose_parity_scenarios_expect_fixed_volume_gn():
    scenarios = scenario_suite("pose_parity_128").scenarios()
    pose_only = [scenario for scenario in scenarios if scenario.slug.startswith("pose_")]

    assert pose_only
    for scenario in pose_only:
        assert scenario.expected_objective == "fixed_volume"
        assert scenario.expected_optimizer == "gn"
