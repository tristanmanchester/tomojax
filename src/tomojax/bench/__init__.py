"""Shared benchmark helpers and reusable benchmark-facing utilities."""

from .alignment_scenarios import (
    AcquisitionSpec,
    AlignmentScenario,
    PhantomSpec,
    ScenarioExpectation,
    ScenarioSuite,
    phantom_spec,
    scenario_by_slug,
    scenario_catalog,
    scenario_suite,
    validate_scenario_catalog,
)
from .forward_projector import (
    ForwardProjectorBenchmarkConfig,
    benchmark_backend,
    make_forward_projector_fixture,
    preset_config,
    run_forward_projector_benchmark,
    write_benchmark_json,
)

__all__ = [
    "AcquisitionSpec",
    "AlignmentScenario",
    "ForwardProjectorBenchmarkConfig",
    "PhantomSpec",
    "ScenarioExpectation",
    "ScenarioSuite",
    "benchmark_backend",
    "make_forward_projector_fixture",
    "phantom_spec",
    "preset_config",
    "run_forward_projector_benchmark",
    "scenario_by_slug",
    "scenario_catalog",
    "scenario_suite",
    "validate_scenario_catalog",
    "write_benchmark_json",
]
