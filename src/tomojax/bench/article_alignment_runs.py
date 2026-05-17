"""Article alignment scenario and profile contracts.

This module owns the article/demo run catalogue used by the developer script in
``scripts/generate_alignment_before_after_128.py``. The script should orchestrate
files and CLI arguments; scenario translation and profile defaults live here so
they can be tested and reused without importing the script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tomojax.bench.alignment_scenarios import AlignmentScenario, scenario_suite

DEFAULT_LEVELS = (8, 4, 2, 1)


@dataclass(frozen=True)
class ArticleScenario:
    slug: str
    title: str
    description: str
    geometry_type: str
    geometry_dofs: tuple[str, ...]
    active_dofs: tuple[str, ...] = ()
    schedule: str = ""
    scenario_category: str = "capability"
    scenario_family: str = "parallel_ct"
    expectation: str = "success"
    expected_status: tuple[str, ...] = ()
    headline_eligible: bool = True
    phantom_key: str = "phantom94"
    expected_objective: str = "bilevel_cv"
    expected_optimizer: str = "validation_lm"
    expected_loss: str = "l2_otsu"
    hidden_det_u_px: float = 0.0
    hidden_det_v_px: float = 0.0
    hidden_detector_roll_deg: float = 0.0
    hidden_axis_rot_x_deg: float = 0.0
    hidden_axis_rot_y_deg: float = 0.0
    supplied_det_u_px: float | None = None
    supplied_det_v_px: float | None = None
    supplied_detector_roll_deg: float | None = None
    supplied_axis_rot_x_deg: float | None = None
    supplied_axis_rot_y_deg: float | None = None
    nominal_tilt_deg: float = 30.0
    theta_span_deg: float | None = None

    @property
    def true_tilt_deg(self) -> float:
        return float(self.nominal_tilt_deg + self.hidden_axis_rot_x_deg)


@dataclass(frozen=True)
class ArticleRunProfile:
    name: str
    size: int
    views: int
    levels: tuple[int, ...]
    outer_iters: int
    recon_iters: int
    tv_prox_iters: int
    views_per_batch: int
    gather_dtype: str
    early_stop: bool
    early_stop_rel_impr: float
    early_stop_patience: int


def docs_profile() -> ArticleRunProfile:
    return ArticleRunProfile(
        name="docs_128",
        size=128,
        views=128,
        levels=DEFAULT_LEVELS,
        outer_iters=16,
        recon_iters=20,
        tv_prox_iters=12,
        views_per_batch=1,
        gather_dtype="bf16",
        early_stop=True,
        early_stop_rel_impr=1e-3,
        early_stop_patience=2,
    )


def diagnostic_profile() -> ArticleRunProfile:
    return ArticleRunProfile(
        name="diagnostic_32",
        size=32,
        views=32,
        levels=(4, 2, 1),
        outer_iters=2,
        recon_iters=4,
        tv_prox_iters=4,
        views_per_batch=1,
        gather_dtype="fp32",
        early_stop=False,
        early_stop_rel_impr=1e-3,
        early_stop_patience=2,
    )


def _setup_value(source: dict[str, float], name: str, default: float = 0.0) -> float:
    return float(source.get(name, default))


def article_scenario_from_catalog(scenario: AlignmentScenario) -> ArticleScenario:
    hidden = dict(scenario.hidden_setup)
    supplied = dict(scenario.supplied_setup)
    setup_dofs = tuple(
        dof
        for dof in scenario.active_dofs
        if dof
        in {
            "det_u_px",
            "det_v_px",
            "detector_roll_deg",
            "axis_rot_x_deg",
            "axis_rot_y_deg",
            "tilt_deg",
        }
    )
    hidden_axis_rot_x = _setup_value(hidden, "axis_rot_x_deg")
    if "tilt_deg" in hidden and "axis_rot_x_deg" not in hidden:
        hidden_axis_rot_x = _setup_value(hidden, "tilt_deg") - scenario.nominal_tilt_deg
    supplied_axis_rot_x = supplied.get("axis_rot_x_deg")
    if "tilt_deg" in supplied and supplied_axis_rot_x is None:
        supplied_axis_rot_x = float(supplied["tilt_deg"]) - scenario.nominal_tilt_deg
    return ArticleScenario(
        slug=scenario.slug,
        title=scenario.title,
        description=scenario.description,
        geometry_type=scenario.geometry_type,
        geometry_dofs=setup_dofs,
        active_dofs=tuple(scenario.active_dofs),
        schedule=scenario.schedule,
        scenario_category=scenario.category,
        scenario_family=scenario.family,
        expectation=scenario.expectation.kind,
        expected_status=tuple(scenario.expectation.expected_status),
        headline_eligible=scenario.headline_eligible,
        phantom_key=scenario.phantom_key,
        expected_objective=scenario.expected_objective,
        expected_optimizer=scenario.expected_optimizer,
        expected_loss=scenario.expected_loss,
        hidden_det_u_px=_setup_value(hidden, "det_u_px"),
        hidden_det_v_px=_setup_value(hidden, "det_v_px"),
        hidden_detector_roll_deg=_setup_value(hidden, "detector_roll_deg"),
        hidden_axis_rot_x_deg=hidden_axis_rot_x,
        hidden_axis_rot_y_deg=_setup_value(hidden, "axis_rot_y_deg"),
        supplied_det_u_px=supplied.get("det_u_px"),
        supplied_det_v_px=supplied.get("det_v_px"),
        supplied_detector_roll_deg=supplied.get("detector_roll_deg"),
        supplied_axis_rot_x_deg=supplied_axis_rot_x,
        supplied_axis_rot_y_deg=supplied.get("axis_rot_y_deg"),
        nominal_tilt_deg=scenario.nominal_tilt_deg,
        theta_span_deg=scenario.acquisition_span_deg,
    )


def article_scenario_catalog() -> list[ArticleScenario]:
    return article_scenario_catalog_for_kind("default")


def article_visual_stress_scenario_catalog() -> list[ArticleScenario]:
    """Return more aggressive perturbations used for naive-FBP demo selection."""
    return article_scenario_catalog_for_kind("visual_stress")


def article_scenario_catalog_for_kind(kind: str) -> list[ArticleScenario]:
    suite = scenario_suite(kind)
    scenarios = [article_scenario_from_catalog(scenario) for scenario in suite.scenarios()]
    if kind in {"visual_stress", "stress", "stress_128"}:
        validate_article_visual_stress_acquisition(scenarios)
    return scenarios


def profile_from_args(args: Any) -> ArticleRunProfile:
    profile_name = "diagnostic" if args.profile == "smoke" else str(args.profile)
    base = docs_profile() if profile_name == "docs" else diagnostic_profile()
    return ArticleRunProfile(
        name=base.name,
        size=int(args.size or base.size),
        views=int(args.views or base.views),
        levels=tuple(int(v) for v in (args.levels or base.levels)),
        outer_iters=int(args.outer_iters if args.outer_iters is not None else base.outer_iters),
        recon_iters=int(args.recon_iters if args.recon_iters is not None else base.recon_iters),
        tv_prox_iters=int(
            args.tv_prox_iters if args.tv_prox_iters is not None else base.tv_prox_iters
        ),
        views_per_batch=int(
            args.views_per_batch if args.views_per_batch is not None else base.views_per_batch
        ),
        gather_dtype=str(args.gather_dtype or base.gather_dtype),
        early_stop=bool(args.early_stop if args.early_stop is not None else base.early_stop),
        early_stop_rel_impr=float(
            args.early_stop_rel_impr
            if args.early_stop_rel_impr is not None
            else base.early_stop_rel_impr
        ),
        early_stop_patience=int(
            args.early_stop_patience
            if args.early_stop_patience is not None
            else base.early_stop_patience
        ),
    )


def article_theta_span_deg(scenario: ArticleScenario) -> float:
    if scenario.theta_span_deg is not None:
        return float(scenario.theta_span_deg)
    if scenario.geometry_type == "lamino":
        return 360.0
    return 180.0


def has_axis_direction_perturbation(scenario: ArticleScenario) -> bool:
    return (
        abs(float(scenario.hidden_axis_rot_x_deg)) > 1e-7
        or abs(float(scenario.hidden_axis_rot_y_deg)) > 1e-7
    )


def validate_article_visual_stress_acquisition(scenarios: list[ArticleScenario]) -> None:
    for scenario in scenarios:
        if has_axis_direction_perturbation(scenario) and scenario.theta_span_deg is None:
            raise ValueError(
                f"Visual-stress axis scenario {scenario.slug!r} must set theta_span_deg "
                "explicitly so nominal geometry type does not silently choose acquisition span."
            )


__all__ = [
    "DEFAULT_LEVELS",
    "ArticleRunProfile",
    "ArticleScenario",
    "article_scenario_catalog",
    "article_scenario_catalog_for_kind",
    "article_scenario_from_catalog",
    "article_theta_span_deg",
    "article_visual_stress_scenario_catalog",
    "diagnostic_profile",
    "docs_profile",
    "has_axis_direction_perturbation",
    "profile_from_args",
    "validate_article_visual_stress_acquisition",
]
