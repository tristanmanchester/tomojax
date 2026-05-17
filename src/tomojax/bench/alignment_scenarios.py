from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from tomojax.align.api import dof_spec, schedule_preset

if TYPE_CHECKING:
    from collections.abc import Mapping

GeometryType = Literal["parallel", "lamino"]
ScenarioCategory = Literal["capability", "stress", "pose_parity", "diagnostic"]
ScenarioFamily = Literal["parallel_ct", "full_rotation_axis", "laminography", "pose"]
ExpectationKind = Literal["success", "stress", "diagnostic", "rejected", "weak"]


@dataclass(frozen=True, slots=True)
class PhantomSpec:
    key: str
    kind: str
    source: str
    seed: int
    n_cubes: int
    n_spheres: int
    placement: str
    radial_exponent: float
    selection: str
    shared_across_cases: bool = True

    def to_manifest(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind,
            "seed": int(self.seed),
            "shared_across_cases": bool(self.shared_across_cases),
            "source": self.source,
            "n_cubes": int(self.n_cubes),
            "n_spheres": int(self.n_spheres),
            "placement": self.placement,
            "radial_exponent": float(self.radial_exponent),
            "selection": self.selection,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionSpec:
    geometry_type: GeometryType
    span_deg: float
    n_views_policy: str = "profile"
    nominal_tilt_deg: float = 30.0

    def __post_init__(self) -> None:
        if float(self.span_deg) <= 0.0:
            raise ValueError("AcquisitionSpec.span_deg must be positive")


@dataclass(frozen=True, slots=True)
class ScenarioExpectation:
    kind: ExpectationKind
    headline_eligible: bool
    expected_status: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class AlignmentScenario:
    slug: str
    title: str
    description: str
    category: ScenarioCategory
    family: ScenarioFamily
    acquisition: AcquisitionSpec
    active_dofs: tuple[str, ...]
    schedule: str
    expectation: ScenarioExpectation
    phantom_key: str = "phantom94"
    hidden_setup: Mapping[str, float] = field(default_factory=dict)
    supplied_setup: Mapping[str, float] = field(default_factory=dict)
    expected_objective: str = "bilevel_cv"
    expected_optimizer: str = "validation_lm"
    expected_loss: str = "l2_otsu"

    @property
    def geometry_type(self) -> GeometryType:
        return self.acquisition.geometry_type

    @property
    def acquisition_span_deg(self) -> float:
        return float(self.acquisition.span_deg)

    @property
    def n_views_policy(self) -> str:
        return self.acquisition.n_views_policy

    @property
    def nominal_tilt_deg(self) -> float:
        return float(self.acquisition.nominal_tilt_deg)

    @property
    def headline_eligible(self) -> bool:
        return bool(self.expectation.headline_eligible)

    def to_manifest(
        self, *, suite_name: str | None = None, n_views: int | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "scenario_category": self.category,
            "scenario_family": self.family,
            "category": self.category,
            "family": self.family,
            "geometry_type": self.geometry_type,
            "acquisition_span_deg": self.acquisition_span_deg,
            "theta_span_deg": self.acquisition_span_deg,
            "n_views_policy": self.n_views_policy,
            "nominal_tilt_deg": self.nominal_tilt_deg,
            "phantom_key": self.phantom_key,
            "hidden_setup": dict(self.hidden_setup),
            "supplied_setup": dict(self.supplied_setup),
            "active_dofs": list(self.active_dofs),
            "schedule": self.schedule,
            "expected_objective": self.expected_objective,
            "expected_optimizer": self.expected_optimizer,
            "expected_loss": self.expected_loss,
            "expectation": self.expectation.kind,
            "expected_status": list(self.expectation.expected_status),
            "headline_eligible": self.headline_eligible,
        }
        if suite_name is not None:
            payload["suite_name"] = suite_name
        if n_views is not None:
            payload["n_views"] = int(n_views)
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioSuite:
    name: str
    description: str
    scenario_slugs: tuple[str, ...]
    evidence_size: int = 128

    def scenarios(self) -> tuple[AlignmentScenario, ...]:
        return tuple(scenario_by_slug(slug) for slug in self.scenario_slugs)


PHANTOM94 = PhantomSpec(
    key="phantom94",
    kind="random_shapes/center_biased_sphere_cubes_spheres",
    source="tomojax.data.phantoms.random_cubes_spheres",
    seed=20260893,
    n_cubes=22,
    n_spheres=22,
    placement="center_biased_sphere",
    radial_exponent=0.75,
    selection="phantom_picker_128_10x10_center_biased_sphere_slot_94",
)

_SUCCESS = ScenarioExpectation(
    kind="success",
    headline_eligible=True,
    expected_status=("converged", "underconverged"),
)
_STRESS = ScenarioExpectation(
    kind="stress",
    headline_eligible=True,
    expected_status=("converged", "underconverged"),
)
_WEAK = ScenarioExpectation(
    kind="weak",
    headline_eligible=False,
    expected_status=("ill_conditioned", "underconverged"),
)
_REJECTED = ScenarioExpectation(
    kind="rejected",
    headline_eligible=False,
    expected_status=("rejected",),
)
_DIAGNOSTIC = ScenarioExpectation(
    kind="diagnostic",
    headline_eligible=False,
)


def _acq(
    geometry_type: GeometryType,
    span_deg: float,
    *,
    nominal_tilt_deg: float = 30.0,
) -> AcquisitionSpec:
    return AcquisitionSpec(
        geometry_type=geometry_type,
        span_deg=float(span_deg),
        nominal_tilt_deg=float(nominal_tilt_deg),
    )


def _scenario(
    slug: str,
    title: str,
    description: str,
    *,
    category: ScenarioCategory,
    family: ScenarioFamily,
    acquisition: AcquisitionSpec,
    active_dofs: tuple[str, ...],
    schedule: str,
    expectation: ScenarioExpectation,
    hidden_setup: Mapping[str, float] | None = None,
    supplied_setup: Mapping[str, float] | None = None,
    expected_objective: str = "bilevel_cv",
    expected_optimizer: str = "validation_lm",
    expected_loss: str = "l2_otsu",
) -> AlignmentScenario:
    return AlignmentScenario(
        slug=slug,
        title=title,
        description=description,
        category=category,
        family=family,
        acquisition=acquisition,
        active_dofs=active_dofs,
        schedule=schedule,
        expectation=expectation,
        hidden_setup=dict(hidden_setup or {}),
        supplied_setup=dict(supplied_setup or {}),
        expected_objective=expected_objective,
        expected_optimizer=expected_optimizer,
        expected_loss=expected_loss,
    )


_CAPABILITY_SCENARIOS: tuple[AlignmentScenario, ...] = (
    _scenario(
        "parallel_cor_u_m004",
        "Parallel CT: detector/ray-grid centre -4 px",
        "Canonical detector-u centre offset estimated with the COR preset.",
        category="capability",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("det_u_px",),
        schedule="cor",
        expectation=_SUCCESS,
        hidden_setup={"det_u_px": -4.0},
    ),
    _scenario(
        "parallel_detector_roll_p2p5",
        "Parallel CT: detector roll +2.5 deg",
        "Canonical detector-plane roll correction.",
        category="capability",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("detector_roll_deg",),
        schedule="detector_roll",
        expectation=_SUCCESS,
        hidden_setup={"detector_roll_deg": 2.5},
    ),
    _scenario(
        "parallel_axis_pitch_full360_p2p0",
        "Full-rotation arbitrary axis: pitch +2 deg",
        "Mild full-rotation axis pitch correction.",
        category="capability",
        family="full_rotation_axis",
        acquisition=_acq("parallel", 360.0),
        active_dofs=("axis_rot_x_deg",),
        schedule="axis_direction",
        expectation=_SUCCESS,
        hidden_setup={"axis_rot_x_deg": 2.0},
    ),
    _scenario(
        "parallel_axis_yaw_full360_m2p0",
        "Full-rotation arbitrary axis: yaw -2 deg",
        "Mild full-rotation axis yaw correction.",
        category="capability",
        family="full_rotation_axis",
        acquisition=_acq("parallel", 360.0),
        active_dofs=("axis_rot_y_deg",),
        schedule="axis_direction",
        expectation=_SUCCESS,
        hidden_setup={"axis_rot_y_deg": -2.0},
    ),
    _scenario(
        "parallel_axis_pitch_yaw_full360",
        "Full-rotation arbitrary axis: pitch and yaw",
        "Coupled full-rotation axis-direction correction.",
        category="capability",
        family="full_rotation_axis",
        acquisition=_acq("parallel", 360.0),
        active_dofs=("axis_rot_x_deg", "axis_rot_y_deg"),
        schedule="axis_direction",
        expectation=_SUCCESS,
        hidden_setup={"axis_rot_x_deg": 1.5, "axis_rot_y_deg": -1.5},
    ),
    _scenario(
        "lamino_tilt_34p4",
        "Laminography: true tilt 34.4 deg",
        "Nominal tilt is 30 deg; the hidden instrument tilt delta is +4.4 deg.",
        category="capability",
        family="laminography",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("tilt_deg",),
        schedule="lamino_tilt",
        expectation=_SUCCESS,
        hidden_setup={"axis_rot_x_deg": 4.4, "tilt_deg": 34.4},
    ),
    _scenario(
        "parallel_cor_roll_combo",
        "Parallel CT: detector centre plus roll",
        "Coupled detector-u centre offset and detector-plane roll.",
        category="capability",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("det_u_px", "detector_roll_deg"),
        schedule="expert_coupled",
        expectation=_SUCCESS,
        hidden_setup={"det_u_px": -3.0, "detector_roll_deg": 2.0},
    ),
    _scenario(
        "lamino_cor_tilt_combo",
        "Laminography: detector centre plus tilt",
        "Combined detector-u offset and laminography tilt correction.",
        category="capability",
        family="laminography",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("det_u_px", "tilt_deg"),
        schedule="expert_coupled",
        expectation=_SUCCESS,
        hidden_setup={"det_u_px": -3.0, "axis_rot_x_deg": 4.4, "tilt_deg": 34.4},
    ),
    _scenario(
        "parallel_setup_safe",
        "Parallel CT: staged setup-safe alignment",
        "Staged COR, detector roll, axis direction, and pose polish.",
        category="capability",
        family="parallel_ct",
        acquisition=_acq("parallel", 360.0),
        active_dofs=(
            "det_u_px",
            "detector_roll_deg",
            "axis_rot_x_deg",
            "axis_rot_y_deg",
            "alpha",
            "beta",
            "phi",
            "dx",
            "dz",
        ),
        schedule="setup_safe",
        expectation=_SUCCESS,
        hidden_setup={"det_u_px": -3.0, "detector_roll_deg": 2.0, "axis_rot_x_deg": 1.0},
        expected_objective="staged",
        expected_optimizer="staged",
    ),
    _scenario(
        "lamino_setup_safe",
        "Laminography: staged setup-safe alignment",
        "Staged detector centre, tilt/axis direction, and pose polish.",
        category="capability",
        family="laminography",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("det_u_px", "axis_rot_x_deg", "alpha", "beta", "phi", "dx", "dz"),
        schedule="setup_safe",
        expectation=_SUCCESS,
        hidden_setup={"det_u_px": -3.0, "axis_rot_x_deg": 4.4, "tilt_deg": 34.4},
        expected_objective="staged",
        expected_optimizer="staged",
    ),
)

_STRESS_SCENARIOS: tuple[AlignmentScenario, ...] = (
    _scenario(
        "stress_parallel_cor_u_m008",
        "Parallel CT: detector/ray-grid centre -8 px",
        "Large detector-u centre offset for visual artifact screening.",
        category="stress",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("det_u_px",),
        schedule="cor",
        expectation=_STRESS,
        hidden_setup={"det_u_px": -8.0},
    ),
    _scenario(
        "stress_parallel_detector_roll_p10",
        "Parallel CT: detector roll +10 deg",
        "Large detector-plane roll for visual artifact screening.",
        category="stress",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("detector_roll_deg",),
        schedule="detector_roll",
        expectation=_STRESS,
        hidden_setup={"detector_roll_deg": 10.0},
    ),
    _scenario(
        "stress_parallel_axis_pitch_full360_p18",
        "Full-rotation arbitrary axis: pitch +18 deg",
        "Large full-rotation axis pitch for visual artifact screening.",
        category="stress",
        family="full_rotation_axis",
        acquisition=_acq("parallel", 360.0),
        active_dofs=("axis_rot_x_deg",),
        schedule="axis_direction",
        expectation=_STRESS,
        hidden_setup={"axis_rot_x_deg": 18.0},
    ),
    _scenario(
        "stress_parallel_axis_yaw_full360_m18",
        "Full-rotation arbitrary axis: yaw -18 deg",
        "Large full-rotation axis yaw for visual artifact screening.",
        category="stress",
        family="full_rotation_axis",
        acquisition=_acq("parallel", 360.0),
        active_dofs=("axis_rot_y_deg",),
        schedule="axis_direction",
        expectation=_STRESS,
        hidden_setup={"axis_rot_y_deg": -18.0},
    ),
    _scenario(
        "stress_lamino_tilt_50",
        "Laminography: true tilt 50 deg",
        "Large hidden tilt delta from nominal 30 deg for visual artifact screening.",
        category="stress",
        family="laminography",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("tilt_deg",),
        schedule="lamino_tilt",
        expectation=_STRESS,
        hidden_setup={"axis_rot_x_deg": 20.0, "tilt_deg": 50.0},
    ),
    _scenario(
        "stress_lamino_cor_tilt_combo",
        "Laminography: detector centre plus strong tilt",
        "Strong coupled lamino setup error for visual artifact screening.",
        category="stress",
        family="laminography",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("det_u_px", "tilt_deg"),
        schedule="expert_coupled",
        expectation=_STRESS,
        hidden_setup={"det_u_px": -6.0, "axis_rot_x_deg": 16.0, "tilt_deg": 46.0},
    ),
)

_POSE_PARITY_SCENARIOS: tuple[AlignmentScenario, ...] = (
    _scenario(
        "pose_only_parallel_5dof",
        "Parallel CT: pose-only 5-DOF parity",
        "Pose-only GN parity case for alpha, beta, phi, dx, and dz.",
        category="pose_parity",
        family="pose",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("alpha", "beta", "phi", "dx", "dz"),
        schedule="pose_only",
        expectation=_SUCCESS,
        expected_objective="fixed_volume",
        expected_optimizer="gn",
    ),
    _scenario(
        "pose_only_lamino_5dof",
        "Laminography: pose-only 5-DOF parity",
        "Laminography pose-only GN parity case.",
        category="pose_parity",
        family="pose",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("alpha", "beta", "phi", "dx", "dz"),
        schedule="pose_only",
        expectation=_SUCCESS,
        expected_objective="fixed_volume",
        expected_optimizer="gn",
    ),
    _scenario(
        "pose_phi_only_lamino",
        "Laminography: staged phi-only pose alignment",
        "Real-lamo-inspired phi-only fixed-volume pose stage.",
        category="pose_parity",
        family="pose",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=34.4),
        active_dofs=("phi",),
        schedule="pose_phi_only",
        expectation=_SUCCESS,
        expected_objective="fixed_volume",
        expected_optimizer="gn",
    ),
    _scenario(
        "pose_dx_dz_after_phi_lamino",
        "Laminography: dx/dz pose polish after phi",
        "Real-lamo-inspired translation polish stage after phi alignment.",
        category="pose_parity",
        family="pose",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=34.4),
        active_dofs=("dx", "dz"),
        schedule="pose_dx_dz_after_phi",
        expectation=_SUCCESS,
        expected_objective="fixed_volume",
        expected_optimizer="gn",
    ),
    _scenario(
        "setup_then_pose_polish_parallel",
        "Parallel CT: setup then pose polish",
        "Setup-safe staged alignment followed by fixed-volume pose polish.",
        category="pose_parity",
        family="parallel_ct",
        acquisition=_acq("parallel", 360.0),
        active_dofs=(
            "det_u_px",
            "detector_roll_deg",
            "axis_rot_x_deg",
            "axis_rot_y_deg",
            "alpha",
            "beta",
            "phi",
            "dx",
            "dz",
        ),
        schedule="setup_safe",
        expectation=_SUCCESS,
        expected_objective="staged",
        expected_optimizer="staged",
    ),
    _scenario(
        "setup_then_pose_polish_lamino",
        "Laminography: setup then pose polish",
        "Lamino setup-safe staged alignment followed by fixed-volume pose polish.",
        category="pose_parity",
        family="laminography",
        acquisition=_acq("lamino", 360.0, nominal_tilt_deg=30.0),
        active_dofs=("det_u_px", "axis_rot_x_deg", "alpha", "beta", "phi", "dx", "dz"),
        schedule="setup_safe",
        expectation=_SUCCESS,
        expected_objective="staged",
        expected_optimizer="staged",
    ),
)

_DIAGNOSTIC_SCENARIOS: tuple[AlignmentScenario, ...] = (
    _scenario(
        "diagnostic_parallel_axis_pitch_180",
        "Diagnostic: 180-degree axis pitch",
        "Known weak arbitrary-axis setup from 180-degree parallel data.",
        category="diagnostic",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("axis_rot_x_deg",),
        schedule="axis_direction",
        expectation=_WEAK,
        hidden_setup={"axis_rot_x_deg": 2.0},
    ),
    _scenario(
        "diagnostic_parallel_axis_yaw_180",
        "Diagnostic: 180-degree axis yaw",
        "Known weak arbitrary-axis setup from 180-degree parallel data.",
        category="diagnostic",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("axis_rot_y_deg",),
        schedule="axis_direction",
        expectation=_WEAK,
        hidden_setup={"axis_rot_y_deg": -2.0},
    ),
    _scenario(
        "diagnostic_cor_plus_pose_dx_gauge",
        "Diagnostic: detector centre plus pose dx gauge",
        "Gauge-coupled detector centre and pose translation should require an anchor.",
        category="diagnostic",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("det_u_px", "dx"),
        schedule="expert_coupled",
        expectation=_REJECTED,
        hidden_setup={"det_u_px": -4.0},
    ),
    _scenario(
        "diagnostic_roll_plus_mean_phi_gauge",
        "Diagnostic: detector roll plus mean phi gauge",
        "Gauge-coupled detector roll and global object orientation should be diagnosed.",
        category="diagnostic",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("detector_roll_deg", "phi"),
        schedule="expert_coupled",
        expectation=_REJECTED,
        hidden_setup={"detector_roll_deg": 2.5},
    ),
    _scenario(
        "diagnostic_all_data_bilevel_setup_forbidden",
        "Diagnostic: all-data bilevel setup is forbidden",
        "Default setup discovery must not use all-data bilevel scoring.",
        category="diagnostic",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("det_u_px",),
        schedule="cor",
        expectation=_REJECTED,
        hidden_setup={"det_u_px": -4.0},
        expected_objective="all_data_bilevel",
    ),
    _scenario(
        "diagnostic_fixed_volume_setup_discovery_forbidden",
        "Diagnostic: fixed-volume setup discovery is forbidden",
        "Default setup discovery must not use fixed-volume same-data scoring.",
        category="diagnostic",
        family="parallel_ct",
        acquisition=_acq("parallel", 180.0),
        active_dofs=("det_u_px",),
        schedule="cor",
        expectation=_REJECTED,
        hidden_setup={"det_u_px": -4.0},
        expected_objective="fixed_volume",
        expected_optimizer="gn",
    ),
)

_SCENARIOS: tuple[AlignmentScenario, ...] = (
    *_CAPABILITY_SCENARIOS,
    *_STRESS_SCENARIOS,
    *_POSE_PARITY_SCENARIOS,
    *_DIAGNOSTIC_SCENARIOS,
)

_SCENARIO_BY_SLUG = {scenario.slug: scenario for scenario in _SCENARIOS}

_SUITES: dict[str, ScenarioSuite] = {
    "capability_128": ScenarioSuite(
        name="capability_128",
        description="Normal correctable 128^3 setup-geometry evidence cases.",
        scenario_slugs=tuple(s.slug for s in _CAPABILITY_SCENARIOS),
    ),
    "stress_128": ScenarioSuite(
        name="stress_128",
        description="Visually obvious 128^3 docs and demo stress cases.",
        scenario_slugs=tuple(s.slug for s in _STRESS_SCENARIOS),
    ),
    "pose_parity_128": ScenarioSuite(
        name="pose_parity_128",
        description="Pose-only GN and staged setup-plus-pose parity cases.",
        scenario_slugs=tuple(s.slug for s in _POSE_PARITY_SCENARIOS),
    ),
    "diagnostic_128": ScenarioSuite(
        name="diagnostic_128",
        description="Weak, gauge-coupled, or intentionally rejected cases.",
        scenario_slugs=tuple(s.slug for s in _DIAGNOSTIC_SCENARIOS),
    ),
    "smoke_64": ScenarioSuite(
        name="smoke_64",
        description="Representative 64^3 smoke subset, not the evidence suite.",
        scenario_slugs=(
            "parallel_cor_u_m004",
            "parallel_detector_roll_p2p5",
            "lamino_tilt_34p4",
        ),
        evidence_size=64,
    ),
}

_SUITES["comprehensive_128"] = ScenarioSuite(
    name="comprehensive_128",
    description="Capability, stress, pose parity, and diagnostics in one catalog suite.",
    scenario_slugs=(
        *_SUITES["capability_128"].scenario_slugs,
        *_SUITES["stress_128"].scenario_slugs,
        *_SUITES["pose_parity_128"].scenario_slugs,
        *_SUITES["diagnostic_128"].scenario_slugs,
    ),
)

_SUITE_ALIASES = {
    "default": "capability_128",
    "capability": "capability_128",
    "visual_stress": "stress_128",
    "stress": "stress_128",
    "diagnostic": "diagnostic_128",
    "pose_parity": "pose_parity_128",
}


def phantom_spec(key: str) -> PhantomSpec:
    normalized = str(key).strip().lower()
    if normalized == "phantom94":
        return PHANTOM94
    raise ValueError(f"Unknown phantom spec {key!r}; valid phantoms: phantom94")


def scenario_catalog() -> tuple[AlignmentScenario, ...]:
    return _SCENARIOS


def scenario_by_slug(slug: str) -> AlignmentScenario:
    try:
        return _SCENARIO_BY_SLUG[str(slug)]
    except KeyError as exc:
        raise ValueError(f"Unknown alignment scenario {slug!r}") from exc


def scenario_suite(name: str) -> ScenarioSuite:
    key = str(name).strip().lower()
    key = _SUITE_ALIASES.get(key, key)
    try:
        return _SUITES[key]
    except KeyError as exc:
        valid = ", ".join(sorted((*_SUITES.keys(), *_SUITE_ALIASES.keys())))
        raise ValueError(
            f"Unknown alignment scenario suite {name!r}; valid suites: {valid}"
        ) from exc


def validate_scenario_catalog() -> None:
    slugs = [scenario.slug for scenario in _SCENARIOS]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Alignment scenario slugs must be unique")
    if slugs != sorted(slugs, key=slugs.index):
        raise ValueError("Alignment scenario order must be deterministic")
    for scenario in _SCENARIOS:
        phantom_spec(scenario.phantom_key)
        for dof in scenario.active_dofs:
            dof_spec(dof)
        if scenario.schedule == "expert_coupled":
            schedule_preset(
                scenario.schedule,
                active_dofs=scenario.active_dofs,
                gauge_policy="prior_required",
            )
        else:
            schedule_preset(scenario.schedule)
        if (
            scenario.category in {"capability", "stress"}
            and scenario.family == "full_rotation_axis"
            and float(scenario.acquisition_span_deg) != 360.0
        ):
            raise ValueError(
                f"Headline arbitrary-axis scenario {scenario.slug!r} must use 360 degrees"
            )
        if (
            scenario.expectation.kind in {"diagnostic", "rejected", "weak"}
            and scenario.headline_eligible
        ):
            raise ValueError(f"Diagnostic scenario {scenario.slug!r} cannot be headline eligible")
    for suite in _SUITES.values():
        for slug in suite.scenario_slugs:
            scenario_by_slug(slug)
