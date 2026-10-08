"""Product alignment modes: the configuration each one runs, and cone-beam setup.

A mode names a goal; :func:`alignment_plan` turns it into the schedule, solver
settings and resolution levels that reach it, the same for ``tomojax.alignment``
in Python and ``tomojax align`` on the command line.

``pose``
    Per-view motion (rotations and translations), with the coupled solver.
``cor``
    The setup geometry only: detector centre (parallel beams), or the axis
    offset and detector roll (cone beams).
``cor-then-pose``
    Per-view motion whose constant part becomes the detector centre
    (parallel), or the cone axis calibration followed by per-view motion.
``full``
    The setup stages (centre, roll, axis direction) and then per-view motion,
    coarse to fine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING, Literal

from tomojax.alignment._config import AlignConfig, resolved_schedule_for_config
from tomojax.alignment._model.dofs import DOF_NAMES
from tomojax.alignment._model.schedules import AlignmentSchedule, schedule_preset
from tomojax.alignment._objectives.loss_specs import L2LossSpec
from tomojax.alignment._profiles import normalize_quality, resolve_profiled_cli_defaults
from tomojax.core.validation import option_name

if TYPE_CHECKING:
    from collections.abc import Iterable

    import jax

    from tomojax.core.geometry.base import Detector, Geometry, Grid

LOG = logging.getLogger(__name__)

type AlignmentMode = Literal["pose", "cor", "cor-then-pose", "full"]
MODES: tuple[AlignmentMode, ...] = ("pose", "cor", "cor-then-pose", "full")

_SCHEDULES = {"cor": "cor", "cor-then-pose": "cor_then_pose", "full": "setup_safe"}
_PROFILE_KEYS = (
    "projector_backend",
    "gather_dtype",
    "regulariser",
    "reconstruction",
    "views_per_batch",
    "checkpoint_projector",
    "pose_model",
)
# Setup DOFs a cone beam calibrates as its axis offset and detector roll.
_CONE_AXIS_DOFS = frozenset({"det_u_px", "detector_roll_deg"})


def normalize_mode(mode: str) -> AlignmentMode:
    """Return the canonical mode name; case and ``_`` versus ``-`` do not matter."""
    name = option_name(mode, separator="-")
    for candidate in MODES:
        if name == candidate:
            return candidate
    raise ValueError(f"alignment mode must be one of {', '.join(MODES)}; got {mode!r}")


@dataclass(frozen=True)
class AlignmentPlan:
    """What an alignment run executes: its configuration and resolution levels."""

    mode: AlignmentMode
    config: AlignConfig
    levels: tuple[int, ...]
    pose_solver: Literal["coupled", "alternating"]


def coupled_levels(grid: Grid) -> tuple[int, ...]:
    """Coarse-to-fine factors keeping at least 32 voxels on the shortest axis.

    Coarse levels cost little and halved a 256-cubed laminography alignment's
    time with unchanged pose accuracy and a better volume.
    """
    shortest = min(grid.nx, grid.ny, grid.nz)
    return (*(factor for factor in (4, 2) if shortest // factor >= 32), 1)


def coupled_overrides(config: AlignConfig, *, explicit: Iterable[str] = ()) -> dict[str, object]:
    """Settings the coupled pose solver needs, keeping the ``explicit`` fields' values."""
    keep = set(explicit)
    options: dict[str, object] = {
        "gn_coupling": "joint",
        "gn_joint_solver": "pose_eliminated",
        "ray_integrator": config.ray_integrator if "ray_integrator" in keep else "joseph",
        "gather_dtype": "fp32",
        "opt_method": "gn",
        "pose_model": "per_view",
        # A global shift search extends the local solver's capture range.
        "seed_translations": config.seed_translations if "seed_translations" in keep else True,
    }
    if "loss" not in keep:
        options["loss"] = L2LossSpec()
    if "tv_weight" not in keep:
        # Default-weight TV biased coupled pose recovery in the free-voxel pilot.
        options["tv_weight"] = 0.0
    return options


def alignment_plan(
    mode: str,
    grid: Grid,
    *,
    quality: str = "fast",
    levels: Iterable[int] | None = None,
    freeze: Iterable[str] = (),
    config: AlignConfig | None = None,
) -> AlignmentPlan:
    """Return the configuration and levels that run ``mode`` at ``quality``.

    ``config`` supplies expert settings and is used as given; without it the
    quality profile's defaults apply. The mode sets the schedule unless
    ``config`` names one (or ``optimise_dofs``). ``freeze`` keeps the named
    parameters (for example ``"dy"``) at their initial values.
    """
    name = normalize_mode(mode)
    tier = normalize_quality(quality)
    expert = config is not None
    cfg = config or AlignConfig(quality=tier, pose_translation_frame="detector")
    if not expert:
        current = {key: getattr(cfg, key) for key in _PROFILE_KEYS}
        resolved = resolve_profiled_cli_defaults(
            quality=tier, current=current, configured_keys=set()
        )
        cfg = replace(
            cfg,
            **{key: resolved[key] for key in _PROFILE_KEYS},  # type: ignore[arg-type]
        )
    if cfg.schedule is None and cfg.optimise_dofs is None:
        if name == "pose":
            schedule = "lightning_pose" if tier == "fast" else "tortoise_pose"
        else:
            schedule = _SCHEDULES[name]
        cfg = replace(cfg, schedule=schedule)
    frozen = tuple(dict.fromkeys((*(cfg.freeze or ()), *freeze)))
    if frozen:
        cfg = replace(cfg, freeze=frozen)
    pose_solver: Literal["coupled", "alternating"] = (
        "coupled" if name in {"pose", "cor-then-pose"} else "alternating"
    )
    if pose_solver == "coupled" and not expert:
        cfg = replace(cfg, outer_iterations=30, **coupled_overrides(cfg))  # type: ignore[arg-type]
    elif not expert:
        # Joseph plane sampling is the fastest forward model for every mode.
        cfg = replace(cfg, ray_integrator="joseph")
    if levels is not None:
        run_levels = tuple(int(f) for f in levels)
    elif name == "full":
        run_levels = (4, 2, 1)
    elif pose_solver == "coupled":
        run_levels = coupled_levels(grid)
    else:
        run_levels = (1,)
    return AlignmentPlan(mode=name, config=cfg, levels=run_levels, pose_solver=pose_solver)


def pose_stages_only(config: AlignConfig) -> AlignConfig | None:
    """``config`` without its setup stages, or None when only setup stages remain."""
    if config.optimise_dofs is not None:
        dofs = tuple(name for name in config.optimise_dofs if name in DOF_NAMES)
        return replace(config, optimise_dofs=dofs) if dofs else None
    if config.schedule is None:
        return config
    base = (
        config.schedule
        if isinstance(config.schedule, AlignmentSchedule)
        else schedule_preset(config.schedule)
    )
    stages = tuple(
        stage for stage in base.stages if all(name in DOF_NAMES for name in stage.active_dofs)
    )
    if not stages:
        return None
    return replace(config, schedule=AlignmentSchedule(name=f"{base.name}_poses", stages=stages))


@dataclass(frozen=True)
class ConeSetup:
    """A cone scan's setup stages, run as :func:`tomojax.recon.calibrate_cone_axis`.

    ``geometry`` carries the calibrated axis offset and detector roll;
    ``config`` keeps only the pose stages, or is None when none remain.
    """

    geometry: Geometry
    config: AlignConfig | None
    axis_offset: float
    detector_roll_deg: float
    heights: tuple[float, ...]
    slab_offsets: tuple[float, ...]

    def to_dict(self) -> dict[str, float | list[float]]:
        """The calibration, as alignment records keep it."""
        return {
            "axis_offset": self.axis_offset,
            "detector_roll_deg": self.detector_roll_deg,
            "heights": list(self.heights),
            "slab_offsets": list(self.slab_offsets),
        }


def cone_setup(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array,
    config: AlignConfig,
) -> ConeSetup | None:
    """Calibrate a cone scan's axis when ``config`` asks for setup stages.

    Returns None for parallel beams and for schedules without setup stages
    (other than ``cor_then_pose``, which calibrates the axis first). Setup
    DOFs a cone beam cannot estimate (the axis direction) raise when named in
    ``optimise_dofs`` and are skipped with a warning in preset schedules.
    """
    from tomojax.core.geometry.cone import is_cone_beam
    from tomojax.geometry import ConeGeometry
    from tomojax.recon import calibrate_cone_axis

    resolved = resolved_schedule_for_config(config)
    setup = set(resolved.active_geometry_dofs)
    if not is_cone_beam(geometry) or (not setup and resolved.name != "cor_then_pose"):
        return None
    if not isinstance(geometry, ConeGeometry):
        raise ValueError(
            "cone-beam axis calibration takes one source-detector arrangement without pose "
            "corrections; align segmented or posed scans with mode='pose'"
        )
    if setup - _CONE_AXIS_DOFS:
        unsupported = ", ".join(sorted(setup - _CONE_AXIS_DOFS))
        message = (
            "cone-beam alignment calibrates the axis offset and detector roll; it cannot "
            f"estimate {unsupported}"
        )
        if config.optimise_dofs is not None:
            raise ValueError(message)
        LOG.warning("%s, so those setup stages are skipped", message)
    calibration = calibrate_cone_axis(geometry, grid, detector, projections)
    calibrated = calibration.apply(geometry)
    LOG.info(
        "Cone-beam axis offset %.4f (%.3f detector px at the axis), detector roll %.4f deg",
        calibration.axis_offset,
        calibration.axis_offset * calibrated.beam.magnification / float(detector.du),
        calibration.detector_roll_deg,
    )
    return ConeSetup(
        geometry=calibrated,
        config=pose_stages_only(config),
        axis_offset=calibration.axis_offset,
        detector_roll_deg=calibration.detector_roll_deg,
        heights=calibration.heights,
        slab_offsets=calibration.slab_offsets,
    )


__all__ = [
    "MODES",
    "AlignmentMode",
    "AlignmentPlan",
    "ConeSetup",
    "alignment_plan",
    "cone_setup",
    "coupled_levels",
    "coupled_overrides",
    "normalize_mode",
    "pose_stages_only",
]
