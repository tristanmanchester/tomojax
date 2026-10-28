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

from tomojax.alignment._config import (
    AlignConfig,
    coupled_pose_defaults,
    resolved_schedule_for_config,
)
from tomojax.alignment._model.dofs import DOF_NAMES
from tomojax.alignment._model.schedules import AlignmentSchedule, schedule_preset
from tomojax.alignment._profiles import alignment_profile_policy, normalize_quality

if TYPE_CHECKING:
    from collections.abc import Iterable

    import jax

    from tomojax.core.geometry.base import Detector, Geometry, Grid

LOG = logging.getLogger(__name__)

type AlignmentMode = Literal["pose", "cor", "cor-then-pose", "full"]
MODES: tuple[AlignmentMode, ...] = ("pose", "cor", "cor-then-pose", "full")

_SCHEDULES = {"cor": "cor", "cor-then-pose": "cor_then_pose", "full": "setup_safe"}
# Setup DOFs a cone beam calibrates as its axis offset and detector roll.
_CONE_AXIS_DOFS = frozenset({"det_u_px", "detector_roll_deg"})


def normalize_mode(mode: str) -> AlignmentMode:
    """Return ``mode``, checked: one of :data:`MODES`, spelled as there (the CLI's names)."""
    for candidate in MODES:
        if mode == candidate:
            return candidate
    raise ValueError(f"alignment mode must be one of {', '.join(MODES)}; got {mode!r}")


@dataclass(frozen=True)
class AlignmentPlan:
    """What an alignment run executes: its configuration and resolution levels.

    Parameters
    ----------
    mode : str
        Alignment mode: "pose", "cor", "cor-then-pose" or "full".
    config : AlignConfig
        Resolved solver configuration.
    levels : tuple of int
        Coarse-to-fine resolution factors.
    pose_solver : {"coupled", "alternating"}
        Joint volume/pose updates or alternating fixed-volume pose updates.
    """

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


def alignment_plan(
    mode: str,
    grid: Grid,
    *,
    quality: str = "fast",
    geometry: Geometry | None = None,
    levels: Iterable[int] | None = None,
    freeze: Iterable[str] = (),
    config: AlignConfig | None = None,
) -> AlignmentPlan:
    """Return the configuration and levels that run ``mode`` at ``quality``.

    ``config`` supplies expert settings and is used as given; without it the
    quality profile's defaults apply. The mode sets the schedule unless
    ``config`` names one (or ``optimise_dofs``). ``freeze`` keeps the named
    parameters (for example ``"dy"``) at their initial values.
    When ``geometry`` is a cone beam, fast coupled plans limit each inexact
    linear update to ten CG steps; nonlinear acceptance and outer convergence
    still decide progress. Without geometry context, budgets are conservative.

    Parameters
    ----------
    mode : str
        Alignment mode: "pose", "cor", "cor-then-pose" or "full".
    grid : Grid
        Full-resolution volume grid used to choose automatic levels.
    quality : {"fast", "reference"}, optional
        Solver quality policy. Default is "fast".
    geometry : Geometry or None, optional
        Scan context for geometry-specific iteration budgets. None uses
        conservative budgets without beam-specific information.
    levels : iterable of int or None, optional
        Explicit coarse-to-fine factors. None chooses them from the mode and grid.
    freeze : iterable of str, optional
        Parameter names to hold fixed, in addition to config.freeze.
    config : AlignConfig or None, optional
        Expert settings. None chooses the quality policy's defaults.

    Returns
    -------
    AlignmentPlan
        Configuration, resolution levels and pose-solver choice for the run.

    Raises
    ------
    ValueError
        If mode, quality or requested configuration settings are invalid.

    Examples
    --------
    >>> from tomojax import Grid
    >>> from tomojax.alignment import alignment_plan
    >>> plan = alignment_plan("pose", Grid(16, 16, 16, 1, 1, 1), quality="reference")
    >>> plan.levels
    (1,)
    """
    from tomojax.core.geometry.cone import is_cone_beam

    name = normalize_mode(mode)
    tier = normalize_quality(quality)
    expert = config is not None
    cfg = config or AlignConfig(quality=tier, pose_translation_frame="detector")
    if not expert:
        cfg = replace(cfg, **alignment_profile_policy(tier).to_dict())
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
        cfg = replace(
            cfg,
            gn_joint_iterations=(
                10 if tier == "fast" and is_cone_beam(geometry) else cfg.gn_joint_iterations
            ),
            **coupled_pose_defaults(),
        )
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

    Parameters
    ----------
    geometry : Geometry
        Geometry with calibrated cone-axis offset and detector roll.
    config : AlignConfig | None
        Remaining pose-stage settings, or None when no pose stages remain.
    axis_offset : float
        Calibrated absolute cone-axis offset in physical length units.
    detector_roll_deg : float
        Calibrated absolute detector roll in degrees.
    heights : tuple[float, ...]
        Physical z coordinates of the calibration slab centres.
    slab_offsets : tuple[float, ...]
        Sharpest physical axis offset at each slab height.
    """

    geometry: Geometry
    config: AlignConfig | None
    axis_offset: float
    detector_roll_deg: float
    heights: tuple[float, ...]
    slab_offsets: tuple[float, ...]

    def to_dict(self) -> dict[str, float | list[float]]:
        """The calibration, as alignment records keep it.

        Returns
        -------
        dict
            Absolute calibrated values and per-slab estimates.
        """
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

    Parameters
    ----------
    geometry : Geometry
        Scan geometry with its view angles and physical detector placement.
    grid : Grid
        Volume grid with physical voxel spacing.
    detector : Detector
        Detector shape and placement in the same physical units as the grid.
    projections : array-like
        Line integrals with shape (views, nv, nu), in geometry view order.
    config : AlignConfig
        Alignment settings and requested setup/pose stages.

    Returns
    -------
    ConeSetup or None
        Calibrated cone geometry and remaining pose stages; None if setup is not needed.

    Raises
    ------
    ValueError
        If explicitly requested setup parameters cannot be estimated for cone beams.
    """
    from tomojax.core.geometry.cone import is_cone_beam
    from tomojax.geometry import ConeGeometry
    from tomojax.recon import calibrate_cone_axis

    resolved = resolved_schedule_for_config(config)
    setup = set(resolved.active_geometry_dofs)
    if not is_cone_beam(geometry) or (not setup and resolved.name != "cor_then_pose"):
        return None
    if any(stage.objective_kind == "projection_consistency" for stage in resolved.stages):
        raise ValueError("spectral setup requires a parallel beam, not a cone beam")
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
    "normalize_mode",
    "pose_stages_only",
]
