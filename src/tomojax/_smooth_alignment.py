"""One joint alignment workflow for declared acquisition geometries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax._planning import resolve_grid
from tomojax._scan import Reconstruction, describe, record_of, with_grid
from tomojax.alignment._integration import CentreSelection, fit_poses
from tomojax.backends.api import is_oom_error
from tomojax.errors import InvalidParameter
from tomojax.geometry import stack_view_poses
from tomojax.progress import warn

if TYPE_CHECKING:
    from tomojax._field_of_view import Cylinder
    from tomojax._parameters import RegionName
    from tomojax._region import RegionEstimate
    from tomojax._scan import Alignment, Scan
    from tomojax.alignment import AlignConfig
    from tomojax.alignment._integration import FitSchedule
    from tomojax.geometry import Grid


def align_poses(
    scan: Scan,
    *,
    fit: str | tuple[str, ...] = "translations",
    plan: JointSelection,
) -> Reconstruction:
    """Jointly fit one image and the selected global and per-view parameters."""
    from tomojax._calibration import publish_calibration

    grid = plan.grid
    if grid is not None:
        scan = with_grid(scan, grid)
    views = len(scan.angles)
    native = (scan.detector.du, scan.detector.dv)
    _, metadata, _, _ = describe(scan.geometry)
    try:
        volume, working, fitted, record = fit_poses(
            scan.projections,
            np.asarray(stack_view_poses(scan.geometry, views), np.float64),
            scan.grid,
            scan.detector,
            fit=fit,
            keep_grid=grid is not None,
            frames=plan.frames,
            detector_roll=float(cast("float", metadata.get("detector_roll", 0.0))),
            view_positions=scan.view_positions,
            schedule=plan.schedule,
            config=plan.config,
        )
    except Exception as exc:
        if grid is None or not is_oom_error(exc):
            raise
        raise MemoryError(
            f"tj.align ran out of device memory on the {grid.nx} x {grid.ny} x {grid.nz} grid it "
            "was given: an explicit grid= is used as given, at its full resolution. Omit grid= "
            "to let the fit bound its own working scale."
        ) from exc
    changes = record["calibration"]
    corrected, values = publish_calibration(scan, fitted, changes, native_pixel_size=native)
    corrected = with_grid(corrected, working)
    calibration = {name: {"value": values[name], "change": changes[name]} for name in changes}
    info = {**record, "method": "joint_volume_pose", "calibration": calibration}
    _warn_about_views(record["report"])
    source = record_of(corrected)
    source.geometry_metadata["alignment_info"] = info
    corrected = replace(corrected, source=source)
    return Reconstruction(volume, corrected.geometry, "joint_volume_pose", info, corrected)


def _warn_about_views(report: dict) -> None:
    """Name the views a user should look at before trusting the corrections."""
    isolated, unusual = report["isolated_views"], report["unusual_views"]
    parts = [text.rstrip(".") for text in report["advisories"]]
    if isolated:
        parts.append(f"{len(isolated)} views jump away from both neighbours ({_first(isolated)})")
    if unusual:
        parts.append(f"{len(unusual)} views fit unusually badly ({_first(unusual)})")
    if not parts:
        return
    warn(
        f"tj.align: {'. '.join(parts)}. Review info['report'] and "
        "https://tomojax.dev/align-a-scan/#check-the-result",
        stacklevel=3,
    )


def _first(views: list[int], count: int = 5) -> str:
    shown = ", ".join(map(str, views[:count]))
    return f"views {shown}, ..." if len(views) > count else f"views {shown}"


@dataclass(frozen=True)
class JointSelection:
    """Working support, calibrated ray frames and the fit's execution schedule."""

    grid: Grid | None
    config: AlignConfig
    cylinder: Cylinder | None
    info: dict[str, Any]
    schedule: FitSchedule
    frames: np.ndarray | None


def resolve_alignment(
    scan: Scan,
    *,
    fit: str | tuple[str, ...] = "translations",
    method: str | None = None,
    grid: Grid | None = None,
    region: RegionName | RegionEstimate | None = None,
    config: AlignConfig | None = None,
    shape: tuple[int, int, int] | None = None,
    **settings: Any,
) -> JointSelection | CentreSelection:
    """Resolve the method-specific work without sampling projections."""
    from tomojax.alignment import AlignConfig
    from tomojax.alignment._integration import (
        cone_frames,
        fit_schedule,
        opposite_samples,
        parse_fit,
    )

    supplied = {name: value for name, value in settings.items() if value is not None}
    if method == "opposite-views":
        extras = sorted(
            key
            for key, value in {
                **supplied,
                "config": config,
                "grid": grid,
                "region": region,
                "shape": shape,
            }.items()
            if value is not None
        )
        if extras:
            raise InvalidParameter(
                extras[0],
                'method="opposite-views" fits no image and takes no joint options '
                f"({', '.join(extras)})",
            )
        if fit not in ("center", ("center",)):
            raise InvalidParameter(
                "fit", f'method="opposite-views" fits fit="center" alone, not {fit!r}'
            )
        return opposite_samples(scan.geometry)
    if method not in (None, "joint"):
        raise InvalidParameter("method", f"method is 'joint' or 'opposite-views', not {method!r}")
    _ = parse_fit(fit)
    config = replace(AlignConfig() if config is None else config, **supplied)
    grid, cylinder, info = resolve_grid(scan, grid, region, shape=shape)
    if grid is not None:
        scan = with_grid(scan, grid)
    acquisition = (
        {} if scan.source is None else scan.source.geometry_metadata.get("acquisition", {})
    )
    if acquisition.get("geometry_status") == "not_recorded":
        raise ValueError(
            "alignment requires declared acquisition geometry: this file records no rotation "
            "axis; pass geometry=... to tomojax.load"
        )
    frames = cone_frames(scan.geometry)
    cone = None
    if frames is not None:
        declared = np.asarray(stack_view_poses(scan.with_poses(None).geometry, len(scan.angles)))
        cone = (declared, frames)
    schedule = fit_schedule(
        scan.grid, scan.detector, cone, keep_grid=grid is not None, config=config
    )
    return JointSelection(grid, config, cylinder, info, schedule, frames)


def transfer_alignment(
    train: Scan,
    test: Scan,
    alignment: Alignment,
    *,
    training_segments: np.ndarray,
    held_segments: np.ndarray,
    local: bool,
) -> Scan:
    """Apply fitted globals and segment-interpolated local updates to unseen views."""
    from scipy.spatial.transform import Rotation

    from tomojax._calibration import publish_calibration

    calibration = alignment.info["calibration"]
    if not isinstance(calibration, dict):
        raise TypeError("the alignment records no calibration changes to transfer")
    corrections = {key: entry["change"] for key, entry in calibration.items()}
    fitted = None
    if local:
        # publish_calibration composes old * delta and adds lab translations.
        delta = np.array(alignment.poses, dtype=float, copy=True)
        previous = train.poses
        rotations = Rotation.from_euler("YXZ", delta[:, [1, 0, 2]], degrees=True)
        if previous is not None:
            rotations = (
                Rotation.from_euler("YXZ", previous[:, [1, 0, 2]], degrees=True).inv() * rotations
            )
            delta[:, 3:] -= previous[:, 3:]
        delta[:, :3] = rotations.as_euler("YXZ")[:, [1, 0, 2]]
        fitted = np.empty((len(held_segments), 6))
        train_at, test_at = train.view_positions, test.view_positions
        if train_at is None or test_at is None:
            raise ValueError("transferring local motion needs each view's acquisition position")
        for group in np.unique(held_segments):
            a, b = training_segments == group, held_segments == group
            for col in range(6):
                fitted[b, col] = np.interp(test_at[b], train_at[a], delta[a, col])
    detector = test.detector
    return publish_calibration(
        test, fitted, corrections, native_pixel_size=(detector.du, detector.dv)
    )[0]
