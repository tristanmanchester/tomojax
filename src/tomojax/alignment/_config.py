"""Every setting of :func:`tomojax.align`, with its default, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import operator
from typing import Literal

from tomojax.errors import InvalidParameter

type Interpolation = Literal["linear", "cubic"]
type Coverage = Literal["coverage"]
type DetectorColumns = Literal["none", "fixed", "transient", "both"]
type Motion = Literal["independent", "smooth"]


def _view_ranges(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Read half-open input indices from Python, TOML or the wire format."""
    parsed: list[tuple[int, int]] = []
    try:
        for start, stop in ranges:
            if isinstance(start, bool) or isinstance(stop, bool):
                raise TypeError
            a, b = operator.index(start), operator.index(stop)
            if not 0 <= a < b:
                raise ValueError
            parsed.append((a, b))
    except (TypeError, ValueError) as exc:
        raise InvalidParameter(
            "exclude_views", "exclude_views needs integer pairs with 0 <= start < stop"
        ) from exc
    return tuple(parsed)


@dataclass(frozen=True, kw_only=True)
class AlignConfig:
    """The settings of :func:`tomojax.align`.

    These fields configure the joint volume-and-geometry solver. Passing
    ``config`` or any of the six keyword overrides (``motion``, ``iterations``,
    ``image_iterations``, ``tolerance``, ``interpolation`` and ``resolution``)
    to ``tj.align(scan, fit="center")`` runs the joint solver instead of the
    direct axial center measurement. ``info["config"]`` records the values a
    joint call used, with the ones that depend on the scan resolved. Lengths in
    pixels are in the working detector's pixels unless a field says native;
    angles are in degrees.

    Parameters
    ----------
    motion : {"independent", "smooth"}, optional
        How each view's corrections may differ from its neighbors'. "independent" (the
        default) leaves every view free: no term ties a view's translations or rotations to
        its neighbors', and only the rotations' size prior remains. "smooth" lets the
        translations drift freely along a spline and step from one view to the next only
        where the data call for a slip, and ties each view's rotations to its neighbors';
        use it for a steady stage, where per-view error that is not motion should not be
        followed.
    iterations : int, optional
        Gauss-Newton iterations at the working resolution. Default is 8.
    coarse_iterations : int, optional
        Gauss-Newton iterations at each coarser level. Default is 12.
    levels : tuple of int, optional
        Each level's binning of the working detector, coarsest first, ending at 1.
        Default is ``(2, 1)``; ``(1,)`` fits at the working resolution alone.
    image_iterations : int, optional
        Conjugate-gradient iterations of the image solve in each Gauss-Newton
        iteration. Default is 16.
    image_tolerance : float, optional
        Relative residual at which an image solve may stop before its count.
        Default is 0.03.
    tolerance : float, optional
        A level stops when two successive pose proposals move the views by less
        than this, in working pixels; 0 runs every level to its iterations.
        Default is 0.01.
    min_iterations : int, optional
        The fewest iterations before ``tolerance`` may stop a level. Default is 4.
    resolution : int, optional
        The working detector's longest side, in pixels. Default is 320.
    max_voxels : int, optional
        The working grid's largest number of voxels. Default is 16 000 000.
    interpolation : {"linear", "cubic"} or None, optional
        The projector's in-plane interpolation in the fit. None chooses cubic for
        a fit with an angle on parallel rays and linear otherwise.
    refine_iterations : int, optional
        Iterations of the final per-view refinement with the image held; 0 turns
        it off. Default is 5.
    initial_image_iterations : int, optional
        FISTA iterations for the first level's starting image. Default is 5.
    slip_scale : float, optional
        Under ``motion="smooth"``, the view-to-view step, in native pixels, at which the pull
        towards the neighbors is strongest; larger steps are followed as slips, and the
        rotations' tie to their neighbors is inversely proportional to it. Default is 0.02.
    knot_spacing : int, optional
        Under ``motion="smooth"``, views between the drift's spline knots. Default is 10.
    rotation_sigma : float, optional
        The per-view rotations' size prior, in degrees. Default is 0.5.
    loss_width : float, optional
        Where the robust loss turns from quadratic to linear, in noise standard
        deviations. Default is 2.5.
    loss_contrast_floor : float, optional
        The least loss width, as a fraction of the data's contrast. Default is 0.5.
    exclude_views : tuple of (int, int), optional
        Zero-based, half-open view ranges to leave out of the joint fit. A pair
        (start, stop) excludes start through stop minus one. Their pose updates
        are interpolated in acquisition time within each trajectory. All input
        projections are retained in the result. Default is empty.
    detector_columns : {"none", "fixed", "transient", "both"}, optional
        Column errors to model in the joint fit. "fixed" estimates offsets shared
        by all views. "transient" estimates row-common disturbances per view and
        weights their residual variance. "both" fits both terms. Default is "none".
    image_tv_weight : float or {"coverage"}, optional
        Total-variation weight on the fit's image, as a fraction of its contrast;
        0 (default) leaves the image unregularized. "coverage" scales the penalty by
        measured noise and changes in detector coverage between neighboring voxels.
    image_tv_huber : float, optional
        The total variation's Huber width, as a fraction of the contrast.
        Default is 1.0.
    margin : float, optional
        How far the fit's image extends past each end of the grid, as a fraction
        of its height. Default is 1/16.
    refine_smoothing : float, optional
        The refinement's edge filter width, in native pixels. Default is 8.0.
    refine_neighbors : int, optional
        Views on each side that set a view's expected residual in the
        refinement. Default is 5.
    """

    motion: Motion = "independent"
    iterations: int = 8
    coarse_iterations: int = 12
    levels: tuple[int, ...] = (2, 1)
    image_iterations: int = 16
    image_tolerance: float = field(default=0.03, metadata={"unit": "1"})
    tolerance: float = field(default=0.01, metadata={"unit": "working_pixel"})
    min_iterations: int = 4
    resolution: int = field(default=320, metadata={"unit": "working_pixel"})
    max_voxels: int = field(default=16_000_000, metadata={"unit": "voxel"})
    interpolation: Interpolation | None = None
    refine_iterations: int = 5
    initial_image_iterations: int = 5
    slip_scale: float = field(default=0.02, metadata={"unit": "native_pixel"})
    knot_spacing: int = field(default=10, metadata={"unit": "view"})
    rotation_sigma: float = field(default=0.5, metadata={"unit": "degree"})
    loss_width: float = field(default=2.5, metadata={"unit": "noise_sigma"})
    loss_contrast_floor: float = field(default=0.5, metadata={"unit": "1"})
    exclude_views: tuple[tuple[int, int], ...] = ()
    detector_columns: DetectorColumns = "none"
    image_tv_weight: float | Coverage = field(default=0.0, metadata={"unit": "1"})
    image_tv_huber: float = field(default=1.0, metadata={"unit": "1"})
    margin: float = field(default=1 / 16, metadata={"unit": "1"})
    refine_smoothing: float = field(default=8.0, metadata={"unit": "native_pixel"})
    refine_neighbors: int = field(default=5, metadata={"unit": "view"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "exclude_views", _view_ranges(self.exclude_views))
        self._validate_weights()
        for name, value in (
            ("iterations", self.iterations),
            ("coarse_iterations", self.coarse_iterations),
            ("image_iterations", self.image_iterations),
            ("min_iterations", self.min_iterations),
            ("initial_image_iterations", self.initial_image_iterations),
            ("refine_iterations", self.refine_iterations),
            ("refine_neighbors", self.refine_neighbors),
        ):
            if operator.index(value) < 0:
                raise InvalidParameter(name, f"{name} must be nonnegative")
        for name, value in (
            ("resolution", self.resolution),
            ("knot_spacing", self.knot_spacing),
            ("max_voxels", self.max_voxels),
        ):
            if operator.index(value) < 1:
                raise InvalidParameter(name, f"{name} must be positive")
        if self.detector_columns not in ("none", "fixed", "transient", "both"):
            raise InvalidParameter("detector_columns", "choose none, fixed, transient or both")
        if self.motion not in ("independent", "smooth"):
            raise InvalidParameter(
                "motion", f'motion is "independent" or "smooth", not {self.motion!r}'
            )
        if not self.levels or self.levels[-1] != 1 or any(b < 1 for b in self.levels):
            raise InvalidParameter(
                "levels", f"levels are binnings of at least 1 ending at 1, not {self.levels}"
            )
        if self.interpolation not in (None, "linear", "cubic"):
            raise InvalidParameter(
                "interpolation", f"interpolation is 'linear' or 'cubic', not {self.interpolation!r}"
            )

    def _validate_weights(self) -> None:
        """Check nonnegative regularization scales, including the measured coverage option."""
        for name, value in (
            ("margin", self.margin),
            ("image_tolerance", self.image_tolerance),
            ("tolerance", self.tolerance),
            (
                "image_tv_weight",
                self.image_tv_weight if self.image_tv_weight != "coverage" else 0.0,
            ),
            ("loss_contrast_floor", self.loss_contrast_floor),
            ("refine_smoothing", self.refine_smoothing),
        ):
            if not math.isfinite(value) or value < 0:
                raise InvalidParameter(name, f"{name} must be finite and nonnegative")


def resolve_align_config(config: AlignConfig, fit: tuple[str, ...], *, cone: bool) -> AlignConfig:
    """Choose ``interpolation`` from ``fit`` and beam type when left as None."""
    if config.interpolation is not None:
        return config
    angular = any(name in fit for name in ("tilt", "roll", "rotations"))
    return replace(config, interpolation="cubic" if angular and not cone else "linear")


__all__ = ["AlignConfig"]
