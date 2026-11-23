"""Every setting of :func:`tomojax.align`, with its default, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import operator
from typing import Literal

from tomojax.errors import InvalidParameter

type Interpolation = Literal["linear", "cubic"]
type Motion = Literal["independent", "smooth"]


@dataclass(frozen=True, kw_only=True)
class AlignConfig:
    """The settings of :func:`tomojax.align`.

    ``tj.align``'s keywords ``motion``, ``iterations``, ``image_iterations``,
    ``tolerance``, ``interpolation`` and ``resolution`` replace the fields of the same name; the
    rest are set here. ``info["config"]`` records the values a call used, with
    the ones that depend on the scan resolved. Lengths in pixels are in the
    working detector's pixels unless a field says native; angles are in degrees.

    Parameters
    ----------
    motion : {"independent", "smooth"}, optional
        How each view's corrections may differ from its neighbours'. "independent" (the
        default) leaves every view free: no term ties a view's translations or rotations to
        its neighbours', and only the rotations' size prior remains. "smooth" lets the
        translations drift freely along a spline and step from one view to the next only
        where the data call for a slip, and ties each view's rotations to its neighbours';
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
        towards the neighbours is strongest; larger steps are followed as slips, and the
        rotations' tie to their neighbours is inversely proportional to it. Default is 0.02.
    knot_spacing : int, optional
        Under ``motion="smooth"``, views between the drift's spline knots. Default is 10.
    rotation_sigma : float, optional
        The per-view rotations' size prior, in degrees. Default is 0.5.
    loss_width : float, optional
        Where the robust loss turns from quadratic to linear, in noise standard
        deviations. Default is 2.5.
    loss_contrast_floor : float, optional
        The least loss width, as a fraction of the data's contrast. Default is 0.5.
    image_tv_weight : float, optional
        Total-variation weight on the fit's image, as a fraction of its contrast;
        0 leaves the image unregularized. Default is 0.0.
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
    image_tv_weight: float = field(default=0.0, metadata={"unit": "1"})
    image_tv_huber: float = field(default=1.0, metadata={"unit": "1"})
    margin: float = field(default=1 / 16, metadata={"unit": "1"})
    refine_smoothing: float = field(default=8.0, metadata={"unit": "native_pixel"})
    refine_neighbors: int = field(default=5, metadata={"unit": "view"})

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "coarse_iterations",
            "image_iterations",
            "min_iterations",
            "initial_image_iterations",
            "refine_iterations",
            "refine_neighbors",
        ):
            if operator.index(getattr(self, name)) < 0:
                raise InvalidParameter(name, f"{name} must be nonnegative")
        for name in ("resolution", "knot_spacing", "max_voxels"):
            if operator.index(getattr(self, name)) < 1:
                raise InvalidParameter(name, f"{name} must be positive")
        for name in (
            "margin",
            "image_tolerance",
            "tolerance",
            "image_tv_weight",
            "loss_contrast_floor",
            "refine_smoothing",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise InvalidParameter(name, f"{name} must be finite and nonnegative")
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


def resolve_align_config(config: AlignConfig, fit: tuple[str, ...], *, cone: bool) -> AlignConfig:
    """Choose ``interpolation`` from ``fit`` and beam type when left as None."""
    if config.interpolation is not None:
        return config
    angular = any(name in fit for name in ("tilt", "roll", "rotations"))
    return replace(config, interpolation="cubic" if angular and not cone else "linear")


__all__ = ["AlignConfig"]
