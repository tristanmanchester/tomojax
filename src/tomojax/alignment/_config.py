"""Every setting of :func:`tomojax.align`, with its default, in one place."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

type Interpolation = Literal["linear", "cubic"]


@dataclass(frozen=True, kw_only=True)
class AlignConfig:
    """The settings of :func:`tomojax.align`.

    ``tj.align``'s keywords ``iterations``, ``image_iterations``, ``tolerance``,
    ``interpolation`` and ``resolution`` replace the fields of the same name; the
    rest are set here. ``info["config"]`` records the values a call used, with
    the ones that depend on the scan resolved. Lengths in pixels are in the
    working detector's pixels unless a field says native; angles are in degrees.

    Parameters
    ----------
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
    innovation_scale : float, optional
        The size of a view-to-view step the motion prior counts as an event, in
        native pixels. Default is 0.02.
    knot_spacing : int, optional
        Views between the smooth drift's spline knots. Default is 10.
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

    iterations: int = 8
    coarse_iterations: int = 12
    levels: tuple[int, ...] = (2, 1)
    image_iterations: int = 16
    image_tolerance: float = 0.03
    tolerance: float = 0.01
    min_iterations: int = 4
    resolution: int = 320
    max_voxels: int = 16_000_000
    interpolation: Interpolation | None = None
    refine_iterations: int = 5
    initial_image_iterations: int = 5
    innovation_scale: float = 0.02
    knot_spacing: int = 10
    rotation_sigma: float = 0.5
    loss_width: float = 2.5
    loss_contrast_floor: float = 0.5
    image_tv_weight: float = 0.0
    image_tv_huber: float = 1.0
    margin: float = 1 / 16
    refine_smoothing: float = 8.0
    refine_neighbors: int = 5

    def resolved(self, fit: tuple[str, ...], *, cone: bool) -> AlignConfig:
        """The config with what depends on the scan chosen: the interpolation for ``fit``.

        Angular columns need smooth interpolation as sample points rotate through voxel
        cells; centre and translation fits, and cone rays, keep the cheaper linear basis.

        Parameters
        ----------
        fit : tuple of str
            The parameters fitted, as ``tj.align``'s ``fit``.
        cone : bool
            Whether the rays come from a point source.

        Returns
        -------
        AlignConfig
            This config, or a copy with ``interpolation`` chosen.
        """
        if self.interpolation is not None:
            return self
        angular = any(name in fit for name in ("tilt", "roll", "rotations"))
        return replace(self, interpolation="cubic" if angular and not cone else "linear")

    def __post_init__(self) -> None:
        if not self.levels or self.levels[-1] != 1 or any(b < 1 for b in self.levels):
            raise ValueError(f"levels are binnings of at least 1 ending at 1, not {self.levels}")
        if self.interpolation not in (None, "linear", "cubic"):
            raise ValueError(f"interpolation is 'linear' or 'cubic', not {self.interpolation!r}")


__all__ = ["AlignConfig"]
