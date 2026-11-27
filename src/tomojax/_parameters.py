"""The closed parameter choices shared by Python operations and their front ends."""

from importlib import import_module
from types import ModuleType
from typing import Any, Literal, cast, get_args

from tomojax.errors import InvalidParameter

type Method = Literal["auto", "fbp", "fourier", "cgls", "fista", "spdhg", "sirt", "mlem"]
type ReconMethod = Literal["fbp", "fourier", "cgls", "fista", "spdhg", "sirt", "mlem"]
type Filter = Literal["ramp", "hann", "shepp-logan"]
type Exterior = Literal["cylinder", "zero"]
DEFAULT_EXTERIOR: Exterior = "cylinder"

type RegionName = Literal["auto", "bbox", "cube", "cyl"]
type AlignMethod = Literal["joint", "opposite-views"]
type Fit = Literal["center", "tilt", "roll", "translations", "rotations"]
type Phantom = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
type GeometryName = Literal["parallel", "lamino", "cone"]
type TiltAbout = Literal["x", "z"]

type ProjectorBackend = Literal["auto", "jax", "cuda"]

# How a view between two flat sets blends them: by what a sample of the views shows
# of each set, or by its position between them.
type FlatWeighting = Literal["views", "position"]
# The blend when the caller does not choose one.
FLAT_WEIGHTS: FlatWeighting = "views"


def check_exterior(exterior: Exterior) -> None:
    """Validate the model of the scan beyond the detector."""
    if exterior not in get_args(Exterior.__value__):
        raise InvalidParameter("config.exterior", "exterior must be cylinder or zero")


def normalize_filter(name: str | None) -> Filter:
    """Normalize filter names and preserve the default ramp fallback."""
    normalized = str(name or "").strip().lower()
    normalized = normalized or "ramp"
    if normalized not in get_args(Filter.__value__):
        raise InvalidParameter("filter", f"Unknown filter {name}")
    return cast("Filter", normalized)


def flat_weighting(value: str) -> FlatWeighting:
    """Check how views between two flat sets blend them."""
    if value in ("views", "position"):
        return value
    raise InvalidParameter("flat_weights", f"flat_weights is 'views' or 'position', not {value!r}")


def method_fit(method: str | None, fit: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """The fit selection as a tuple of names; None selects the method's own fit.

    Opposite views fit the center alone; the joint fit (the default method) fits
    per-view translations.
    """
    if fit is None:
        return ("center",) if method == "opposite-views" else ("translations",)
    return (fit,) if isinstance(fit, str) else fit


def check_backend(backend: str, context: str) -> None:
    """Check the backend vocabulary before querying any device."""
    if backend not in get_args(ProjectorBackend.__value__):
        raise InvalidParameter("backend", f"{context}: backend must be 'auto', 'jax' or 'cuda'")


def simulation_distances(
    extent: float, source_to_axis: float | None, source_to_detector: float | None
) -> tuple[float, float]:
    """Resolve the synthetic cone's nominal source distances from the largest volume extent."""
    sod = 3.0 * extent if source_to_axis is None else float(source_to_axis)
    return sod, 1.5 * sod if source_to_detector is None else float(source_to_detector)


class AnnotationModule(ModuleType):
    """Load a numerical module only when a runtime consumer resolves an annotation."""

    def __getattr__(self, name: str) -> Any:
        return getattr(import_module(self.__name__), name)
