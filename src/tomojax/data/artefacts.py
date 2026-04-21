from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypedDict

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

IntensityDriftMode = Literal["none", "linear", "sinusoidal"]


class ArtefactMetadata(TypedDict, total=False):
    enabled: bool
    seed: int
    order: list[str]
    config: JsonObject
    poisson_scale: float
    gaussian_sigma: float
    dead_pixel_indices: list[int]
    hot_pixel_indices: list[int]
    zinger_indices: list[int]
    stripe_columns: list[int]
    stripe_gains: list[float]
    dropped_view_indices: list[int]
    intensity_drift_factors: list[float]
    detector_blur_sigma: float
    dead_pixel_count: int
    hot_pixel_count: int
    zinger_count: int
    stripe_count: int
    dropped_view_count: int


@dataclass(frozen=True, slots=True)
class SimulationArtefacts:
    """Configurable projection-domain artefacts for synthetic tomography data."""

    poisson_scale: float = 0.0
    gaussian_sigma: float = 0.0
    dead_pixel_fraction: float = 0.0
    dead_pixel_value: float = 0.0
    hot_pixel_fraction: float = 0.0
    hot_pixel_value: float = 1.0
    zinger_fraction: float = 0.0
    zinger_value: float = 1.0
    stripe_fraction: float = 0.0
    stripe_gain_sigma: float = 0.0
    dropped_view_fraction: float = 0.0
    dropped_view_fill: float = 0.0
    detector_blur_sigma: float = 0.0
    intensity_drift_amplitude: float = 0.0
    intensity_drift_mode: IntensityDriftMode = "none"

    def to_dict(self) -> JsonObject:
        return {key: _json_scalar(value) for key, value in asdict(self).items()}

    def has_enabled(self) -> bool:
        return (
            self.poisson_scale > 0.0
            or self.gaussian_sigma > 0.0
            or self.dead_pixel_fraction > 0.0
            or self.hot_pixel_fraction > 0.0
            or self.zinger_fraction > 0.0
            or (self.stripe_fraction > 0.0 and self.stripe_gain_sigma > 0.0)
            or self.dropped_view_fraction > 0.0
            or self.detector_blur_sigma > 0.0
            or (
                self.intensity_drift_amplitude != 0.0
                and self.intensity_drift_mode != "none"
            )
        )


ARTEFACT_ORDER = [
    "detector_blur",
    "intensity_drift",
    "stripes",
    "poisson_noise",
    "gaussian_noise",
    "dead_pixels",
    "hot_pixels",
    "zingers",
    "dropped_views",
]

_RNG_OFFSETS = {
    "poisson_noise": 1_009,
    "gaussian_noise": 2_003,
    "dead_pixels": 3_001,
    "hot_pixels": 4_009,
    "zingers": 5_003,
    "stripes": 6_007,
    "dropped_views": 7_001,
}


def artefacts_from_legacy_noise(noise: str, noise_level: float) -> SimulationArtefacts:
    """Map the legacy ``noise`` / ``noise_level`` pair to artefact config."""

    level = float(noise_level)
    if level <= 0.0:
        return SimulationArtefacts()
    if noise == "gaussian":
        return SimulationArtefacts(gaussian_sigma=level)
    if noise == "poisson":
        return SimulationArtefacts(poisson_scale=level)
    return SimulationArtefacts()


def normalise_simulation_artefacts(
    artefacts: SimulationArtefacts | None,
    *,
    noise: str,
    noise_level: float,
) -> SimulationArtefacts:
    """Return explicit artefacts, or legacy noise mapped into artefact fields."""

    if artefacts is not None:
        validate_simulation_artefacts(artefacts)
        return artefacts
    return artefacts_from_legacy_noise(noise, noise_level)


def validate_simulation_artefacts(artefacts: SimulationArtefacts | None) -> None:
    """Raise if an artefact configuration contains invalid parameter values."""

    _validate_config(artefacts or SimulationArtefacts())


def apply_simulation_artefacts(
    projections: jnp.ndarray,
    artefacts: SimulationArtefacts | None,
    *,
    seed: int,
) -> tuple[jnp.ndarray, ArtefactMetadata]:
    """Apply deterministic, composable projection artefacts.

    The forward projector remains pure JAX. Artefacts are intentionally applied
    as a host-side post-processing step, matching the existing simulator noise
    path while keeping all randomness explicit and reproducible.
    """

    cfg = artefacts or SimulationArtefacts()
    validate_simulation_artefacts(cfg)
    metadata: ArtefactMetadata = {
        "enabled": cfg.has_enabled(),
        "seed": int(seed),
        "order": list(ARTEFACT_ORDER),
        "config": cfg.to_dict(),
    }
    if not cfg.has_enabled():
        return projections, metadata

    original_dtype = np.asarray(projections).dtype
    out = np.asarray(projections, dtype=np.float32).copy()
    if out.ndim != 3:
        raise ValueError("projections must have shape (n_views, nv, nu)")

    n_views, nv, nu = out.shape

    if cfg.detector_blur_sigma > 0.0:
        sigma = float(cfg.detector_blur_sigma)
        out = gaussian_filter(out, sigma=sigma, axes=(1, 2), mode="nearest")
        metadata["detector_blur_sigma"] = sigma

    if cfg.intensity_drift_amplitude != 0.0 and cfg.intensity_drift_mode != "none":
        factors = _intensity_drift_factors(
            n_views,
            amplitude=float(cfg.intensity_drift_amplitude),
            mode=cfg.intensity_drift_mode,
        )
        out *= factors[:, None, None]
        metadata["intensity_drift_factors"] = [float(v) for v in factors.tolist()]

    if cfg.stripe_fraction > 0.0 and cfg.stripe_gain_sigma > 0.0:
        count = _fraction_count(nu, cfg.stripe_fraction)
        rng = _rng(seed, "stripes")
        columns = np.sort(rng.choice(nu, size=count, replace=False).astype(np.int32))
        gains = rng.normal(loc=1.0, scale=float(cfg.stripe_gain_sigma), size=count).astype(
            np.float32
        )
        gains = np.maximum(gains, 0.0)
        out[:, :, columns] *= gains[None, None, :]
        metadata["stripe_columns"] = [int(v) for v in columns.tolist()]
        metadata["stripe_gains"] = [float(v) for v in gains.tolist()]
        metadata["stripe_count"] = int(count)

    if cfg.poisson_scale > 0.0:
        scale = float(cfg.poisson_scale)
        rng = _rng(seed, "poisson_noise")
        lam = np.maximum(out, 0.0) * scale
        out = rng.poisson(lam=lam).astype(np.float32) / max(scale, 1e-6)
        metadata["poisson_scale"] = scale

    if cfg.gaussian_sigma > 0.0:
        sigma = float(cfg.gaussian_sigma)
        rng = _rng(seed, "gaussian_noise")
        out += rng.normal(scale=sigma, size=out.shape).astype(np.float32)
        metadata["gaussian_sigma"] = sigma

    if cfg.dead_pixel_fraction > 0.0:
        count = _fraction_count(nv * nu, cfg.dead_pixel_fraction)
        rng = _rng(seed, "dead_pixels")
        indices = np.sort(rng.choice(nv * nu, size=count, replace=False).astype(np.int32))
        rows, cols = np.unravel_index(indices, (nv, nu))
        out[:, rows, cols] = float(cfg.dead_pixel_value)
        metadata["dead_pixel_indices"] = [int(v) for v in indices.tolist()]
        metadata["dead_pixel_count"] = int(count)

    if cfg.hot_pixel_fraction > 0.0:
        count = _fraction_count(nv * nu, cfg.hot_pixel_fraction)
        rng = _rng(seed, "hot_pixels")
        indices = np.sort(rng.choice(nv * nu, size=count, replace=False).astype(np.int32))
        rows, cols = np.unravel_index(indices, (nv, nu))
        out[:, rows, cols] = float(cfg.hot_pixel_value)
        metadata["hot_pixel_indices"] = [int(v) for v in indices.tolist()]
        metadata["hot_pixel_count"] = int(count)

    if cfg.zinger_fraction > 0.0:
        count = _fraction_count(n_views * nv * nu, cfg.zinger_fraction)
        rng = _rng(seed, "zingers")
        indices = np.sort(
            rng.choice(n_views * nv * nu, size=count, replace=False).astype(np.int32)
        )
        out.reshape(-1)[indices] += float(cfg.zinger_value)
        metadata["zinger_indices"] = [int(v) for v in indices.tolist()]
        metadata["zinger_count"] = int(count)

    if cfg.dropped_view_fraction > 0.0:
        count = _fraction_count(n_views, cfg.dropped_view_fraction)
        rng = _rng(seed, "dropped_views")
        indices = np.sort(rng.choice(n_views, size=count, replace=False).astype(np.int32))
        out[indices, :, :] = float(cfg.dropped_view_fill)
        metadata["dropped_view_indices"] = [int(v) for v in indices.tolist()]
        metadata["dropped_view_count"] = int(count)

    return jnp.asarray(out.astype(original_dtype, copy=False)), metadata


def _validate_config(cfg: SimulationArtefacts) -> None:
    for name in (
        "dead_pixel_fraction",
        "hot_pixel_fraction",
        "zinger_fraction",
        "stripe_fraction",
        "dropped_view_fraction",
    ):
        value = float(getattr(cfg, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    for name in (
        "poisson_scale",
        "gaussian_sigma",
        "stripe_gain_sigma",
        "detector_blur_sigma",
    ):
        value = float(getattr(cfg, name))
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    if cfg.intensity_drift_mode not in {"none", "linear", "sinusoidal"}:
        raise ValueError(
            "intensity_drift_mode must be one of 'none', 'linear', or 'sinusoidal'"
        )


def _rng(seed: int, artefact_name: str) -> np.random.Generator:
    offset = _RNG_OFFSETS[artefact_name]
    return np.random.default_rng((int(seed) + offset) % (2**32))


def _fraction_count(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0.0:
        return 0
    return min(total, max(1, int(np.ceil(float(total) * float(fraction)))))


def _intensity_drift_factors(
    n_views: int,
    *,
    amplitude: float,
    mode: IntensityDriftMode,
) -> np.ndarray:
    if n_views <= 0:
        return np.zeros((0,), dtype=np.float32)
    if n_views == 1:
        return np.asarray([max(0.0, 1.0 + amplitude)], dtype=np.float32)

    t = np.linspace(0.0, 1.0, n_views, dtype=np.float32)
    if mode == "linear":
        factors = 1.0 + amplitude * (2.0 * t - 1.0)
    elif mode == "sinusoidal":
        factors = 1.0 + amplitude * np.sin(2.0 * np.pi * t)
    else:
        factors = np.ones((n_views,), dtype=np.float32)
    return np.maximum(factors, 0.0).astype(np.float32)


def _json_scalar(value: object) -> JsonValue:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    return str(value)
