from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from tomojax.io._json import JsonValue

LOG = logging.getLogger(__name__)


PREPROCESS_SCHEMA_VERSION = 1


_PROJECTION_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
)


_ANGLE_PATH = "/entry/sample/transformations/rotation_angle"


_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"


_OUTPUT_DTYPES: dict[str, str] = {"float32": "float32", "float64": "float64"}


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for raw NXtomo flat/dark correction."""

    output_domain: str = "absorption"
    log: bool | None = None
    epsilon: float = 1e-6
    clip_min: float | None = None
    output_dtype: str = "float32"
    data_path: str | None = None
    angles_path: str | None = None
    image_key_path: str | None = None
    assume_dark_field: float | None = None
    assume_flat_field: float | None = None
    select_views: str | None = None
    reject_views: str | None = None
    select_views_file: str | Path | None = None
    reject_views_file: str | Path | None = None
    auto_reject: str = "off"
    outlier_z_threshold: float = 6.0
    crop: str | None = None


@dataclass(slots=True)
class PreprocessResult:
    """Summary of a preprocessing run."""

    sample_count: int
    flat_count: int
    dark_count: int
    output_shape: tuple[int, int, int]
    output_domain: str
    warning_counts: dict[str, int]
    provenance: dict[str, JsonValue]


def _output_domain_from_config(config: PreprocessConfig) -> str:
    """Resolve the production output domain, keeping the old log knob explicit."""
    if config.log is not None:
        return "absorption" if config.log else "transmission"
    return str(config.output_domain).strip().lower()


def validate_preprocess_numeric_config(config: PreprocessConfig) -> tuple[np.dtype, str]:
    """Validate preprocessing output-domain and numeric correction settings."""
    output_domain = _output_domain_from_config(config)
    if output_domain not in {"absorption", "transmission"}:
        raise ValueError("output_domain must be one of: absorption, transmission")
    if not np.isfinite(config.epsilon) or config.epsilon <= 0:
        raise ValueError("epsilon must be a positive finite value")
    if config.clip_min is not None and (not np.isfinite(config.clip_min) or config.clip_min <= 0):
        raise ValueError("clip_min must be a positive finite value when provided")
    if config.output_dtype not in _OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_OUTPUT_DTYPES))
        raise ValueError(f"output_dtype must be one of: {allowed}")
    return np.dtype(_OUTPUT_DTYPES[config.output_dtype]), output_domain


def _validate_config(config: PreprocessConfig) -> tuple[np.dtype, str]:
    output_dtype, output_domain = validate_preprocess_numeric_config(config)
    auto_reject = str(config.auto_reject).strip().lower()
    if auto_reject not in {"off", "nonfinite", "outliers", "both"}:
        raise ValueError("auto_reject must be one of: off, nonfinite, outliers, both")
    if not np.isfinite(config.outlier_z_threshold) or config.outlier_z_threshold <= 0:
        raise ValueError("outlier_z_threshold must be a positive finite value")
    return output_dtype, output_domain
