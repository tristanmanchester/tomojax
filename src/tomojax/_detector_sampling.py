"""Recorded means of line integrals, separate from camera acquisition binning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tomojax.io import ProjectionDataset


def sampling_factors(source: ProjectionDataset | None) -> tuple[int, int]:
    """Read explicit line-integral mean factors in detector (u, v) order.

    Absent provenance describes the supplied pixel-centre rays. Acquisition
    pitch, hardware binning, and intensity averages do not establish this
    post-log operation. An explicit unsupported or malformed record is an
    error rather than a reason to silently fit a different data operator.
    """
    if source is None or "detector_sampling" not in source.geometry_metadata:
        return 1, 1
    description = source.geometry_metadata["detector_sampling"]
    if (
        not isinstance(description, Mapping)
        or description.get("kind") != "line_integral_block_mean"
    ):
        raise ValueError("detector_sampling must describe a line_integral_block_mean")
    try:
        factors = tuple(description["factors_uv"])
    except (KeyError, TypeError) as exc:
        raise ValueError("detector_sampling factors_uv must contain two positive integers") from exc
    if len(factors) != 2 or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in factors
    ):
        raise ValueError("detector_sampling factors_uv must contain two positive integers")
    return int(factors[0]), int(factors[1])


def block_mean_record(source: ProjectionDataset, factors: tuple[int, int]) -> ProjectionDataset:
    """Clone a dataset record and compose a known digital line-integral mean."""
    previous = sampling_factors(source)
    return replace(
        source,
        geometry_metadata={
            **source.geometry_metadata,
            "detector_sampling": {
                "kind": "line_integral_block_mean",
                "factors_uv": [int(old * new) for old, new in zip(previous, factors, strict=True)],
            },
        },
    )
