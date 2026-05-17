"""Compatibility exports for contrast conversion helpers.

The implementation is owned by :mod:`tomojax.io`; new production callers should
import these helpers from ``tomojax.io``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

ArrayLike = Any


def _contrast_impl() -> Any:
    return import_module("tomojax.io._contrast")


def transmission_to_absorption(
    transmission: ArrayLike,
    *,
    min_intensity: float = 1e-6,
) -> Any:
    """Convert transmission (intensity) data into absorption (-log I)."""
    return _contrast_impl().transmission_to_absorption(
        transmission,
        min_intensity=min_intensity,
    )


def absorption_to_transmission(
    absorption: ArrayLike,
    *,
    max_absorption: float | None = None,
) -> Any:
    """Convert absorption data back to transmission (exp(-absorption))."""
    return _contrast_impl().absorption_to_transmission(
        absorption,
        max_absorption=max_absorption,
    )


def flat_dark_to_absorption(
    projections: ArrayLike,
    flats: ArrayLike,
    darks: ArrayLike | None = None,
    *,
    min_intensity: float = 1e-6,
    transmission_min: float | None = None,
) -> Any:
    """Apply flat/dark-field correction and convert to absorption values."""
    return _contrast_impl().flat_dark_to_absorption(
        projections,
        flats,
        darks,
        min_intensity=min_intensity,
        transmission_min=transmission_min,
    )


def flat_dark_to_transmission(
    projections: ArrayLike,
    flats: ArrayLike,
    darks: ArrayLike | None = None,
    *,
    denominator_min: float = 1e-6,
    clip_min: float | None = None,
) -> Any:
    """Apply flat/dark-field correction and return normalized transmission."""
    return _contrast_impl().flat_dark_to_transmission(
        projections,
        flats,
        darks,
        denominator_min=denominator_min,
        clip_min=clip_min,
    )


__all__ = [
    "absorption_to_transmission",
    "flat_dark_to_absorption",
    "flat_dark_to_transmission",
    "transmission_to_absorption",
]
