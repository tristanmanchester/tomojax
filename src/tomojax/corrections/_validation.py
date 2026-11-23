"""Correction inputs that can be checked before reading the detector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.errors import InvalidParameter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._steps import Step


def check_scan_steps(steps: Sequence[Step]) -> None:
    """Check that a line-integral scan still has the domains required by each step.

    Parameters
    ----------
    steps : sequence of Step
        Corrections in user-supplied order.

    Raises
    ------
    InvalidParameter
        If a step requires the original counts or transmission.
    """
    wrong = [
        (i, type(step).__name__) for i, step in enumerate(steps) if step.domain != "line integrals"
    ]
    if wrong:
        names = ", ".join(name for _, name in wrong)
        raise InvalidParameter(
            f"corrections[{wrong[0][0]}]",
            f"{names} act on detector counts or transmission, which a Scan of line "
            "integrals no longer holds: load the frames with tomojax.load_frames and pass "
            "them to Frames.corrected",
        )


def validated_views_per_batch(value: int | None) -> int | None:
    """Check the requested correction batch size.

    Parameters
    ----------
    value : int or None
        Positive view count or automatic selection.

    Returns
    -------
    int or None
        Validated batch size.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value <= 0:
        raise InvalidParameter(
            "views_per_batch", f"views_per_batch must be a positive integer, not {value!r}"
        )
    return int(value)


def positive_finite(value: float, name: str) -> float:
    """Check a positive finite correction parameter.

    Parameters
    ----------
    value : float
        Proposed value.
    name : str
        Field to identify in an error.

    Returns
    -------
    float
        Validated value.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise InvalidParameter(name, f"{name} must be finite and positive, not {value!r}")
    return value
