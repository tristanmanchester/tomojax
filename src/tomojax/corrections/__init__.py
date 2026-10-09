"""Corrections: from detector counts to line integrals, and on line integrals.

:meth:`tomojax.Frames.corrected` turns a scan's counts into line integrals:
any steps on counts, then ``(I - D) / (F - D)`` with its dark and flat fields
(each view's flat interpolated between the flat sets around it), any steps on
transmission, ``-log``, then any steps on line integrals. :meth:`tomojax.Scan.corrected`
runs line-integral steps on a scan. What was done is recorded in
:attr:`tomojax.Scan.corrections` and saved with the scan.

A step is a frozen dataclass of its settings with the ``domain`` it acts on;
see :class:`Step`.
"""

from __future__ import annotations

from ._engine import Corrected, correct_frames, correct_projections
from ._records import Correction
from ._steps import BeamHardening, Paganin, RejectViews, Step, Stripes, Zingers

__all__ = [
    "BeamHardening",
    "Corrected",
    "Correction",
    "Paganin",
    "RejectViews",
    "Step",
    "Stripes",
    "Zingers",
    "correct_frames",
    "correct_projections",
]
