"""Alignment: estimate a scan's geometry corrections.

Most code calls :func:`tomojax.align` on a :class:`tomojax.Scan`. This package
holds what that uses: :class:`AlignConfig` (expert solver settings),
:func:`alignment_plan` (the configuration and levels a mode runs) and
:func:`align_multires`, which aligns arrays against a geometry. Schedules,
losses, gauges and the single-resolution solver are in
``tomojax.alignment.api``.
"""

from __future__ import annotations

from ._modes import MODES, AlignmentPlan, alignment_plan
from .pipeline import AlignConfig, align_multires

__all__ = [
    "MODES",
    "AlignConfig",
    "AlignmentPlan",
    "align_multires",
    "alignment_plan",
]
