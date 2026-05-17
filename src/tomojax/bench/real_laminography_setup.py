"""Reference-regression setup-stage adapters for real-laminography diagnostics."""

from __future__ import annotations

# check-public-imports: allow-private
from tomojax.align._setup_stage import (
    _optimize_setup_geometry_bilevel_for_level as optimize_reference_setup_geometry_bilevel_for_level,
)

__all__ = ["optimize_reference_setup_geometry_bilevel_for_level"]
