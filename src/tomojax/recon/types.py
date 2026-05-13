"""Shared reconstruction type aliases."""

from __future__ import annotations

from typing import Literal

Regulariser = Literal["tv", "huber_tv"]


__all__ = ["Regulariser"]
