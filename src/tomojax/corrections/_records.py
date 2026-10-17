"""The record of a correction applied to a scan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


@dataclass(frozen=True)
class Correction:
    """One correction applied to a scan: what it was, its settings and what it found.

    ``settings`` are the step's own (a flat field's frame and set counts, a
    stripe filter's width); ``found`` holds what the data showed while it ran
    (pixels with no light, non-finite values set to zero). Both are JSON, so
    the record is saved with the scan and read back with it.

    Parameters
    ----------
    name : str
        Correction name, such as ``"flat_dark"`` or ``"stripes"``.
    settings : dict, optional
        JSON-compatible settings used by the correction. Defaults to an empty dict.
    found : dict, optional
        JSON-compatible findings recorded while correcting the data.
        Defaults to an empty dict.
    """

    name: str
    settings: dict[str, Json] = field(default_factory=dict)
    found: dict[str, Json] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"{k}={v}" for k, v in self.settings.items()]
        parts += [f"{k}: {v}" for k, v in self.found.items() if v]
        return f"{self.name}({', '.join(parts)})" if parts else self.name

    def to_dict(self) -> dict[str, Json]:
        """Serialize the correction record to a dictionary.

        Returns
        -------
        dict
            Name, settings and findings, with copies of the two outer dictionaries.
        """
        return {"name": self.name, "settings": dict(self.settings), "found": dict(self.found)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Correction:
        """Restore a correction record from a dictionary.

        Parameters
        ----------
        value : dict
            Record with a required ``name`` and optional ``settings`` and ``found``.

        Returns
        -------
        Correction
            Restored record; missing settings and findings become empty dictionaries.

        Raises
        ------
        KeyError
            If the name is absent.
        """
        return cls(
            str(value["name"]), dict(value.get("settings") or {}), dict(value.get("found") or {})
        )
