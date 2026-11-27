"""Detector offsets in a raw acquisition's original pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.geometry import ConeBeam, Detector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._datasets import ProjectionDataset


@dataclass(frozen=True)
class ViewOffsets:
    """Raw image shifts from the original detector sampling.

    Parameters
    ----------
    pixels : numpy.ndarray
        Per-view shifts along detector (u, v), in pixels.
    distances : numpy.ndarray or None
        Per-view source-to-axis and source-to-detector distances for cone scans.
    center_shift : float
        Recorded horizontal center correction in detector pixels.
    """

    pixels: NDArray[np.float64]
    distances: NDArray[np.float64] | None
    center_shift: float

    def resolve(
        self,
        record: ProjectionDataset,
        *,
        detector: Detector | None = None,
        applied: bool = True,
    ) -> ProjectionDataset:
        """Place a raw acquisition on its resolved detector.

        Parameters
        ----------
        record : ProjectionDataset
            Nominal raw acquisition geometry.
        detector : Detector or None, optional
            Explicit detector, whose center replaces the scanner's center shift.
        applied : bool, optional
            Include the scanner's center and per-view shifts.

        Returns
        -------
        ProjectionDataset
            Effective geometry with the original acquisition values retained as metadata.
        """
        from ._datasets import replace_dataset

        meta = dict(record.geometry_metadata)
        meta["vendor_corrections"] = {**meta["vendor_corrections"], "applied": applied}
        if detector is None:
            detector = record.detector
            if detector is not None and applied:
                detector = replace(detector, center=(-detector.du * self.center_shift, 0.0))
            elif meta.get("detector_center_status") == "recorded":
                meta["detector_center_status"] = "nominal"
        if detector is None:
            return replace_dataset(record, geometry_metadata=meta)
        offsets = (
            self.pixels * (detector.du, detector.dv) if applied else np.zeros_like(self.pixels)
        )
        centers: NDArray[np.float64] = np.asarray(detector.center) + offsets
        poses = None
        if self.distances is None and np.any(centers != centers[0]):
            poses = np.zeros((len(centers), 6), np.float32)
            poses[:, 3:5] = -offsets
        else:
            detector = replace(detector, center=(float(centers[0, 0]), float(centers[0, 1])))
        if self.distances is not None:
            entries = _cone_segments(np.column_stack((self.distances, centers)), detector)
            if len(entries) > 1:
                meta.update(cone_segments=entries, trajectory_lengths=[len(centers)])
            else:
                meta["cone_beam"] = entries[0]["cone_beam"]
        return replace_dataset(
            record,
            detector=detector,
            geometry_metadata=meta,
            align_params=poses,
            align_gauge=None if poses is None else {"pose_translation_frame": "detector"},
        )

    def dropped_message(self) -> str | None:
        """Describe the per-view shifts a complete replacement geometry discards.

        Returns
        -------
        str or None
            Number and magnitude of the displaced views, or None if all shifts are zero.
        """
        moved = np.any(self.pixels != 0, axis=1)
        if not moved.any():
            return None
        largest = np.abs(self.pixels).max(axis=0)
        return (
            f"{int(moved.sum())} of {len(moved)} views carry recorded detector shifts "
            f"(up to {largest[0]:.3g} / {largest[1]:.3g} pixels in u / v) that geometry= replaces; "
            "pass detector= instead of geometry= to keep them"
        )


def _cone_segments(arrangements: NDArray[np.float64], detector: Detector) -> list[dict[str, Any]]:
    """One entry per run of identical (source, detector distance, u, v) arrangements."""
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(arrangements, axis=0), axis=1)) + 1]
    stops = np.r_[boundaries[1:], len(arrangements)]
    return [
        {
            "views": int(stop - first),
            "cone_beam": ConeBeam(float(sod), float(sdd)).to_dict(),
            "detector": replace(detector, center=(float(u), float(v))).to_dict(),
        }
        for first, stop in zip(boundaries, stops, strict=True)
        for sod, sdd, u, v in [arrangements[first]]
    ]
