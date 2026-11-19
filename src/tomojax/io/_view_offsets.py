"""Recorded per-view detector offsets, applied on top of whichever detector is resolved.

A scanner may record each view's image shift in detector pixels while the
detector centre they are measured from is calibrated separately, or not at all.
The record keeps the shifts as ``detector_view_offsets_pixels`` (and, for cone
data, each view's ``cone_view_distances``) so the same per-view geometry is
built from the file's centre or from a caller's ``detector=``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.geometry import ConeBeam, Detector

if TYPE_CHECKING:
    from ._datasets import ProjectionDataset

OFFSETS_KEY = "detector_view_offsets_pixels"
DISTANCES_KEY = "cone_view_distances"


def with_view_offsets(record: ProjectionDataset, detector: Detector) -> ProjectionDataset:
    """``record`` on ``detector``, each view's centre moved by its recorded offset.

    The offsets are recorded detector pixels (u, v), scaled by ``detector``'s
    pitches and added to its centre. Cone views become one ``cone_segments``
    entry per run of identical arrangements, since a detector-image shift is not
    a rigid object translation in cone geometry; parallel views keep one
    detector and carry the shift as a detector-frame pose. A record without
    recorded offsets is returned on ``detector`` as it is.

    Parameters
    ----------
    record : ProjectionDataset
        Loaded record, possibly carrying per-view detector offsets in its
        geometry metadata.
    detector : Detector
        Detector the views are placed on; its pitches convert the offsets.

    Returns
    -------
    ProjectionDataset
        The record on ``detector`` with each view's offset applied.
    """
    meta = dict(record.geometry_metadata)
    if OFFSETS_KEY not in meta:
        return replace(record, detector=detector)
    offsets = np.asarray(meta[OFFSETS_KEY], np.float64) * (detector.du, detector.dv)
    centers = np.asarray(detector.center, np.float64) + offsets
    for key in ("cone_segments", "trajectory_lengths", "cone_beam"):
        meta.pop(key, None)
    poses = None
    if DISTANCES_KEY in meta:
        distances = np.asarray(meta[DISTANCES_KEY], np.float64)
        entries = _cone_segments(np.column_stack((distances, centers)), detector)
        if len(entries) > 1:
            meta.update(cone_segments=entries, trajectory_lengths=[len(centers)])
        else:
            meta["cone_beam"] = entries[0]["cone_beam"]
            detector = Detector.from_dict(entries[0]["detector"])
    elif np.any(centers != centers[0]):
        poses = np.zeros((len(centers), 6), np.float32)
        poses[:, 3:5] = -offsets
    else:
        detector = replace(detector, center=(float(centers[0, 0]), float(centers[0, 1])))
    return replace(
        record,
        detector=detector,
        geometry_metadata=meta,
        align_params=poses,
        align_gauge=None if poses is None else {"pose_translation_frame": "detector"},
    )


def without_view_offsets(record: ProjectionDataset) -> ProjectionDataset:
    """``record`` with its recorded per-view offsets and centre left out: the nominal geometry.

    The offsets are zeroed rather than removed, so a cone record's single arrangement is
    rebuilt from its recorded distances; the detector keeps its pitch, centred.

    Parameters
    ----------
    record : ProjectionDataset
        Loaded record, possibly carrying per-view detector offsets and a recorded centre.

    Returns
    -------
    ProjectionDataset
        The record on its detector's nominal centre, with no per-view offsets.
    """
    meta = dict(record.geometry_metadata)
    if OFFSETS_KEY in meta:
        meta[OFFSETS_KEY] = np.zeros(np.shape(meta[OFFSETS_KEY])).tolist()
    record = replace(record, geometry_metadata=meta)
    if record.detector is None:
        return record
    d = record.detector
    return with_view_offsets(record, Detector(d.nu, d.nv, d.du, d.dv, (0.0, 0.0)))


def dropped_offsets_message(record: ProjectionDataset) -> str | None:
    """Why an explicit ``geometry=`` loses the record's per-view offsets, or None if it doesn't.

    Parameters
    ----------
    record : ProjectionDataset
        Loaded record, possibly carrying per-view detector offsets.

    Returns
    -------
    str or None
        A message naming how many views move and by how much, or None when no
        view carries a non-zero offset.
    """
    offsets = record.geometry_metadata.get(OFFSETS_KEY)
    if offsets is None:
        return None
    offsets = np.asarray(offsets, np.float64)
    moved = np.any(offsets != 0, axis=1)
    if not moved.any():
        return None
    largest = np.abs(offsets).max(axis=0)
    return (
        f"{int(moved.sum())} of {len(offsets)} views carry recorded detector shifts "
        f"(up to {largest[0]:.3g} / {largest[1]:.3g} pixels in u / v) that geometry= replaces; "
        "pass detector= instead of geometry= to keep them"
    )


def _cone_segments(arrangements: np.ndarray, detector: Detector) -> list[dict[str, Any]]:
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
