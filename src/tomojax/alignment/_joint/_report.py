"""What a fit did, stated as facts, and the views a user should look at.

There is no reliable cheap signal that an alignment helped, so the report makes
no verdict: it records how each level ended, the size of the corrections split
into smooth and view-to-view parts, the fitted offsets, and two kinds of view
that stand out. A pose flag marks a view whose correction leaves both
neighbours' in the same direction (an isolated jump, not a slip, which is a
step); a photometric flag marks a view the fitted image explains unusually
badly (a broken frame, diffraction), which need not move its pose at all.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ._pose import _spline_design

# An isolated view leaves both neighbours by this many robust standard deviations
# of the scan's view-to-view change.
_OUTLIER_SIGMAS = 5.0
# An unusual view's residual RMS is at least this multiple of its neighbours'.
_UNUSUAL_RMS_FACTOR = 3.0
# An isolated view must also stand out by this much, in working pixels.
_ISOLATED_MIN_WORKING_PX = 0.1
# A rotation axis within this angle of perpendicular to the beam is axial.
_AXIAL_TILT_DEG = 5.0
_NAMES = ("alpha", "beta", "phi", "dx", "dz", "dy")
# Held-out evidence (lab-board/v1-held-out.md, 2026-11-12), stated where it applies.
_AXIAL_ROTATIONS = (
    "per-view rotations were fitted with the rotation axis perpendicular to the beam; on "
    "held-out data such fits were accurate only to 0.14-0.77 px, with one rotation several "
    "times its true size. Prefer fit='translations' unless an artefact remains."
)


def axial(nominal, beam):
    """Whether the scan's rotation axis is perpendicular to the beam, within ``_AXIAL_TILT_DEG``."""
    rotations = np.asarray(nominal, np.float64)[:, :3, :3]
    turns = Rotation.from_matrix(rotations @ rotations[0].T).as_rotvec()
    sizes = np.linalg.norm(turns, axis=1)
    if sizes.max() <= 0:
        return False
    axis = turns[np.argmax(sizes)] / sizes.max()
    beam = np.asarray(beam, np.float64) / np.linalg.norm(beam)
    return bool(abs(axis @ beam) < np.sin(np.deg2rad(_AXIAL_TILT_DEG)))


def _robust_sd(values, axis=0):
    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def isolated_views(steps):
    """Views whose (views, k) value leaves both neighbours' in the same direction.

    The departure from each neighbour must exceed both ``_OUTLIER_SIGMAS``
    robust standard deviations of the view-to-view change and
    ``_ISOLATED_MIN_WORKING_PX``, in any one column.
    """
    steps = np.asarray(steps, np.float64)
    if len(steps) < 3:
        return []
    limit = np.maximum(
        _OUTLIER_SIGMAS * _robust_sd(np.diff(steps, axis=0)), _ISOLATED_MIN_WORKING_PX
    )
    before, after = steps[1:-1] - steps[:-2], steps[1:-1] - steps[2:]
    same = np.sign(before) == np.sign(after)
    apart = np.minimum(np.abs(before), np.abs(after)) > limit
    return (np.flatnonzero(np.any(same & apart, axis=1)) + 1).tolist()


def unusual_views(energy):
    """Views whose residual RMS is ``_UNUSUAL_RMS_FACTOR`` times their neighbours'.

    A view's residual varies smoothly with angle, as the object presents more or
    less structure, so each view is compared with the median energy of the two
    views on either side. On clean data the ratio stays below about two in RMS;
    broken frames and diffraction exceed it several times over.
    """
    energy = np.maximum(np.asarray(energy, np.float64), 1e-300)
    n = len(energy)
    if n < 3:
        return []
    padded = np.pad(energy, 2, constant_values=np.nan)
    around = np.stack([padded[k : k + n] for k in (0, 1, 3, 4)])
    ratio = energy / np.nanmedian(around, axis=0)
    return np.flatnonzero(ratio >= _UNUSUAL_RMS_FACTOR**2).tolist()


def _split(columns, spacing):
    """The part of each column the prior's drift spline explains, and the rest."""
    design = _spline_design(len(columns), spacing)
    smooth = design @ np.linalg.lstsq(design, columns, rcond=None)[0]
    return smooth, columns - smooth


def build(
    *,
    table,
    local,
    fit,
    axial,
    corrections,
    native,
    binning,
    levels,
    refinement,
    offsets,
    energy,
    spacing,
):
    """The report: per-level endings, correction sizes, offsets and flagged views.

    ``table`` is the (views, 6) correction table in physical units, ``local``
    its fitted per-view columns, ``native`` the detector pitch (du, dv) and
    ``binning`` the fine level's detector binning.
    """
    units = np.array([1.0, 1.0, 1.0, native[0], native[1], native[0]])
    scale = np.where(np.arange(6) < 3, 180 / np.pi, 1 / units)
    per_view = {}
    if local:
        values = np.asarray(table, np.float64)[:, list(local)] * scale[list(local)]
        smooth, rough = _split(values, spacing)
        for k, column in enumerate(local):
            per_view[_NAMES[column]] = {
                "unit": "deg" if column < 3 else "native px",
                "rms": float(np.sqrt(np.mean(values[:, k] ** 2))),
                "max": float(np.max(np.abs(values[:, k]))),
                "smooth_rms": float(np.sqrt(np.mean(smooth[:, k] ** 2))),
                "view_to_view_rms": float(np.sqrt(np.mean(rough[:, k] ** 2))),
            }
    detector = [k for k, column in enumerate(local) if column in (3, 4)]
    isolated = isolated_views(values[:, detector] / binning) if local and detector else []
    offsets = np.asarray(offsets, np.float64)
    return {
        "levels": [{"steps": len(lv["steps"]), "ended": lv["status"]} for lv in levels],
        "refinement": refinement.get("status"),
        "corrections": {"global": corrections, "per_view": per_view},
        "offsets": {
            "offset_range": [float(offsets[:, 0].min()), float(offsets[:, 0].max())],
            "ramp_range": [float(offsets[:, 1].min()), float(offsets[:, 1].max())],
        },
        "isolated_views": isolated,
        "unusual_views": unusual_views(energy),
        "advisories": [_AXIAL_ROTATIONS] if axial and "rotations" in fit else [],
    }
