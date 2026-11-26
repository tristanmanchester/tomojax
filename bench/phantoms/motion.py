"""Physical scan motion: named setup, stage and sample components in physical units.

The taxonomy follows ANSI/ASME B89.3.4M, Axes of Rotation, as Aerotech's Engineering
Reference presents it. A rotary stage has radial, axial and tilt error motion. Each has a
synchronous part, at integer multiples of the rotation frequency and repeatable with angle,
and a smaller asynchronous part that does not repeat. Centering and squareness are setup,
constant for a scan, not error motion. An angular error at an offset is a displacement, so a
stage tilt reaches the sample mostly as translation, tilt times the sample's height above the
bearing.

The stage moves smoothly. The sample may slip: a rigid step in the object frame, a shift with
or without a tip, which persists. Every component is drawn in micrometers, microradians or arc
seconds from stated ranges and recorded separately. Output lengths are millimeters, matching
TomoJAX grids. Nothing is specified in detector pixels.

A view's world-from-object pose is

    P = A @ N @ Rz(encoder error) @ S

- N is the nominal pose; object z is the rotation axis.
- S is the product of the sample's slips so far, in the object frame, each about the sample
  center.
- A is the lab-frame transform: setup axis tilt and roll, the stage's tilt error about the
  bearing (the sample height below the object origin), then center offset, drift, and
  synchronous and asynchronous radial and axial error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np

_UM = 1e-3  # millimeters per micrometer
_ARCSEC = np.pi / (180 * 3600)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _harmonics(theta: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Sum over k of a_k cos(k theta + phase_k), k = 1..K: a repeatable function of angle."""
    k = np.arange(1, len(amplitudes) + 1)[:, None]
    return np.sum(amplitudes[:, None] * np.cos(k * theta[None] + phases[:, None]), axis=0)


@dataclass(frozen=True)
class Slip:
    """A rigid step of the sample in the object frame at ``view``, persisting afterwards.

    ``shift_um`` is the object-frame displacement. ``tip_rad`` turns the sample about the
    horizontal object-frame axis ``tip_axis`` through the sample center; zero for a slip that
    only shifts.
    """

    view: int
    shift_um: tuple[float, float, float]
    tip_rad: float = 0.0
    tip_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def transform(self, center_mm: np.ndarray) -> np.ndarray:
        """The 4x4 object-frame transform: tip about the center, then shift."""
        out = np.eye(4)
        rotation = _axis_angle(np.asarray(self.tip_axis), self.tip_rad)
        out[:3, :3] = rotation
        out[:3, 3] = center_mm - rotation @ center_mm + np.asarray(self.shift_um) * _UM
        return out


@dataclass(frozen=True)
class Motion:
    """One scan's motion, every component in physical units.

    Harmonic amplitudes and phases have one entry per harmonic of the turn. Once-per-turn radial
    motion is the same as the sample sitting off the axis, so a fit sees it only as the
    object's position (a rigid gauge); higher harmonics are the stage's own error motion.
    Radial error motion is lab x and y (perpendicular to the axis), axial is lab z, and stage
    tilt error is about lab x and y.
    """

    center_offset_um: float = 0.0
    axis_tilt: float = 0.0  # about lab x, TomoJAX's tilt sign: lab Rx(-tilt)
    axis_roll_deg: float = 0.0  # about lab y, the beam
    radial_um: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    radial_phase: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    axial_um: tuple[float, ...] = ()
    axial_phase: tuple[float, ...] = ()
    tilt_urad: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    tilt_phase: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    asynchronous_um: float = 0.0
    asynchronous_urad: float = 0.0
    drift_um_per_hour: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drift_curvature: float = 0.0  # second-order share of the drift over the scan
    encoder_arcsec: tuple[float, ...] = ()
    encoder_phase: tuple[float, ...] = ()
    scan_minutes: float = 30.0
    sample_height_mm: float = 0.0
    slips: tuple[Slip, ...] = ()
    asynchronous_seed: int = 0

    def to_json(self) -> str:
        """Every field, for the truth file beside the composed poses."""
        return json.dumps(asdict(self))


def components(motion: Motion, angles_deg: np.ndarray) -> dict[str, np.ndarray]:
    """Each component per view: lengths in mm, angles in radians."""
    theta = np.deg2rad(np.asarray(angles_deg, np.float64))
    n = len(theta)
    t = np.linspace(0.0, 1.0, n) * motion.scan_minutes / 60.0  # hours

    def sync(amplitudes: tuple[float, ...], phases: tuple[float, ...]) -> np.ndarray:
        return _harmonics(theta, np.asarray(amplitudes, np.float64), np.asarray(phases, np.float64))

    radial = (
        np.stack(
            [
                sync(motion.radial_um[0], motion.radial_phase[0]),
                sync(motion.radial_um[1], motion.radial_phase[1]),
            ],
            -1,
        )
        * _UM
    )
    axial = sync(motion.axial_um, motion.axial_phase) * _UM
    tilt = (
        np.stack(
            [
                sync(motion.tilt_urad[0], motion.tilt_phase[0]),
                sync(motion.tilt_urad[1], motion.tilt_phase[1]),
            ],
            -1,
        )
        * 1e-6
    )
    rng = np.random.default_rng(motion.asynchronous_seed)
    asynchronous = rng.standard_normal((n, 3)) * motion.asynchronous_um * _UM
    asynchronous_tilt = rng.standard_normal((n, 2)) * motion.asynchronous_urad * 1e-6
    span = motion.scan_minutes / 60.0
    shape = t + motion.drift_curvature * (t**2 / max(span, 1e-12) - t)
    drift = shape[:, None] * np.asarray(motion.drift_um_per_hour) * _UM
    encoder = sync(motion.encoder_arcsec, motion.encoder_phase) * _ARCSEC
    return {
        "radial_mm": radial,
        "axial_mm": axial,
        "tilt_rad": tilt,
        "asynchronous_mm": asynchronous,
        "asynchronous_tilt_rad": asynchronous_tilt,
        "drift_mm": drift,
        "encoder_rad": encoder,
    }


def poses(
    motion: Motion, nominal: np.ndarray, angles_deg: np.ndarray, center_mm: np.ndarray
) -> dict[str, np.ndarray]:
    """True world-from-object poses and the stage and sample parts that compose them.

    ``center_mm`` is the sample center in the object frame: the point slips tip about.
    Returns ``poses`` and ``stage`` (the lab-frame transforms A), ``sample`` (the slip
    products S) and ``encoder_rad``, plus every entry of :func:`components`.
    """
    nominal = np.asarray(nominal, np.float64)
    n = len(nominal)
    parts = components(motion, angles_deg)
    center = np.asarray(center_mm, np.float64)
    setup = _ry(np.deg2rad(motion.axis_roll_deg)) @ _rx(-np.deg2rad(motion.axis_tilt))
    bearing = np.array([0.0, 0.0, -motion.sample_height_mm])
    stage = np.tile(np.eye(4), (n, 1, 1))
    sample = np.tile(np.eye(4), (n, 1, 1))
    slips = sorted(motion.slips, key=lambda s: s.view)
    current = np.eye(4)
    for i in range(n):
        tilt = parts["tilt_rad"][i] + parts["asynchronous_tilt_rad"][i]
        error = _ry(tilt[1]) @ _rx(tilt[0])
        rotation = error @ setup
        shift = (
            np.array([motion.center_offset_um * _UM, 0.0, 0.0])
            + parts["drift_mm"][i]
            + parts["asynchronous_mm"][i]
            + np.array([parts["radial_mm"][i, 0], parts["radial_mm"][i, 1], parts["axial_mm"][i]])
            + bearing
            - error @ bearing  # the tilt error turns the sample about the bearing
        )
        stage[i, :3, :3], stage[i, :3, 3] = rotation, shift
        for slip in slips:
            if slip.view == i:
                current = current @ slip.transform(center)
        sample[i] = current
    encoder = np.tile(np.eye(4), (n, 1, 1))
    encoder[:, :3, :3] = np.stack([_rz(a) for a in parts["encoder_rad"]])
    composed = stage @ nominal @ encoder @ sample
    return {"poses": composed, "stage": stage, "sample": sample, **parts}


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def draw(rng: np.random.Generator, ranges: dict, *, views: int, sample_size_mm: float) -> Motion:
    """One motion drawn uniformly from ``ranges``, a dict of [low, high] bounds.

    Keys:
    - ``center_um``, ``axis_tilt``, ``axis_roll_deg``: setup;
    - ``harmonics`` (an int);
    - ``radial_um``, ``axial_um``, ``tilt_urad``: each harmonic's peak amplitude;
    - ``asynchronous_fraction``: asynchronous RMS over the synchronous amplitude;
    - ``drift_um_per_hour``, ``drift_curvature``;
    - ``encoder_arcsec``: once and twice per turn;
    - ``scan_minutes``, ``sample_height_mm``;
    - ``slips`` (count bounds), ``slip_um`` (shift magnitude), ``tip`` ("none" or "geometric").

    Harmonic k of a synchronous error is drawn from the bound divided by k. A geometric tip
    is the slip's size over the sample's size, so a point at the sample's edge
    moves by about the slip. Signs and phases are uniform. Each slip's shift has a random
    direction in the object frame, and its tip axis a random horizontal direction.
    """
    harmonics = int(ranges.get("harmonics", 3))

    def amplitudes(key: str) -> tuple[float, ...]:
        return (
            tuple(_uniform(rng, ranges[key]) / k for k in range(1, harmonics + 1))
            if key in ranges
            else ()
        )

    def phases(count: int) -> tuple[float, ...]:
        return tuple(float(p) for p in rng.uniform(0, 2 * np.pi, count))

    def signed(key: str) -> float:
        return _uniform(rng, ranges[key]) * rng.choice([-1.0, 1.0]) if key in ranges else 0.0

    radial = (amplitudes("radial_um"), amplitudes("radial_um"))
    tilt = (amplitudes("tilt_urad"), amplitudes("tilt_urad"))
    axial = amplitudes("axial_um")
    fraction = (
        _uniform(rng, ranges["asynchronous_fraction"]) if "asynchronous_fraction" in ranges else 0.0
    )
    sync_um = max([*radial[0], *radial[1], *axial, 0.0])
    sync_urad = max([*tilt[0], *tilt[1], 0.0])
    encoder = (
        tuple(_uniform(rng, ranges["encoder_arcsec"]) / k for k in (1, 2))
        if "encoder_arcsec" in ranges
        else ()
    )
    drift_rate = (
        _uniform(rng, ranges["drift_um_per_hour"]) if "drift_um_per_hour" in ranges else 0.0
    )
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    count = (
        int(rng.integers(ranges["slips"][0], ranges["slips"][1] + 1)) if "slips" in ranges else 0
    )
    slips = []
    for view in sorted(rng.choice(np.arange(1, views), size=count, replace=False)) if count else ():
        size_um = _uniform(rng, ranges["slip_um"])
        vector = rng.standard_normal(3)
        vector *= size_um / np.linalg.norm(vector)
        tip = size_um * _UM / sample_size_mm if ranges.get("tip", "none") == "geometric" else 0.0
        axis_angle = rng.uniform(0, 2 * np.pi)
        slips.append(
            Slip(
                int(view),
                (float(vector[0]), float(vector[1]), float(vector[2])),
                float(tip * rng.choice([-1.0, 1.0])),
                (float(np.cos(axis_angle)), float(np.sin(axis_angle)), 0.0),
            )
        )
    return Motion(
        center_offset_um=signed("center_um"),
        axis_tilt=signed("axis_tilt"),
        axis_roll_deg=signed("axis_roll_deg"),
        radial_um=radial,
        radial_phase=(phases(len(radial[0])), phases(len(radial[1]))),
        axial_um=axial,
        axial_phase=phases(len(axial)),
        tilt_urad=tilt,
        tilt_phase=(phases(len(tilt[0])), phases(len(tilt[1]))),
        asynchronous_um=fraction * sync_um,
        asynchronous_urad=fraction * sync_urad,
        drift_um_per_hour=(
            float(drift_rate * direction[0]),
            float(drift_rate * direction[1]),
            float(drift_rate * direction[2]),
        ),
        drift_curvature=_uniform(rng, ranges["drift_curvature"])
        if "drift_curvature" in ranges
        else 0.0,
        encoder_arcsec=encoder,
        encoder_phase=phases(len(encoder)),
        scan_minutes=_uniform(rng, ranges["scan_minutes"]) if "scan_minutes" in ranges else 30.0,
        sample_height_mm=_uniform(rng, ranges["sample_height_mm"])
        if "sample_height_mm" in ranges
        else 0.0,
        slips=tuple(slips),
        asynchronous_seed=int(rng.integers(2**31)),
    )
