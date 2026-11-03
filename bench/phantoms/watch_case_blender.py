"""Case the movement of ``watch_blender.py`` as a complete wristwatch; export an STL per material.

Run headless (Blender 4.5 LTS):

    blender -b --factory-startup -P bench/phantoms/watch_case_blender.py -- OUT_DIR

The 29 mm movement sits in a 36.4 mm 316L steel case. Seen from the dial,
12 o'clock is +y and 3 o'clock is -x (the dial faces -z). Lengths are
millimetres, and different materials never overlap.

- Case: a middle with four lugs, a bezel holding a domed sapphire crystal on a
  rubber gasket, and a screwed caseback ring with a sapphire display window
  over its own gasket. A fluted crown on a stem and tube at 3 o'clock.
- Dial: a brass disc with a date window at 3 o'clock, eleven applied 18-carat
  gold indices (a double one at 12) and lume plots at every hour.
- Hands at 10:09:30: dauphine hour and minute hands of blued steel with lume
  inlays, a lancet centre seconds hand with a counterweight, on nested pipes.
- Date: a thin steel ring with 31 inner teeth between the dial and the plate.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watch_blender as w

CASE_MATERIALS = (
    "steel316",
    "sapphire",
    "lume",
    "rubber",
    "dial",
)  # the dial is brass, kept apart for rendering
for name in CASE_MATERIALS:
    w.PARTS.setdefault(name, [])

INNER = 14.75  # case bore around the movement's 14.5 mm casing ring
OUTER = 18.2
DIAL = (-0.55, -0.15)
CRYSTAL_R = 14.85
TIME = (10, 9, 30)


def hand_direction(fraction: float) -> np.ndarray:
    """Unit direction of a hand a ``fraction`` of the way round the dial."""
    a = math.tau * fraction
    return np.array([-math.sin(a), math.cos(a)])


LUG_X = 11.45  # lug centres either side of a 20 mm strap
YZ_TO_XYZ = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)  # a (y, z) outline extruded along x


def side_profile(outline: list[tuple[float, float]], side: int) -> w.Solid:
    """A prism whose (y, z) ``outline`` (y mirrored by ``side``) runs right across the case in x."""
    yz = np.array([(side * y, z) for y, z in outline])
    return w.prism(yz, -30.0, 30.0).turned(YZ_TO_XYZ, np.zeros(3))


def lug(side: int, x: float) -> object:
    """A lug at 12 (``side`` 1) or 6 o'clock, sloping to its tip and curving towards the wrist."""
    root, tip = np.array([x, side * 13.0]), np.array([x, side * 21.1])
    flank = w.bar(root, tip, 3.4, 2.5)
    start = math.atan2(*(flank[1] - tip)[::-1])
    outline = np.concatenate([flank[:2], w.arc(tip, 1.25, start, start + math.pi)[1:-1], flank[2:]])
    obj = w.as_object("lug", w.prism(outline, 0.6, 5.05))
    dial_side = side_profile([(15.6, 0.6), (22.6, 2.1), (22.6, -2.0), (15.6, -2.0)], side)
    wrist = [
        (17.2, 5.05),
        (18.8, 4.95),
        (20.4, 4.65),
        (21.6, 4.25),
        (22.6, 3.8),
        (22.6, 8.0),
        (17.2, 8.0),
    ]
    hole = w.lathe([(0, -1.6), (0.5, -1.6), (0.5, 1.6), (0, 1.6)]).turned(
        w.along(np.array([1.0, 0, 0])), np.array([x, side * 20.0, 3.1])
    )
    obj = w.cut(obj, dial_side, side_profile(wrist, side), hole)
    return w.chamfer(obj, 0.18)


def case_middle() -> None:
    """The case middle with its four lugs and their spring-bar holes."""
    profile = [
        (INNER, -0.9),
        (17.55, -0.9),
        (17.95, -0.55),
        (OUTER, 0.4),
        (OUTER, 5.1),
        (17.95, 6.05),
        (17.55, 6.4),
        (INNER, 6.4),
    ]
    middle = w.as_object("case middle", w.lathe(profile))
    w.fuse("case middle", [middle, *(lug(side, x) for side in (1, -1) for x in (LUG_X, -LUG_X))])
    w.part("case middle", "steel316", middle)


def bezel_and_crystal() -> None:
    """The bezel and the domed sapphire crystal on its gasket."""
    bezel = [
        (15.15, -0.9),
        (17.55, -0.9),
        (17.55, -1.35),
        (17.2, -2.0),
        (16.4, -2.45),
        (15.2, -2.62),
        (14.35, -2.62),
        (14.35, -2.42),
        (15.15, -2.42),
    ]
    w.part("bezel", "steel316", w.lathe(bezel))
    gasket = [(CRYSTAL_R, -2.38), (15.15, -2.38), (15.15, -1.45), (CRYSTAL_R, -1.45)]
    w.part("crystal gasket", "rubber", w.lathe(gasket))
    dome = [(r, -2.36 - 0.22 * (1 - (r / CRYSTAL_R) ** 2)) for r in np.linspace(CRYSTAL_R, 0, 24)]
    crystal = [(0.0, -1.45), (CRYSTAL_R, -1.45), *dome]
    w.part("crystal", "sapphire", w.lathe(crystal))


def caseback() -> None:
    """The caseback ring, its gasket and the sapphire display window."""
    w.part("caseback gasket", "rubber", w.tube(15.3, 16.6, 6.4, 6.55))
    ring = [
        (12.6, 6.55),
        (17.55, 6.55),
        (17.55, 7.05),
        (16.9, 7.55),
        (11.9, 7.55),
        (11.9, 7.4),
        (12.6, 7.4),
    ]  # a lip over the window edge
    back = w.as_object("caseback", w.lathe(ring))
    notches = [
        w.prism(
            w.rounded_bar(16.9 * hand_direction(k / 6), 18.2 * hand_direction(k / 6), 0.45),
            7.2,
            7.8,
        )
        for k in range(6)
    ]
    w.part("caseback", "steel316", w.cut(back, *notches))
    w.part("display window", "sapphire", w.cylinder(12.6, 6.75, 7.4))


def crown() -> None:
    """The fluted crown on its stem and tube at 3 o'clock."""
    axis = w.along(np.array([-1.0, 0.0, 0.0]))
    z = 2.8
    tube = w.lathe([(0.75, 0.0), (1.05, 0.0), (1.05, 1.6), (0.75, 1.6)])
    stem = w.lathe([(0, -3.5), (0.45, -3.5), (0.45, 3.0), (0, 3.0)])
    flutes = w.prism(w.ratchet_teeth(24, 2.75, 2.5), 0.0, 2.6)
    cap = w.lathe([(0, 2.6), (2.5, 2.6), (2.2, 3.0), (0, 3.05)])
    base = np.array([-OUTER + 0.1, 0.0, z])
    w.part("crown tube", "steel316", tube.turned(axis, base))
    w.part("stem", "steel316", stem.turned(axis, base))
    w.part(
        "crown",
        "steel316",
        w.fuse(
            "crown",
            [
                flutes.turned(axis, base + np.array([-1.6, 0, 0])),
                cap.turned(axis, base + np.array([-1.6, 0, 0])),
            ],
        ),
    )


def dial() -> None:
    """The dial with its date window, gold indices and lume plots."""
    disc = w.as_object("dial", w.cylinder(14.35, *DIAL))
    window = w.prism(
        np.array([[-13.0, -0.65], [-11.4, -0.65], [-11.4, 0.65], [-13.0, 0.65]]),
        DIAL[0] - 0.1,
        DIAL[1] + 0.1,
    )
    hole = w.cylinder(0.62, DIAL[0] - 0.1, DIAL[1] + 0.1)
    w.part("dial", "dial", w.cut(disc, window, hole))
    for hour in range(12):
        u = hand_direction(hour / 12)
        plot = w.cylinder(0.32, DIAL[0] - 0.1, DIAL[0], 13.75 * u)
        w.part(f"lume plot {hour}", "lume", plot)
        if hour == 3:
            continue  # the date window
        n = np.array([-u[1], u[0]])
        offsets = (-0.42, 0.42) if hour == 0 else (0.0,)
        for o in offsets:
            index = w.prism(
                w.rounded_bar(10.6 * u + o * n, 12.9 * u + o * n, 0.55 if hour % 3 == 0 else 0.4),
                DIAL[0] - 0.22,
                DIAL[0],
            )
            w.part(f"index {hour}", "gold", index)


def date_wheel() -> None:
    """The date ring with 31 inner teeth between the dial and the plate."""
    outer = w.circle(13.4)
    teeth = w.ratchet_teeth(31, 11.6, 11.15)
    ring = w.as_object("date ring", w.prism(outer, -0.13, -0.03))
    w.part("date ring", "steel", w.cut(ring, w.prism(teeth, -0.2, 0.05)))


def dauphine(direction: np.ndarray, length: float, width: float, tail: float) -> np.ndarray:
    """A dauphine hand's kite outline along ``direction``, widest a third of the way out."""
    n = np.array([-direction[1], direction[0]])
    return np.array(
        [
            -tail * direction,
            width * n + 0.32 * length * direction,
            length * direction,
            -width * n + 0.32 * length * direction,
        ]
    )


def hands() -> None:
    """Hour, minute and seconds hands on their nested pipes."""
    hour, minute, second = TIME
    pipes = [  # z range and radii of the nested pipes: hour wheel, cannon pinion, seconds staff
        ("hour pipe", (-0.95, -0.15), 0.55, 0.32),
        ("cannon pipe", (-1.12, -0.15), 0.30, 0.17),
        ("seconds staff", (-1.27, -0.15), 0.15, 0.0),
    ]
    for name, (z0, z1), r1, r0 in pipes:
        solid = w.tube(r0, r1, z0, z1) if r0 else w.cylinder(r1, z0, z1)
        w.part(name, "steel", solid)
    for name, fraction, length, width, z, material in (
        ("hour hand", (hour + minute / 60) / 12, 8.6, 0.85, (-0.95, -0.86), "steel"),
        ("minute hand", (minute + second / 60) / 60, 12.9, 0.7, (-1.12, -1.03), "steel"),
    ):
        u = hand_direction(fraction)
        body = w.as_object(name, w.prism(dauphine(u, length, width, 1.2), *z))
        hub = w.cylinder(0.9 if name == "hour hand" else 0.75, *z)
        body = w.fuse(name, [body, hub])
        n = np.array([-u[1], u[0]])
        kite = np.array(
            [
                0.30 * length * u,
                0.42 * length * u - 0.45 * width * n,
                0.85 * length * u,
                0.42 * length * u + 0.45 * width * n,
            ]
        )
        inlay = w.prism(kite, *z)
        w.part(name, material, w.cut(body, inlay))
        w.part(f"{name} lume", "lume", inlay)
    u = hand_direction(second / 60)
    n = np.array([-u[1], u[0]])
    needle = w.prism(w.rounded_bar(-3.0 * u, 13.6 * u, 0.12), -1.27, -1.21)
    hub = w.cylinder(0.45, -1.27, -1.21)
    weight = w.cylinder(0.6, -1.27, -1.21, -3.0 * u)
    w.part("seconds hand", "steel", w.fuse("seconds hand", [needle, hub, weight]))


def build() -> dict:
    """Model the movement and its case; returns the description."""
    description = w.build()
    case_middle()
    bezel_and_crystal()
    caseback()
    crown()
    dial()
    date_wheel()
    hands()
    description.update(
        radius_mm=23.2,
        height_mm=[-2.62, 7.55],
        case={"diameter_mm": 2 * OUTER, "dial_z_mm": list(DIAL), "time": list(TIME)},
    )
    return description


def main() -> None:
    """Build the watch into the directory after ``--``."""
    out = Path(sys.argv[sys.argv.index("--") + 1])
    import bpy  # pyright: ignore[reportMissingImports]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    w.export(out, build())


if __name__ == "__main__":
    main()
