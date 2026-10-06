"""Build a multi-material chip-package phantom in Blender; export one STL per material.

Run headless (Blender 4.5 LTS):

    blender -b --factory-startup -P bench/phantoms/chip_package_blender.py -- OUT_DIR

Lengths are millimetres (one Blender unit = 1 mm). The package is flat in the
x-y plane and thin along z, the laminography rotation axis:

- substrate: glass-epoxy laminate, cut around the ground plane and vias
- ground_plane: copper sheet inside the substrate, cleared around the vias
- vias: copper barrels through the substrate
- traces: copper lines on the top surface
- die: silicon chip on the substrate
- wires: gold bond wires arching from die pads to the traces
- mold: silica-filled epoxy box enclosing die, wires and traces
- solder: tin-silver-copper balls under the substrate, three with gas voids

Every pair of meshes is disjoint except that die, wires and traces lie inside
the mold (``NESTED``): a ray's mold path length includes them, so a nested
part's material replaces the mold's along its own path length. Wires rest on
the die and traces, touching them only tangentially, so no boolean cut is
needed where parts meet; cuts there leave open seams.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

import bpy  # pyright: ignore[reportMissingImports]
from mathutils import Vector  # pyright: ignore[reportMissingImports]

SUBSTRATE = (1.8, 1.8, 0.24)  # x, y, z extent; z = 0 is its top surface
PLANE_THICKNESS = 0.018
TRACE_THICKNESS = 0.02
DIE = (0.72, 0.64, 0.12)
DIE_GAP = 0.01  # die-attach layer, left as mold material
MOLD_HEIGHT = 0.30
BALL_DIAMETER = 0.14
WIRE_DIAMETER = 0.025
VOIDED_BALLS = {(0, 1), (2, 2), (3, 0)}
NESTED = {"die": "mold", "wires": "mold", "traces": "mold"}

Obj = Any  # bpy.types.Object; Blender ships no type information (and Python 3.11)


def box(name: str, size: tuple[float, float, float], center: tuple[float, float, float]) -> Obj:
    """Add an axis-aligned box."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return obj


def cylinder(name: str, radius: float, depth: float, center: tuple[float, float, float]) -> Obj:
    """Add a z-aligned cylinder."""
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=center, vertices=32)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def sphere(name: str, radius: float, center: tuple[float, float, float]) -> Obj:
    """Add a UV sphere."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center, segments=32, ring_count=16)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def boolean(target: Obj, cutter: Obj, operation: str = "DIFFERENCE") -> None:
    """Apply an exact boolean modifier to ``target``."""
    modifier = target.modifiers.new(name=f"{operation}_{cutter.name}", type="BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = cutter
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def union(name: str, parts: list[Obj]) -> Obj:
    """Merge parts of one material so that overlaps count once."""
    base = parts[0]
    for part in parts[1:]:
        boolean(base, part, "UNION")
        bpy.data.objects.remove(part)
    base.name = name
    return base


def remove(obj: Obj) -> None:
    """Delete a helper object."""
    bpy.data.objects.remove(obj)


def wire(name: str, start: Vector, end: Vector, height: float) -> Obj:
    """Sweep a round tube along a Bezier arch from ``start`` to ``end``."""
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = WIRE_DIAMETER / 2
    curve.bevel_resolution = 4
    curve.resolution_u = 12
    curve.use_fill_caps = True
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(2)
    middle = (start + end) / 2 + Vector((0, 0, height))
    for point, location in zip(spline.bezier_points, (start, middle, end), strict=True):
        point.co = location
        point.handle_left_type = point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.active_object


def via_positions() -> list[tuple[float, float]]:
    """Vias on a 4x4 lattice, leaving the area under the die clear."""
    grid = [(x, y) for x in (-0.66, -0.33, 0.33, 0.66) for y in (-0.66, -0.22, 0.22, 0.66)]
    return [(x, y) for x, y in grid if not (abs(x) < 0.4 and abs(y) < 0.4)]


def pad_rows(vias: list[tuple[float, float]]) -> dict[tuple[float, float], float]:
    """Give each via its own pad row along the die edge it faces."""
    rows: dict[tuple[float, float], float] = {}
    for side in (-1.0, 1.0):
        facing = sorted(
            (v for v in vias if math.copysign(1.0, v[0]) == side), key=lambda v: (v[1], abs(v[0]))
        )
        spacing = 0.54 / max(len(facing) - 1, 1)
        rows.update({v: -0.27 + spacing * k for k, v in enumerate(facing)})
    return rows


def substrate_layers(vias: list[tuple[float, float]]) -> tuple[Obj, Obj, Obj]:
    """Laminate, internal copper plane with clearance holes, and via barrels."""
    sx, sy, sz = SUBSTRATE
    substrate = box("substrate", SUBSTRATE, (0, 0, -sz / 2))
    plane = box("ground_plane", (sx - 0.12, sy - 0.12, PLANE_THICKNESS), (0, 0, -sz / 2))
    barrels = []
    for i, (x, y) in enumerate(vias):
        barrels.append(cylinder(f"via{i}", 0.04, sz, (x, y, -sz / 2)))
        clearance = cylinder(f"clear{i}", 0.07, PLANE_THICKNESS * 3, (x, y, -sz / 2))
        boolean(plane, clearance)
        remove(clearance)
    via = union("vias", barrels)
    boolean(substrate, via)
    boolean(substrate, plane)
    return substrate, plane, via


def top_side(vias: list[tuple[float, float]]) -> tuple[Obj, Obj, Obj, Obj]:
    """Traces, die, bond wires and the mold box that encloses them."""
    rows = pad_rows(vias)
    traces = []
    for i, (x, y) in enumerate(vias):
        start = Vector((math.copysign(DIE[0] / 2 + 0.06, x), rows[(x, y)], 0))
        end = Vector((x, y, 0))
        direction = end - start
        middle = (start + end) / 2
        trace = box(f"trace{i}", (direction.length + 0.08, 0.05, TRACE_THICKNESS), (0, 0, 0))
        trace.rotation_euler[2] = math.atan2(direction.y, direction.x)
        trace.location = (middle.x, middle.y, TRACE_THICKNESS / 2)
        bpy.ops.object.transform_apply(location=True, rotation=True)
        traces.append(trace)
    trace = union("traces", traces)
    die = box("die", DIE, (0, 0, DIE_GAP + DIE[2] / 2))
    radius = WIRE_DIAMETER / 2
    wires = []
    for i, (x, y) in enumerate(vias):
        # Wire ends rest tangentially on the die top and the trace top.
        row = rows[(x, y)]
        pad = Vector((math.copysign(DIE[0] / 2 - 0.05, x), row, DIE_GAP + DIE[2] + radius))
        land = Vector((math.copysign(DIE[0] / 2 + 0.12, x), row, TRACE_THICKNESS + radius))
        wires.append(wire(f"wire{i}", pad, land, 0.09))
    gold = union("wires", wires)
    sx, sy, _ = SUBSTRATE
    mold = box("mold", (sx - 0.2, sy - 0.2, MOLD_HEIGHT), (0, 0, MOLD_HEIGHT / 2))
    return trace, die, gold, mold


def solder_balls() -> Obj:
    """A 4x4 ball grid under the substrate, flattened where it wets the pads."""
    z_bottom = -SUBSTRATE[2]
    balls, voids = [], []
    for i, x in enumerate((-0.6, -0.2, 0.2, 0.6)):
        for j, y in enumerate((-0.6, -0.2, 0.2, 0.6)):
            centre = (x, y, z_bottom - BALL_DIAMETER / 2 + 0.02)
            ball = sphere(f"ball{i}{j}", BALL_DIAMETER / 2, centre)
            cut = box(f"cut{i}{j}", (0.3, 0.3, 0.1), (x, y, z_bottom + 0.05))
            boolean(ball, cut)
            remove(cut)
            balls.append(ball)
            if (i, j) in VOIDED_BALLS:
                voids.append((ball, sphere(f"void{i}{j}", 0.025, (x + 0.02, y, centre[2]))))
    for ball, void in voids:
        boolean(ball, void)
        remove(void)
    return union("solder", balls)


def export(obj: Obj, path: Path) -> None:
    """Write one object as binary STL."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True, ascii_format=False)


def build(out: Path) -> None:
    """Model the package and export each material."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vias = via_positions()
    parts = [*substrate_layers(vias), *top_side(vias), solder_balls()]
    out.mkdir(parents=True, exist_ok=True)
    for obj in parts:
        export(obj, out / f"{obj.name}.stl")
        print(f"exported {obj.name}: {len(obj.data.polygons)} faces")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    build(Path(argv[0] if argv else "chip_package"))
