r"""Partition closed material meshes at a physical reconstruction box using Blender.

    blender -b --factory-startup -P bench/phantoms/partition_meshes_blender.py -- \\
        MESH_DIR OUTPUT_DIR --centre 0 0 3 --extent 6 6 6

Lengths and STL coordinates are millimetres, vectors are xyz. Each material is
intersected with the box and subtracted from it independently, leaving closed
``interior/NAME.stl`` and ``exterior/NAME.stl`` surfaces. Empty materials are
omitted. This partitions the simulator's continuous object, not solver voxels.
The exterior must be rendered separately: pixel-averaged logs are not additive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    """Export the two closed halves of every STL in the input directory."""
    import bpy  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("meshes", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--centre", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_cube_add(size=1, location=args.centre)
    box = bpy.context.object
    box.dimensions = args.extent
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    records = {}
    for side, operation in (("interior", "INTERSECT"), ("exterior", "DIFFERENCE")):
        out = args.out / side
        out.mkdir(parents=True, exist_ok=False)
        records[side] = {}
        for path in sorted(args.meshes.glob("*.stl")):
            bpy.ops.wm.stl_import(filepath=str(path))
            obj = bpy.context.object
            modifier = obj.modifiers.new("reconstruction box", "BOOLEAN")
            modifier.operation, modifier.solver, modifier.object = operation, "EXACT", box
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            faces = len(obj.data.polygons)
            records[side][path.stem] = faces
            if faces:
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.ops.wm.stl_export(filepath=str(out / path.name), export_selected_objects=True)
            bpy.data.objects.remove(obj, do_unlink=True)
    (args.out / "partition.json").write_text(
        json.dumps({"centre_mm": args.centre, "extent_mm": args.extent, "faces": records}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
