"""Render the watch meshes (one STL per material) as a photograph-like image in Blender.

    blender -b --factory-startup -P bench/phantoms/watch_render_blender.py -- MESH OUT VIEW [N]

MESH is the directory of STLs, OUT the PNG to write and N the Cycles samples.

VIEW is ``dial`` (three-quarter view of the dial), ``back`` (through the display
caseback), ``section`` (cut through the crown axis) or ``top`` (the dial face on).
Cycles renders on the GPU when one is available.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import bpy  # pyright: ignore[reportMissingImports]
from mathutils import Vector  # pyright: ignore[reportMissingImports]

LOOK = {  # base colour, metallic, roughness, transmission, IOR, emission strength
    "steel316": ((0.80, 0.80, 0.81), 1.0, 0.16, 0.0, 1.5, 0.0),
    "steel": ((0.10, 0.16, 0.55), 1.0, 0.22, 0.0, 1.5, 0.0),
    "brass": ((0.86, 0.66, 0.36), 1.0, 0.28, 0.0, 1.5, 0.0),
    "gold": ((1.00, 0.76, 0.33), 1.0, 0.12, 0.0, 1.5, 0.0),
    "titanium": ((0.58, 0.60, 0.64), 1.0, 0.32, 0.0, 1.5, 0.0),
    "silicon": ((0.22, 0.12, 0.45), 0.6, 0.12, 0.0, 1.5, 0.0),
    "ruby": ((0.85, 0.04, 0.08), 0.0, 0.02, 1.0, 1.77, 0.0),
    "sapphire": (
        (1.00, 1.00, 1.00),
        0.0,
        0.0,
        1.0,
        1.1,
        0.0,
    ),  # thin-glass look: a full 1.77 shows the window edge
    "glucydur": ((0.88, 0.62, 0.48), 1.0, 0.2, 0.0, 1.5, 0.0),
    "nivarox": ((0.40, 0.42, 0.46), 1.0, 0.25, 0.0, 1.5, 0.0),
    "nivaflex": ((0.40, 0.42, 0.46), 1.0, 0.3, 0.0, 1.5, 0.0),
    "lume": ((0.86, 0.95, 0.82), 0.0, 0.5, 0.0, 1.5, 0.35),
    "rubber": ((0.02, 0.02, 0.02), 0.0, 0.6, 0.0, 1.5, 0.0),
    "dial": ((0.80, 0.81, 0.80), 0.35, 0.33, 0.0, 1.5, 0.0),
}
VIEWS = {  # camera position, target, lens (mm)
    "dial": ((-30.0, -55.0, -95.0), (0.0, 0.0, 0.0), 70),
    "back": ((30.0, -55.0, 95.0), (0.0, 0.0, 3.0), 70),
    "section": ((-30.0, -95.0, -30.0), (0.0, 0.0, 2.5), 80),
    "top": ((0.0, 0.0, -120.0), (0.0, 0.0, 0.0), 60),
}


def material(name: str) -> object:
    """A Principled material for the STL called ``name``."""
    colour, metallic, roughness, transmission, ior, glow = LOOK[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = ior
    if name == "sapphire":
        bsdf.inputs["Specular IOR Level"].default_value = 1.0
    if glow:
        bsdf.inputs["Emission Color"].default_value = (0.45, 1.0, 0.6, 1.0)
        bsdf.inputs["Emission Strength"].default_value = glow
    return mat


def aim(eye: Vector, target: Vector, up: Vector) -> object:
    """Camera rotation from ``eye`` towards ``target``, with ``up`` at the top of the frame."""
    from mathutils import Matrix  # pyright: ignore[reportMissingImports]

    back = (eye - target).normalized()
    right = up.cross(back).normalized()
    return Matrix((right, back.cross(right), back)).transposed()


def light(name: str, at: tuple, energy: float, size: float) -> None:
    """An area light at ``at`` aimed at the watch."""
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size = energy, size
    obj = bpy.data.objects.new(name, data)
    obj.location = at
    obj.rotation_euler = (Vector((0, 0, 1)) - Vector(at)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)


def stage(scene: object, view: str, samples: int) -> None:
    """Lights, a dark studio background and Cycles settings, on the GPU when there is one."""
    sign = -1 if view in ("dial", "section", "top") else 1
    light("key", (-60, 40, sign * 70), 2.2e5, 40)
    light("fill", (70, -20, sign * 40) if sign < 0 else (95, 35, 12), 8.0e4, 60)
    light("rim", (10, -80, sign * 20), 6.0e4, 30)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.055, 0.065, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    scene.world = world
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1200
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except TypeError:
            continue
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return


def main() -> None:
    """Render the view and file named after ``--``."""
    args = sys.argv[sys.argv.index("--") + 1 :]
    meshes, out, view = Path(args[0]), Path(args[1]), args[2]
    samples = int(args[3]) if len(args) > 3 else 256
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    section = bpy.data.objects.new("section", bpy.data.meshes.new("section"))
    if view == "section":
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -30, 2.5))
        section = bpy.context.object
        section.scale = (80, 60, 40)
        section.hide_render = True
    for stl in sorted(meshes.glob("*.stl")):
        bpy.ops.wm.stl_import(filepath=str(stl))
        obj = bpy.context.selected_objects[0]
        obj.data.materials.append(material(stl.stem))
        bpy.ops.object.shade_auto_smooth(angle=math.radians(35))
        if view == "section":
            cut = obj.modifiers.new("section", "BOOLEAN")
            cut.operation, cut.solver, cut.object = "DIFFERENCE", "EXACT", section
    camera_at, target, lens = VIEWS[view]
    up = (0.0, 1.0, 0.0) if view != "section" else (0.0, 0.0, -1.0)
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    cam.data.lens = lens
    cam.location = camera_at
    cam.rotation_euler = aim(Vector(camera_at), Vector(target), Vector(up)).to_euler()
    bpy.context.scene.collection.objects.link(cam)
    scene = bpy.context.scene
    scene.camera = cam
    stage(scene, view, samples)
    scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
