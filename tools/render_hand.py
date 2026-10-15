"""Blender script: load a House of Dextra URDF, animate finger curls, render a turntable loop.

Usage: blender -b -P render_hand.py -- <urdf> <out_dir> [--still] [--frames N] [--res R] [--samples S]
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
URDF, OUT = argv[0], argv[1]
STILL = "--still" in argv
def opt(name, default):
    return type(default)(argv[argv.index(name) + 1]) if name in argv else default
FRAMES = opt("--frames", 150)
RES = opt("--res", 720)
SAMPLES = opt("--samples", 48)
ROOT_RPY = (opt("--rx", 0.0), opt("--ry", 0.0), opt("--rz", 0.0))
BG = (0x14 / 255, 0x14 / 255, 0x16 / 255)
BASE = os.path.dirname(URDF)


def origin_mat(el):
    o = el.find("origin") if el is not None else None
    if o is None:
        return Matrix.Identity(4)
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return Matrix.Translation(xyz) @ Euler(rpy, "XYZ").to_matrix().to_4x4()


root = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = root.findall("joint")
children = {}
child_set = set()
for j in joints:
    children.setdefault(j.find("parent").get("link"), []).append(j)
    child_set.add(j.find("child").get("link"))
root_link = next(n for n in links if n not in child_set)

# ── Scene ────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def material(name, rgb, rough, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


MAT_PRINT = material("print", (0.80, 0.82, 0.86), 0.45)        # 3D-printed shells
MAT_SERVO = material("servo", (0.035, 0.035, 0.04), 0.35)       # servos
MAT_ACCENT = material("accent", (0.06, 0.13, 0.55), 0.35)        # grammar links / tips


def mat_for(fname):
    f = os.path.basename(fname)
    if "grammar" in f or "tip" in f:
        return MAT_ACCENT
    if "lever" in f or "palm_vis" in f:
        return MAT_PRINT
    return MAT_SERVO


def load_obj(path):
    """Minimal OBJ reader that keeps raw URDF coordinates (no axis conversion)."""
    vs, fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                fs.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    me = bpy.data.meshes.new(os.path.basename(path))
    me.from_pydata(vs, [], fs)
    me.validate()
    return me


mesh_cache = {}
visuals = []  # (link_name, local_mat, obj)
for lname, l in links.items():
    for v in l.findall("visual"):
        m = v.find("geometry/mesh")
        if m is None:
            continue
        path = os.path.join(BASE, m.get("filename"))
        if path not in mesh_cache:
            mesh_cache[path] = load_obj(path)
        obj = bpy.data.objects.new(f"{lname}", mesh_cache[path])
        obj.data.materials.clear()
        obj.data.materials.append(mat_for(path))
        obj.rotation_mode = "QUATERNION"
        scene.collection.objects.link(obj)
        sc = [float(s) for s in m.get("scale", "1 1 1").split()]
        visuals.append((lname, origin_mat(v) @ Matrix.Diagonal((*sc, 1)), obj))

for me in mesh_cache.values():
    for p in me.polygons:
        p.use_smooth = False


def finger_angles(t):
    """Joint angles at loop phase t in [0,1): a wave of curls across the fingers."""
    q = {}
    for j in joints:
        if j.get("type") != "revolute":
            continue
        n = j.get("name")
        lim = j.find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        fidx = int(n[1]) if n[0] == "f" and n[1].isdigit() else 1
        phase = 2 * math.pi * t - 0.6 * fidx
        s = 0.5 - 0.5 * math.cos(phase)  # 0..1
        if "palm_joint" in n:  # abduction: gentle sway
            q[n] = 0.25 * (lo + hi) / 2 + 0.25 * hi * math.sin(phase)
        else:
            q[n] = lo + (0.12 + 0.6 * s) * (hi - lo)
    return q


def fk(q, world):
    out = {root_link: world}
    stack = [root_link]
    while stack:
        p = stack.pop()
        for j in children.get(p, []):
            c = j.find("child").get("link")
            T = out[p] @ origin_mat(j)
            if j.get("type") in ("revolute", "continuous"):
                ax = Vector([float(v) for v in j.find("axis").get("xyz").split()]).normalized()
                T = T @ Quaternion(ax, q.get(j.get("name"), 0.0)).to_matrix().to_4x4()
            out[c] = T
            stack.append(c)
    return out


root_rot = Euler(ROOT_RPY, "XYZ").to_matrix().to_4x4()


def pose(frame):
    t = (frame - 1) / FRAMES
    spin = Matrix.Rotation(-0.55 + 1.1 * (0.5 - 0.5 * math.cos(2 * math.pi * t)), 4, "Z")
    T = fk(finger_angles(t), spin @ root_rot)
    for lname, local, obj in visuals:
        obj.matrix_world = T[lname] @ local


# Centre the hand at the origin using the rest pose bounding box.
T0 = fk(finger_angles(0.0), root_rot)
pts = [T0[ln] @ loc @ Vector(c) for ln, loc, o in visuals for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
center, size = (lo + hi) / 2, max(hi - lo)
root_rot = Matrix.Translation(-center) @ root_rot
print("hand size", size, "center", tuple(center))

frames = [1] if STILL else range(1, FRAMES + 1)
for f in frames:
    pose(f)
    for _, _, o in visuals:
        o.keyframe_insert("location", frame=f)
        o.keyframe_insert("rotation_quaternion", frame=f)
        o.keyframe_insert("scale", frame=f)

# Camera, lights, floor-less studio
cam_d = opt("--camd", 2.6) * size
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 50
scene.collection.objects.link(cam)
el, az = math.radians(opt("--el", 22.0)), math.radians(opt("--az", -90.0))
cam.location = (cam_d * math.cos(el) * math.cos(az), cam_d * math.cos(el) * math.sin(az), cam_d * math.sin(el))
cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam


def light(name, kind, energy, loc, size_=None, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, kind)
    d.energy, d.color = energy, color
    if size_:
        d.size = size_
    o = bpy.data.objects.new(name, d)
    o.location = loc
    o.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(o)


s = size
light("key", "AREA", opt("--light", 1.0) * 30, (2 * s, -2 * s, 2.5 * s), 1.2 * s)
light("fill", "AREA", opt("--light", 1.0) * 10, (-2.5 * s, -1.5 * s, 0.8 * s), 1.5 * s, (0.85, 0.9, 1.0))
light("rim", "AREA", opt("--light", 1.0) * 35, (-0.5 * s, 2.5 * s, 1.8 * s), 0.8 * s, (0.65, 0.75, 1.0))

world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (*BG, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.4
scene.world = world

scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for dev_type in (() if "--cpu" in argv else ("CUDA",)):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == dev_type
            scene.cycles.device = "GPU"
            print("using", dev_type)
            break
    except TypeError:
        pass
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.render.resolution_x = scene.render.resolution_y = RES
scene.frame_start, scene.frame_end = 1, (1 if STILL else FRAMES)
scene.render.fps = 30
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
os.makedirs(OUT, exist_ok=True)
scene.render.filepath = os.path.join(OUT, "f_")
bpy.ops.render.render(animation=True)
