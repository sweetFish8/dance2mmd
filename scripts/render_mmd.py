"""Blender headless render: PMX model + our VMD -> video (Eevee).

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/render_mmd.py -- \
      --pmx models/sparkle_real/sparkle.pmx --vmd data/illit.vmd \
      --out /tmp/illit_render.mp4 --end 150 --res 540x960

mmd_tools reproduces MMD materials/sphere/toon/morphs, so the face + fox mask
render faithfully (unlike the three.js preview). Audio is muxed separately by
the caller with ffmpeg.
"""
import bpy, sys, os, math, addon_utils

# ---- args after '--' ----
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

PMX = os.path.abspath(arg("--pmx"))
VMD = os.path.abspath(arg("--vmd"))
OUT = os.path.abspath(arg("--out", "/tmp/illit_render.mp4"))
END = int(arg("--end", "150"))
RES = arg("--res", "540x960")
RW, RH = (int(x) for x in RES.split("x"))

print(f"[render] pmx={PMX}\n[render] vmd={VMD}\n[render] out={OUT} end={END} res={RW}x{RH}")

# ---- clean scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("mmd_tools", default_set=True, persistent=True)

# ---- import model + motion ----
# Physics ON by default: 火花 has *detached sleeves* + long hair that should hang
# and sway naturally. (--nophysics for rigid.)
USE_PHYSICS = "--nophysics" not in argv
types = {'MESH', 'ARMATURE', 'MORPHS', 'DISPLAY'} | ({'PHYSICS'} if USE_PHYSICS else set())
bpy.ops.mmd_tools.import_model(filepath=PMX, scale=0.08, clean_model=True,
                               fix_IK_links=True, types=types)

# The import_vmd OPERATOR applies to context.selected_objects, which is empty in
# --background mode (select_set doesn't populate it without a window). Call the
# internal VMDImporter directly on the model's armature/meshes/root instead.
from mmd_tools.core.vmd import importer as vmd_importer
from mmd_tools.core import model as mmd_model
from mmd_tools.utils import makePmxBoneMap

root = next(o for o in bpy.data.objects if getattr(o, "mmd_type", "NONE") == "ROOT")
rig = mmd_model.Model(root)
targets = {root, rig.armature()}
try:
    ph = rig.morph_slider.placeholder()
    if ph: targets.add(ph)
except Exception:
    pass
targets |= set(rig.meshes())

imp = vmd_importer.VMDImporter(filepath=VMD, scale=0.08, bone_mapper=makePmxBoneMap, frame_margin=5)
for o in targets:
    if o is not None:
        imp.assign(o)

_arm = rig.armature()
_n = len(_arm.animation_data.action.fcurves) if _arm.animation_data and _arm.animation_data.action else 0
print(f"[render] applied animation fcurves={_n}")

# ---- scene timing ----
sc = bpy.context.scene
sc.render.fps = 30
sc.frame_start = 0
sc.frame_end = END

# stable cloth/hair physics: more substeps + solver iters, cache the full range.
if USE_PHYSICS:
    rbw = sc.rigidbody_world
    if rbw is None:
        bpy.ops.rigidbody.world_add(); rbw = sc.rigidbody_world
    rbw.substeps_per_frame = 6
    rbw.solver_iterations = 10
    rbw.point_cache.frame_start = 0
    rbw.point_cache.frame_end = END
    # render steps frames sequentially from 0, so the sim runs/caches in order.
    # frame_margin in the VMD import leaves a few bind-pose frames for it to settle.
    sc.frame_set(0)

# ---- frame the model with a camera (Blender is Z-up; MMD model faces -Y) ----
import mathutils
mins = mathutils.Vector(( 1e9,  1e9,  1e9))
maxs = mathutils.Vector((-1e9, -1e9, -1e9))
for ob in bpy.data.objects:
    if ob.type == 'MESH':
        for c in ob.bound_box:
            wc = ob.matrix_world @ mathutils.Vector(c)
            mins = mathutils.Vector((min(mins[i], wc[i]) for i in range(3)))
            maxs = mathutils.Vector((max(maxs[i], wc[i]) for i in range(3)))
center = (mins + maxs) / 2
size = maxs - mins
height = max(size.z, 0.1)
print(f"[render] model bbox size={tuple(round(v,2) for v in size)} center={tuple(round(v,2) for v in center)}")

cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
sc.collection.objects.link(cam); sc.camera = cam
dist = height * 1.7
cam.location = (center.x, center.y - dist, center.z + height * 0.05)
# look at center
direction = center - mathutils.Vector(cam.location)
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
cam_data.lens = 50

# ---- stage world: vertical gradient backdrop (deep navy -> magenta) ----
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
nt = world.node_tree; nt.nodes.clear()
n_out = nt.nodes.new("ShaderNodeOutputWorld")
n_bg = nt.nodes.new("ShaderNodeBackground")
n_ramp = nt.nodes.new("ShaderNodeValToRGB")
n_sep = nt.nodes.new("ShaderNodeSeparateXYZ")
n_tc = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(n_tc.outputs["Window"], n_sep.inputs["Vector"])
nt.links.new(n_sep.outputs["Y"], n_ramp.inputs["Fac"])
nt.links.new(n_ramp.outputs["Color"], n_bg.inputs["Color"])
nt.links.new(n_bg.outputs["Background"], n_out.inputs["Surface"])
n_ramp.color_ramp.elements[0].position = 0.0
n_ramp.color_ramp.elements[0].color = (0.015, 0.012, 0.03, 1)   # bottom: dark navy
n_ramp.color_ramp.elements[1].position = 1.0
n_ramp.color_ramp.elements[1].color = (0.10, 0.02, 0.09, 1)     # top: deep magenta
n_bg.inputs[1].default_value = 1.0

# ---- floor plane (subtle gloss to catch rim light) ----
bpy.ops.mesh.primitive_plane_add(size=30, location=(center.x, center.y, mins.z))
floor = bpy.context.active_object
fmat = bpy.data.materials.new("floor"); fmat.use_nodes = True
fb = fmat.node_tree.nodes["Principled BSDF"]
fb.inputs["Base Color"].default_value = (0.02, 0.02, 0.035, 1)
fb.inputs["Roughness"].default_value = 0.45
floor.data.materials.append(fmat)

# ---- 3-point + accent lighting ----
def add_area(name, loc, energy, color, size):
    d = bpy.data.lights.new(name, 'AREA'); d.energy = energy; d.color = color; d.size = size
    o = bpy.data.objects.new(name, d); sc.collection.objects.link(o); o.location = loc
    dir = center - mathutils.Vector(loc)
    o.rotation_euler = dir.to_track_quat('-Z', 'Y').to_euler()
    return o
cz = center.z
add_area("Key",  (center.x - 2.0, center.y - 2.4, cz + 1.6), 420, (1.0, 0.97, 0.92), 2.5)   # warm key, front-left
add_area("Fill", (center.x + 2.2, center.y - 1.8, cz + 0.6), 120, (0.80, 0.85, 1.0), 3.0)   # cool fill, front-right
add_area("Rim",  (center.x + 0.4, center.y + 2.6, cz + 2.2), 600, (1.0, 0.55, 0.75), 2.0)   # pink rim/back, edges hair+sleeves
add_area("Rim2", (center.x - 1.2, center.y + 2.0, cz + 1.2), 350, (0.6, 0.7, 1.0), 1.5)     # cool back-left kicker

# ---- render settings (Eevee) + glow/AO/reflections ----
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 32
sc.eevee.use_bloom = True
sc.eevee.bloom_threshold = 1.0
sc.eevee.bloom_intensity = 0.025
sc.eevee.bloom_radius = 5.0
sc.eevee.use_gtao = True
sc.eevee.use_soft_shadows = True
sc.eevee.use_ssr = True
try:
    sc.view_settings.look = 'Medium High Contrast'
except Exception:
    pass
sc.render.resolution_x = RW
sc.render.resolution_y = RH
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'FFMPEG'
sc.render.ffmpeg.format = 'MPEG4'
sc.render.ffmpeg.codec = 'H264'
sc.render.ffmpeg.constant_rate_factor = 'MEDIUM'
sc.render.filepath = OUT

print("[render] rendering...")
bpy.ops.render.render(animation=True)
print(f"[render] DONE -> {OUT}")
