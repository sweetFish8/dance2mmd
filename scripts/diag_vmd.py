import bpy, sys, os, addon_utils
argv = sys.argv[sys.argv.index("--")+1:]
PMX=os.path.abspath(argv[argv.index("--pmx")+1]); VMD=os.path.abspath(argv[argv.index("--vmd")+1])
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("mmd_tools", default_set=True, persistent=True)
bpy.ops.mmd_tools.import_model(filepath=PMX, scale=0.08, clean_model=True)
# show some model bone japanese names
arm=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]
names_j=[]
for pb in arm.pose.bones:
    nj=getattr(pb.mmd_bone,'name_j','') if hasattr(pb,'mmd_bone') else ''
    names_j.append((pb.name, nj))
print("[diag] total bones:", len(names_j))
print("[diag] sample (blender_name | mmd_name_j):")
for bn,nj in names_j[:20]: print("   ", repr(bn), "|", repr(nj))
# which of our driven names exist as name_j?
want=["センター","上半身","上半身2","首","左腕","左ひじ","右腕","右ひじ","左足","左ひざ","右足","右ひざ","左足ＩＫ","右足ＩＫ"]
jset={nj for _,nj in names_j}
print("[diag] our vmd bones present as name_j:")
for w in want: print("   ", w, "->", "YES" if w in jset else "NO")

for mapper in ('PMX','BLENDER'):
    bpy.ops.mmd_tools.import_vmd(filepath=VMD, scale=0.08, bone_mapper=mapper)
    ad=arm.animation_data
    n=len(ad.action.fcurves) if ad and ad.action else 0
    print(f"[diag] mapper={mapper}: fcurves={n}")
    # check 左腕 rotation range
    if ad and ad.action:
        import math
        # find the blender bone whose name_j == 左腕
        bname=None
        for bn,nj in names_j:
            if nj=="左腕": bname=bn; break
        if bname:
            fcs=[fc for fc in ad.action.fcurves if f'pose.bones["{bname}"].rotation_quaternion' in fc.data_path]
            if fcs:
                vals=[kp.co[1] for kp in fcs[0].keyframe_points]
                print(f"   左腕({bname}) quat[0] range: {min(vals):.3f}..{max(vals):.3f}, keys={len(vals)}")
            else: print(f"   左腕({bname}) NO rotation fcurves")
        else: print("   no bone with name_j=左腕")
    if ad and ad.action: bpy.data.actions.remove(ad.action)
