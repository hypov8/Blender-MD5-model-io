

'''
md5 models

update by hypov8
================
2024-08-26
-cleaned up code
-output optimized md5mesh weight. no duplicates
-print shorter/optimized number values to mesh/anim file
-reduced precision on some values, like uv map(.000000) and vertex weights(.00). using numpy
-moved global variables to class
-added file close and try block
-removed dupe variables for old/new blender

2024-08-29
-fixed dialog box not saving previous settings
-changed weight string to 4 decimal places
-added export buttons to ui panel
-added export selected only in ui panel
-added selection check on export
-error message dialog now splits on words
-cleanup error messages
-export all frames enabled as default (export mesh+anim)

2024-09-01
-sort objects by name. this is so skin files are consistent
-removed md5 bone layer system in favor of the checkbox 'Deform' on bones
-moved tool panel to 3D UI
-added tool panel option to export only 'Deform' bones
-cleaned up adding new action, so failure can delete correct action name
-export all anims option removed from batch export. all is now default.
-cleaned up properties for dialog boxes, duplicated moved to a common class.

2024-09-10
-exporter ignores hidden objects in the active layer


todo
====
replace numpy for floats
import anims if matching struct fails.

'''
# pylint: disable=locally-disabled, line-too-long, wrong-import-position, invalid-name, too-many-lines, missing-function-docstring


bl_info = {
    "name": "id tech 4 MD5 format",
    "author": "nemyax, 2.8 Update: Samson, hypov8",
    "version": (1, 11, 20240829),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "description": "Import and export md5mesh and md5anim",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

import bpy
import os
import bmesh
import os.path
import mathutils as mu
import math
import re
from numpy import (
    format_float_positional
)
from mathutils import(
    Quaternion,
    Matrix,
    Vector
)
from bpy.utils import (
    register_class,
    unregister_class
)
from bpy.props import (
    BoolProperty,
    FloatProperty,
    StringProperty,
    IntProperty,
    CollectionProperty,
    PointerProperty,
    EnumProperty,
)
from bpy_extras.io_utils import (
    ExportHelper,
    ImportHelper,
    path_reference_mode
)
from bpy.types import (
    Operator,
    OperatorFileListElement,
    PropertyGroup,
    WindowManager,
    Armature,
    Panel,
    TOPBAR_MT_file_import,
    TOPBAR_MT_file_export
)

class MD5_GlobalProps(PropertyGroup, Armature):
    ''' global values '''
    name = "MD5_GlobalProps"

    armature_list = [] #TODO multiple?
    bone_list = []
    mesh_list = []

    boneLayer_idx = IntProperty(
        name="Bone Layer id",
        description="Bone layer reserved for MD5 export",
        min=1, max=32,
        default=5
    )
    collection_prefix = StringProperty(
        name="collection prefix",
        default="",
    )
    errorMsg = StringProperty(
        name="errorMsg",
        default="",
    )
    export_onlySelectedMesh = BoolProperty(
        name="Only 'Selected' Mesh",
        description=(
            "Only exported selected mesh.\n" +
            "Disabled: exports all mesh inside active collection."),
        default=False
    )
    export_onlyDeformBones = BoolProperty(
        name="Only 'Deform' Bones",
        description=(
            "Export only bones marked as deform.\n"+
            "Disabled: export's all bones.\n" +
            "Note: 'Deform' checkbox found in 'Bone Properties->Deform'."
        ),
        default=True
    )

    def addCollection(self, context):
        ''' add objects and armature pointers '''
        for a in context[0]:
            self.armature_list.append(a)
        for b in context[1]:
            self.bone_list.append(b)
        for o in context[2]:
            self.mesh_list.append(o)

    def clear(self):
        ''' clear data '''
        self.armature_list.clear()
        self.bone_list.clear()
        self.mesh_list.clear()
        self.errorMsg = ""

    def getCollection(self):
        ''' return (bone,mesh) '''
        return (self.armature_list, self.bone_list, self.mesh_list)

    def getBoneMesh(self):
        ''' return (bone,mesh) '''
        return (self.bone_list, self.mesh_list)

    def getBones(self):
        ''' return bones '''
        return self.bone_list

    def getArmature(self):
        ''' return armature '''
        return self.armature_list


class MD5_Common_Prop:
    ''' common export options '''
    path_mode = path_reference_mode
    check_extension = True
    reorientDegrees = EnumProperty(
        items=(
            ('0', '0 Degrees', 'Do not reorient'),
            ('90', '90 Degrees ( X to Y )', 'Rotate 90 degrees (e.g. reorient facing +X to facing +Y)'),
            ('-90', '-90 Degrees ( Y to X )', 'Rotate -90 degrees (e.g. reorient facing +Y to facing +X'),
            ('180', '180 Degrees', 'Rotate 180 degrees')),
        name="Reorient Model",
        description=(
            "Degrees to rotate model during export." +
            " Useful to reorient models to face Y axis if desired.\n" +
            " 90 Degrees rotates clockwise from above.\n" +
            " -90 Rotates counter-clockwise from above."),
        default='0'
    )
    scaleFactor = FloatProperty(
        name="Scale",
        description="Scale all data",
        min=0.01, max=1000.0,
        soft_min=0.01,
        soft_max=1000.0,
        default=1.0,
    )


class MD5_CommonMesh_Prop:
    ''' common to mesh '''
    fixWindings = BoolProperty(
        name="Fix Eye Deform",
        description="Only select if having issues with materials flagged with eyeDeform",
        default=False
    )


class MD5_CommonAnim_Prop:
    ''' common to anims '''
    baseframeTPose = BoolProperty(
        name="T-Pose Baseframe",
        description=(
            "Exports the models T-Pose as the baseframe.\n"+
            "Disabled: Use the values from the first frame of animation to generate the baseframe."),
        default=False,
    )


def check_version(major, minor, _):
    '''
   Check blender version
    check_version(2, 80, 0) < 0:
    return:
      0: same version
      1: newer version
     -1: older version
    '''
    ver_id = tuple(bpy.app.version)
    cur_major, cur_minor = ver_id[0], ver_id[1]

    # matching major version
    if cur_major == major:
        if cur_minor == minor: # minor version identical
            return 0
        if cur_minor > minor: # minor version newer
            return 1
    # newer major ver
    if cur_major > major:
        return 1
    # older version
    return -1

def make_annotations(cls):
    """Add annotation attribute to fields to avoid Blender 2.8+ warnings
    https://github.com/OpenNaja/cobra-tools/blob/master/addon_updater_ops.py"""
    if not hasattr(bpy.app, "version") or check_version(2, 80, 0) < 0:
        return cls
    if check_version(2, 93, 0) < 0:
        bl_props = {k: v for k, v in cls.__dict__.items()
                    if isinstance(v, tuple)}
    else:
        bl_props = {k: v for k, v in cls.__dict__.items()
                    if isinstance(v, bpy.props._PropertyDeferred)}
    if bl_props:
        if '__annotations__' not in cls.__dict__:
            setattr(cls, '__annotations__', {})
        annotations = cls.__dict__['__annotations__']
        for k, v in bl_props.items():
            annotations[k] = v
            delattr(cls, k)
    return cls


###
### Import functions
###

### .md5mesh import

def read_md5mesh(context, path, matrix, mergeVertices):
    meshName = path.split(os.sep)[-1].split(".")[-2]
    collection = bpy.data.collections.new(meshName)
    context.scene.collection.children.link(collection)

    i = "\s+(\d+)"
    w = "\s+(.+?)"
    a = "(.+?)"
    j_re = re.compile(
        "\s*\""+a+"\""+w+"\s+\("+w*3+"\s+\)\s+\("+w*3+"\s+\).*")
    v_re = re.compile("\s*vert"+i+"\s+\("+w*2+"\s+\)"+i*2+".*")
    t_re = re.compile("\s*tri"+i*4+".*")
    w_re = re.compile("\s*weight"+i*2+w+"\s+\("+w*3+"\).*")
    e_re = re.compile("\s*}.*")
    js_re = re.compile("\s*joints\s+{.*")
    n_re = re.compile("\s*(numverts).*")
    m_re = re.compile("\s*mesh\s+{.*")
    s_re = re.compile("\s*shader\s+\""+a+"\".*")
    fh = open(path, "r")
    md5mesh = fh.readlines()
    fh.close()
    m = None
    while not m:
        m = js_re.match(md5mesh.pop(0))
    arm_o, ms = do_joints(context, md5mesh, j_re, e_re, matrix, meshName, collection)
    pairs = []
    while md5mesh:
        mat_name, bm = do_mesh(md5mesh, s_re, v_re, t_re, w_re, e_re, n_re, ms)
        if mergeVertices > 0.00:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=mergeVertices)
        pairs.append((mat_name, bm))
        skip_until(m_re, md5mesh)

    for mat_name, bm in pairs:
        translationtable = str.maketrans("\\", "/")
        tempstring = str.translate(mat_name, translationtable)
        lindex = str.rfind(tempstring, "/")
        if lindex == -1:
            lindex = 0
        tempstring = tempstring[lindex+1:len(tempstring)]
        mesh = bpy.data.meshes.new(tempstring+"_mesh")
        bm.to_mesh(mesh)
        bm.free()
        mesh_o = bpy.data.objects.new(tempstring, mesh)
        vgs = mesh_o.vertex_groups
        for jn, _ in ms:
            vgs.new(name=jn)
        arm_mod = mesh_o.modifiers.new(type='ARMATURE', name=meshName+"_MD5_skeleton")
        arm_mod.object = arm_o
        collection.objects.link(mesh_o)
        oldactive = context.view_layer.objects.active
        context.view_layer.objects.active = mesh_o

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.material_slot_add()
        try:
            mat = bpy.data.materials[mat_name]
        except KeyError:
            mat = bpy.data.materials.new(mat_name)
        mesh_o.material_slots[-1].material = mat
        bpy.ops.object.mode_set()
        context.view_layer.objects.active = oldactive

def do_mesh(md5mesh, s_re, v_re, t_re, w_re, e_re, n_re, ms):
    bm = bmesh.new()
    mat_name = gather(s_re, n_re, md5mesh)[0][0]
    vs, ts, ws = gather_multi([v_re, t_re, w_re], e_re, md5mesh)
    wd = bm.verts.layers.deform.verify()
    uvs = bm.loops.layers.uv.verify()
    for _, vi in enumerate(vs):
        wt, nwt = map(int, vi[3:])
        w0 = ws[wt]
        mtx = ms[int(w0[1])][1]
        xyz = mtx @ Vector(map(float, w0[3:]))
        new_v = bm.verts.new(xyz)
        bm.verts.index_update()
        for i in ws[wt:wt+nwt]:
            index = int(i[1])
            val = float(i[2])
            new_v[wd][index] = val
    if check_version(2, 80, 0) >= 0:
        bm.verts.ensure_lookup_table()
    for t in ts:
        # bm.verts.ensure_lookup_table()
        tvs = [bm.verts[a] for a in map(int, t[1:])]
        tvy = tvs
        try:
            new_f = bm.faces.new([tvy[2], tvy[0], tvy[1]]) # fix windings for eyedeform issue
            new_f.normal_flip() # seems normals need to be flipped for 2.8
            new_f.smooth = True # smooth faces
        except:
            continue
        bm.faces.index_update()
        for vn in tvs:
            ln = [l for l in new_f.loops if l.vert == vn][0]
            u0, v0 = map(float, vs[vn.index][1:3])
            ln[uvs].uv = (u0, 1.0 - v0)
    return mat_name, bm

def do_joints(context, md5mesh, j_re, e_re, correctMatrix, meshName, collection):
    joints = {}
    jdata = gather(j_re, e_re, md5mesh)
    for i, jd in enumerate(jdata):
        joints[i] = jd
    arm = bpy.data.armatures.new(meshName+"_MD5_Skeleton")
    arm_o = bpy.data.objects.new(meshName+"_MD5_Armature", arm)
    collection.objects.link(arm_o)
    context.view_layer.objects.active = arm_o
    bpy.ops.object.mode_set()
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm.edit_bones
    ms = []
    for j in joints.values():
        j_name = j[0]
        eb = ebs.new(j_name)
        p = int(j[1])
        if p >= 0:
            eb.parent = ebs[joints[p][0]]
        tx, ty, tz, rx, ry, rz = [float(a) for a in j[2:]]
        quat = -Quaternion(restore_quat(rx, ry, rz))
        mtx = Matrix.Translation((tx, ty, tz)) @ quat.to_matrix().to_4x4()
        mtx = correctMatrix.to_4x4() @ mtx
        ms.append((j_name, mtx))
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, 1.0, 0.0)
        eb.matrix = mtx
        eb.length = 5.0
    bpy.ops.object.mode_set()
    return arm_o, ms

### .md5anim import functions

def read_md5anim(context, fullPath, correctionMatrix, ao, action):
    ''' read anim files'''
    skel = bone_tree_blender(ao.data)

    fh = open(fullPath, "r")
    md5anim = fh.readlines()
    fh.close()
    w = "\s+(.+?)"
    a = "(.+?)"
    j_re = re.compile("\s*\""+a+"\""+w*3+".*")
    e_re = re.compile("\s*}.*")
    bf0_re = re.compile("\s*(baseframe)\s+{.*")
    bf1_re = re.compile("\s*\("+w*3+"\s+\)\s+\("+w*3+"\s+\).*")
    f_re = re.compile("\s*(frame).*")

    hier = gather(j_re, e_re, md5anim)
    for i, hi in enumerate(hier):
        jname, par, flags = hi[:-1]
        hier[i] = (i, jname, int(par), int(flags))
    md5skel = bone_tree_md5(hier)
    if skel != md5skel:
        return md5_error_messages(context, "no_arm_match"), {'CANCELLED'}
    skip_until(bf0_re, md5anim)
    bframe = gather(bf1_re, e_re, md5anim)
    bxfs = get_base_xforms(bframe)
    skip_until(f_re, md5anim)
    pbs = [ao.pose.bones[j[1]] for j in hier]
    frames = pad_frames(hier, bxfs, do_frames(e_re, md5anim))
    fcurves = get_fcurves(pbs, action)
    xf_keys = convert_xforms(context, pbs, transpose(frames), correctionMatrix)

    end_frame = len(frames) - 1
    context.scene.frame_set(0)
    bpy.data.scenes[0].frame_end = end_frame
    set_keys(flatten_channels(fcurves), flatten_frames(xf_keys), 0)

    return "Animation imported successfully.", {'FINISHED'}

def do_frames(e_re, md5anim):
    valid = re.compile("[-\.0-9\s]+")
    val = re.compile("(\S+)")
    result = [[]]
    while md5anim:
        l = md5anim.pop(0)
        if e_re.match(l):
            result.append([])
            continue
        if valid.match(l):
            vals = [float(a) for a in val.findall(l)]
            if vals:
                result[-1].append(vals)
    return [a for a in result if a]

def get_base_xforms(bframe):
    return [[float(b) for b in a] for a in bframe]

def convert_xforms(context, pbs, val_lists, correctionMatrix):
    result = []
    wm = context.window_manager
    print("cursor update max = "+str(len(val_lists)))
    wm.progress_begin(0, len(val_lists))
    curfr = 0
    for pb, states in zip(pbs, val_lists):
        curfr = curfr + 1
        wm.progress_update(curfr)
        result.append([])
        par = pb.parent
        if par:
            tweak = pb.bone.matrix_local.inverted() @ par.bone.matrix_local
        else:
            tweak = pb.bone.matrix_local.inverted() @ correctionMatrix
        for vl in states:
            l = Vector(vl[:3])
            q = Quaternion(restore_quat(vl[3], vl[4], vl[5]))
            mtx = q.to_matrix().to_4x4()
            mtx.translation = l
            mtx = tweak @ mtx
            xf = []
            xf.extend(mtx.translation[:])
            xf.extend(mtx.to_quaternion()[:])
            xf.extend(mtx.to_euler()[:])
            result[-1].append(xf)
    wm.progress_end()
    return result

def pad_frames(hier, bxfs, frames):
    result = []
    for val_lists in frames:
        result.append([])
        for j, xf in zip(hier, bxfs):
            xf0 = xf[:]
            flags = j[3]
            if not flags:
                vl = []
            else:
                vl = val_lists.pop(0)
            mask = 1
            for i in range(6):
                if mask & flags:
                    xf0[i] = vl.pop(0)
                mask *= 2
            result[-1].append(xf0)
    return result

def transpose(table):
    result = []
    if table:
        while table[0]:
            result.append([])
            for col in table:
                result[-1].append(col.pop(0))
    return result

def get_fcurves(pbs, action):
    fcurves = []
    l = "location"
    q = "rotation_quaternion"
    e = "rotation_euler"
    for pb in pbs:
        actiongroup = action.groups.new(pb.name)
        pb.keyframe_insert(l)
        pb.keyframe_insert(q)
        pb.keyframe_insert(e)
        entry = {l:{}, q:{}, e:{}}
        pbn = pb.name
        fc_re = re.compile("pose\.bones\[."+pbn+".\]\.("+l+"|"+q+"|"+e+")")
        for fc in pb.id_data.animation_data.action.fcurves:
            m = fc_re.match(fc.data_path)
            if m:
                key1 = m.group(1)
                key2 = fc.array_index
                entry[key1][key2] = fc
                fc.group = actiongroup
        fcurves.append(entry)
    return fcurves

def list_fcurves(fcurves):
    l = "location"
    q = "rotation_quaternion"
    e = "rotation_euler"
    return [
        fcurves[l][0], fcurves[l][1], fcurves[l][2],
        fcurves[q][0], fcurves[q][1], fcurves[q][2], fcurves[q][3],
        fcurves[e][0], fcurves[e][1], fcurves[e][2]]

def flatten_channels(fcurves):
    result = []
    for a in fcurves:
        result.extend([b.keyframe_points for b in list_fcurves(a)])
    return result

def flatten_frames(pbs): # :: [[[a]]] -> [[a]]
    result = []
    for b in pbs:
        temp = [[] for _ in range(10)]
        for frame in b:
            for i in range(10):
                temp[i].append(frame[i])
        result.extend(temp)
    return result

def set_keys(channels, val_lists, f_start):
    for ch, vl in zip(channels, val_lists):
        i = f_start
        for v in vl:
            ch.insert(i, v)
            i += 1

### parsing and utility functions

def gather(regex, end_regex, ls):
    return gather_multi([regex], end_regex, ls)[0]

def gather_multi(regexes, end_regex, ls):
    result = [[] for _ in regexes]
    n = len(regexes)
    while ls:
        l = ls.pop(0)
        if end_regex.match(l):
            break
        for i in range(n):
            m = regexes[i].match(l)
            if m:
                result[i].append(m.groups())
                break
    return result

def skip_until(regex, ls):
    while ls:
        if regex.match(ls.pop(0)):
            break

def restore_quat(rx, ry, rz):
    t = 1.0 - (rx * rx) - (ry * ry) - (rz * rz)
    if t < 0.0:
        return (0.0, rx, ry, rz)
    return (-math.sqrt(t), rx, ry, rz)

def bone_tree_blender(arm):
    return btb(None, [b for b in arm.bones if b.use_deform])

def btb(b, bs):
    ''' recursive; shouldn't matter for poxy md5 skeletons '''
    ch = sorted([a for a in bs if a.parent == b], key=lambda x: x.name)
    return [[c.name, btb(c, bs)] for c in ch]

def bone_tree_md5(lst):
    root = [a for a in lst if a[2] == -1][0]
    return [[root[1], btm(root, lst)]]

def btm(e, l):
    ch = sorted([a for a in l if a[2] == e[0]], key=lambda x: x[1])
    return [[c[1], btm(c, l)] for c in ch]

###
### Export functions
###

def record_parameters(correctionMatrix):
    return "".join([
        " // Parameters used during export:",
        " Reorient: {};".format(bool(correctionMatrix.to_euler()[2])),
        " Scale: {}".format(correctionMatrix.decompose()[2][0])])

def define_components(obj, bm, bones, correctionMatrix):
    scaleFactor = correctionMatrix.to_scale()[0]
    armature = [a for a in bpy.data.armatures if bones[0] in a.bones[:]][0]
    armatureObj = [o for o in bpy.data.objects if o.data == armature][0]
    boneNames = [b.name for b in bones]
    allVertGroups = obj.vertex_groups[:]
    weightGroupIndexes = [vg.index for vg in allVertGroups if vg.name in boneNames]
    uvData = bm.loops.layers.uv.active
    weightData = bm.verts.layers.deform.active
    tris = [[f.index, f.verts[2].index, f.verts[1].index, f.verts[0].index]
            for f in bm.faces] # reverse vert order to flip normal
    verts = []
    weights = []
    wtIndex = 0
    firstWt = 0
    for vert in bm.verts:
        vGroupDict = vert[weightData]
        wtDict = dict([(k, vGroupDict[k]) for k in vGroupDict.keys()
                       if k in weightGroupIndexes])
        u = vert.link_loops[0][uvData].uv.x
        v = 1 - vert.link_loops[0][uvData].uv.y # MD5 wants it flipped
        numWts = len(wtDict.keys())
        verts.append([vert.index, u, v, firstWt, numWts])
        wtScaleFactor = 1.0 / sum(wtDict.values())
        firstWt += numWts
        for vGroup in wtDict:
            bone = [b for b in bones
                    if b.name == allVertGroups[vGroup].name][0]
            boneIndex = bones.index(bone)
            coords4d =\
                bone.matrix_local.inverted() @\
                armatureObj.matrix_world.inverted() @\
                obj.matrix_world @\
                (vert.co.to_4d() * scaleFactor)
            x, y, z = coords4d[:3]
            weight = wtDict[vGroup] * wtScaleFactor
            wtEntry = [wtIndex, boneIndex, weight, x, y, z]
            weights.append(wtEntry)
            wtIndex += 1
    return (verts, tris, weights)

def compare_weights(wt1, wt2):
    '''
    return weight index or -1 when failed
    hypov8 compare each component
    '''
    # TODO optimize
    if  (wt1[1] == wt2[1] and # bone idx
         wt1[2] == wt2[2] and # weight
         wt1[3] == wt2[3] and # X
         wt1[4] == wt2[4] and # Y
         wt1[5] == wt2[5]): # Z
        return wt1[0] # return index
    return -1

def optimize_components(verts, weights):
    '''
    hypov8
    optimize weight by removing duplicates
    note: vertex that share the same cords/weights can use the same index
    '''
    out_verts = []
    out_weights = []
    curWtIdx = 0
    for v in verts:
        offset = curWtIdx
        matchNum = 0
        idxWts = v[3] # current weight index on vertex
        numWts = v[4] # weight counts

        for out_wts in out_weights:
            curWt = weights[idxWts + matchNum]
            idx = compare_weights(out_wts, curWt)
            if idx > -1:
                matchNum += 1
                if matchNum == numWts:
                    offset = idx - matchNum + 1
                    break
            elif matchNum > 0:
                break  # no more matching

        out_verts.append([v[0], v[1], v[2], offset, v[4]])
        if matchNum == 0: # no match, so add weight
            for k in range(idxWts, idxWts + numWts):
                wt = weights[k]
                out_weights.append([curWtIdx, wt[1], wt[2], wt[3], wt[4], wt[5]])
                curWtIdx += 1
    # TODO optimize.
    return (out_verts, out_weights)

def prnt_f(f, count=10):
    '''
    hypov8
    print float as short as posible
    1.0, 1.00001 etc..
    round decimal places to count. default 10
    TODO remove numpy
    '''
    return format_float_positional(
        float(f), precision=count, fractional=True, trim='0')

def make_hierarchy_block(bones, boneIndexLookup):
    block = ["hierarchy {\n"]
    xformIndex = 0
    for b in bones:
        parent_name = ""
        if b.parent:
            parentIndex = boneIndexLookup[b.parent.name]
            parent_name = b.parent.name
        else:
            parentIndex = -1
        block.append("\t\"{}\"\t{} 63 {}\t\t// {}\n".format(
            b.name, parentIndex, xformIndex, parent_name))
        xformIndex += 6
    block.append("}\n\n")
    return block

def make_baseframe_block(bones, correctionMatrix):
    block = ["baseframe {\n"]
    armature = bones[0].id_data
    armObject = [o for o in bpy.data.objects
                 if o.data == armature][0]
    armMatrix = armObject.matrix_world
    for b in bones:
        objSpaceMatrix = b.matrix_local
        if b.parent:
            bMatrix =\
            b.parent.matrix_local.inverted() @\
            armMatrix @\
            objSpaceMatrix
        else:
            bMatrix = correctionMatrix @ objSpaceMatrix
        xPos, yPos, zPos = bMatrix.translation
        xOrient, yOrient, zOrient = (-bMatrix.to_quaternion()).normalized()[1:]
        block.append(
            "\t( {} {} {} ) ( {} {} {} )\n".format(
                prnt_f(xPos), prnt_f(yPos), prnt_f(zPos),
                prnt_f(xOrient), prnt_f(yOrient), prnt_f(zOrient)))
    block.append("}\n\n")
    return block

def make_joints_block(bones, boneIndexLookup, correctionMatrix):
    block = []
    block.append("joints {\n")
    for b in bones:
        parent_name = ""
        if b.parent:
            parentIndex = boneIndexLookup[b.parent.name]
            parent_name = b.parent.name
        else:
            parentIndex = -1
        boneMatrix = correctionMatrix @ b.matrix_local
        xPos, yPos, zPos = boneMatrix.translation
        xOrient, yOrient, zOrient =\
        (-boneMatrix.to_quaternion()).normalized()[1:] # MD5 wants it negated
        block.append(
            "\t\"{}\"\t{} ( {} {} {} ) ( {} {} {} )\t\t// {}\n".format(
                b.name, parentIndex,
                prnt_f(xPos), prnt_f(yPos), prnt_f(zPos),
                prnt_f(xOrient), prnt_f(yOrient), prnt_f(zOrient),
                parent_name))
    block.append("}\n\n")
    return block

def make_mesh_block(obj, bones, correctionMatrix, fixWindings):
    shaderName = "default"
    ms = obj.material_slots
    if ms:
        taken = [s for s in ms if s.material]
        if taken:
            shaderName = taken[0].material.name
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    triangulate(cut_up(strip_wires(bm)))
    verts, tris, weights = define_components(obj, bm, bones, correctionMatrix)
    verts, weights = optimize_components(verts, weights)
    bm.free()
    block = []
    block.append("mesh {\n")
    block.append("\tshader \"{}\"\n\n".format(shaderName))
    # vertex
    block.append("\tnumverts {}\n".format(len(verts)))
    for v in verts:
        block.append(
            "\tvert {} ( {} {} ) {} {}\n".format(
                v[0],            # vertIndex
                prnt_f(v[1], 6), # texU,
                prnt_f(v[2], 6), # texV
                v[3],            # weightIndex,
                v[4]))           # weightCount
    # triangles
    block.append("\n\tnumtris {}\n".format(len(tris)))
    for t in tris:
        if fixWindings:
            # fix windings - current blender windings break eyeDeform in D3 materials
            block.append("\ttri {} {} {} {}\n".format(t[0], t[3], t[1], t[2]))
        else:
            block.append("\ttri {} {} {} {}\n".format(t[0], t[1], t[2], t[3]))
    # weights
    block.append("\n\tnumweights {}\n".format(len(weights)))
    for w in weights:
        block.append(\
        "\tweight {} {} {} ( {} {} {} )\n".format(
            w[0],            # weightIndex
            w[1],            # jointIndex
            prnt_f(w[2], 4), # weightValue
            prnt_f(w[3]),    # xPos
            prnt_f(w[4]),    # yPos
            prnt_f(w[5])))   # zPos
    block.append("}\n\n")
    return block

def strip_wires(bm):
    [bm.faces.remove(f) for f in bm.faces if len(f.verts) < 3]
    [bm.edges.remove(e) for e in bm.edges if not e.link_faces[:]]
    [bm.verts.remove(v) for v in bm.verts if v.is_wire]
    for seq in [bm.verts, bm.faces, bm.edges]:
        seq.index_update()
    return bm

def cut_up(bm):
    uvData = bm.loops.layers.uv.active
    for v in bm.verts:
        for e in v.link_edges:
            linkedFaces = e.link_faces
            if len(linkedFaces) > 1:
                uvSets = []
                for lf in linkedFaces:
                    uvSets.append([
                        l1[uvData].uv for l1 in lf.loops
                        if l1.vert == v][0])
                if uvSets.count(uvSets[0]) != len(uvSets):
                    e.tag = True
                    v.tag = True
        if v.tag:
            seams = [e for e in v.link_edges if e.tag]
            v.tag = False
            bmesh.utils.vert_separate(v, seams)
    for maybeBowTie in bm.verts: # seems there's no point in a proper test
        boundaries = [
            e for e in maybeBowTie.link_edges
            if len(e.link_faces) == 1]
        bmesh.utils.vert_separate(maybeBowTie, boundaries)
    for seq in [bm.verts, bm.faces, bm.edges]:
        seq.index_update()
    return bm

def triangulate(bm):
    nonTris = [f for f in bm.faces if len(f.verts) > 3]
    bmesh.ops.triangulate(bm, faces=nonTris)
    return bm

def write_md5mesh(filePath, prerequisites, correctionMatrix, fixWindings):
    bones, meshObjects = prerequisites
    boneIndexLookup = {}
    for b in bones:
        boneIndexLookup[b.name] = bones.index(b)
    md5joints = make_joints_block(bones, boneIndexLookup, correctionMatrix)
    md5meshes = []
    for mo in meshObjects:
        md5meshes.append(make_mesh_block(mo, bones, correctionMatrix, fixWindings))
    f = open(filePath, 'w')
    try:
        lines = []
        lines.append("MD5Version 10" + record_parameters(correctionMatrix) + "\n")
        lines.append("commandline \"\"\n")
        lines.append("\n")
        lines.append("numJoints " + str(len(bones)) + "\n")
        lines.append("numMeshes " + str(len(meshObjects)) + "\n")
        lines.append("\n")
        lines.extend(md5joints)
        for m in md5meshes:
            lines.extend(m)
        for line in lines:
            f.write(line)
    finally:
        f.close()
    return

def write_md5anim(context, filePath, prerequisites, correctionMatrix,
                  useTimeline, frame_range, baseframeTPose):
    '''
    export the .md5anim for the action currently
    associated with the armature animation'''

    context.view_layer.update()

    goBack = context.scene.frame_current

    if useTimeline:
        startFrame = context.scene.frame_preview_start
        endFrame = context.scene.frame_preview_end
    else:
        startFrame = int(frame_range[0])
        endFrame = int(frame_range[-1])

    bones, meshObjects = prerequisites
    armObj = [o for o in bpy.data.objects if o.data == bones[0].id_data][0]
    pBones = armObj.pose.bones
    boneIndexLookup = {}
    for b in bones:
        boneIndexLookup[b.name] = bones.index(b)
    hierarchy = make_hierarchy_block(bones, boneIndexLookup)

    genBaseFrame = False
    if not baseframeTPose:
        # write animated frame 1 to baseframe
        baseframe = ["baseframe {\n"]
        genBaseFrame = True
    else:
        baseframe = make_baseframe_block(bones, correctionMatrix)

    bounds = []
    frames = []
    for frame in range(startFrame, endFrame + 1):
        context.scene.frame_set(frame)
        verts = []
        for mo in meshObjects:
            bm = bmesh.new()
            depsgraph = context.evaluated_depsgraph_get()
            depsgraph.update()
            bm.from_object(mo, depsgraph)
            verts.extend([
                correctionMatrix @ mo.matrix_world @ v.co.to_4d()
                for v in bm.verts])
            bm.free()
        minX = min([co[0] for co in verts])
        minY = min([co[1] for co in verts])
        minZ = min([co[2] for co in verts])
        maxX = max([co[0] for co in verts])
        maxY = max([co[1] for co in verts])
        maxZ = max([co[2] for co in verts])
        bounds.append(
            "\t( {} {} {} ) ( {} {} {} )\n".format(
                prnt_f(minX), prnt_f(minY), prnt_f(minZ),
                prnt_f(maxX), prnt_f(maxY), prnt_f(maxZ))
        )

        frameBlock = ["frame {} {{\n".format(frame - startFrame)]
        scaleFactor = correctionMatrix.to_scale()[0]
        for b in bones:
            pBone = pBones[b.name]
            pBoneMatrix = pBone.matrix
            if pBone.parent:
                diffMatrix = pBone.parent.matrix.inverted() @ \
                    armObj.matrix_world @ (pBoneMatrix * scaleFactor)
            else:
                diffMatrix = correctionMatrix @ pBoneMatrix
            xPos, yPos, zPos = diffMatrix.translation
            xOrient, yOrient, zOrient =\
            (-diffMatrix.to_quaternion()).normalized()[1:]
            frameBlock.append(
                "\t{} {} {} {} {} {}\n".format(
                    prnt_f(xPos), prnt_f(yPos), prnt_f(zPos),
                    prnt_f(xOrient), prnt_f(yOrient), prnt_f(zOrient)))
            if genBaseFrame:
                baseframe.append(
                    "\t( {} {} {} ) ( {} {} {} )\n".format(
                        prnt_f(xPos), prnt_f(yPos), prnt_f(zPos),
                        prnt_f(xOrient), prnt_f(yOrient), prnt_f(zOrient)))

        if genBaseFrame:
            baseframe.append("}\n\n")
            genBaseFrame = False

        frameBlock.append("}\n\n")
        frames.extend(frameBlock)
    f = open(filePath, 'w')
    try:
        numJoints = len(bones)
        bounds.insert(0, "bounds {\n")
        bounds.append("}\n")
        bounds.append("\n")
        lines = []
        lines.append("MD5Version 10" + record_parameters(correctionMatrix) + "\n")
        lines.append("commandline \"\"\n")
        lines.append("\n")
        lines.append("numFrames " + str(endFrame - startFrame + 1) + "\n")
        lines.append("numJoints " + str(numJoints) + "\n")
        lines.append("frameRate " + str(context.scene.render.fps) + "\n")
        lines.append("numAnimatedComponents " + str(numJoints * 6) + "\n")
        lines.append("\n")
        for chunk in [hierarchy, bounds, baseframe, frames]:
            lines.extend(chunk)
        for line in lines:
            f.write(line)
    finally:
        f.close()
    context.scene.frame_set(goBack)
    return


###
### Operators and auxiliary functions
###

# Functions

def concat_strings(strings):
    result = ""
    for s in strings:
        result = result + "\n" + s
    return result

def md5_error_messages(context, eid, *details):
    ''' print error message '''

    if eid == 'no_selection':
        return ("Nothing selected to export.\n" +
                "Please select an object in the collection" +
                " you would like to export, and try again.")
    if eid == 'no_selectedmesh':
        return ("No valid mesh selected.\n" +
                "Select the mesh object you want to export, and retry export.")
    if eid == 'no_deformables':
        return ("No armature-deformed meshes found.\n" +
                "Select the collection or object you want to export, and retry export.")
    if eid == 'multiple_armatures':
        return ("The selected object, or the collection it belongs to, contains more than one armature.\n" +
                "Select an object in a collection that cantains and uses only one armature, and try again.")
    if eid == 'no_armature':
        return ("No deforming armature is associated with the selected object or it's collection.\n" +
                "Select the collection, or an object in the collection you want to export, and try again.")
    if eid == 'layer_empty':
        return ("The deforming armature in the collection has no bones.\n" +
                "Add bones to armature or check that 'Deform' is enabled on your deforming bones")
    if eid == 'missing_parents':
        return ("One or more bones in the armature have missing/invalid parents.\n" +
                "Include the parent bones(listed below) for export by enabling the 'Deform' option on each bone.\n" +
                "Offending bones: " + concat_strings(details[0]))
    if eid == 'orphans':
        return ("There are multiple root bones (listed below)" +
                " in the export-bound collection, but only one root bone is allowed in MD5.\n" +
                "Revise your armature's tree structure or check bone 'Deform' option so its parent is included.\n" +
                "Root bones: " + concat_strings(details[0]))
    if eid == 'unweighted_verts':
        return ("The '" + details[0][0] + "' object contains " + str(details[0][1]) +
                " vertices with no deformation weights assigned.\n" +
                "Valid MD5 data cannot be produced.\n" +
                "Paint non-zero weights on all the vertices in the mesh, and retry export.")
    if eid == 'zero_weight_verts':
        return ("The '" + details[0][0] + "' object contains " + str(details[0][1]) +
                " vertices with zero(<0.0001) weights assigned.\n" +
                "This can cause adverse effects.\n" +
                "Paint non-zero weights on all the vertices in the mesh, " +
                " or use the Clean operation in the weight paint tools.\n" +
                "Note: if using Clean, anything with a weight less 0.0001" +
                " is considered a zero weight, so use this limit in the clean tool.\n" +
                "Please correct zero weights and retry export.")
    if eid == 'no_uvs':
        return ("The '" + details[0] + "' object has no UV coordinates.\n" +
                "Valid MD5 data cannot be produced.\n" +
                "Unwrap the object or exclude it from your selection, and retry export.")
    if eid == 'no_arm':
        return ("No armature found to add animation to.\n" +
                "The active object is not an Armature, or the collection the active" +
                " object belongs to does not contain a valid armature.\n" +
                "Select a valid armature or object in the desired collection, and retry import.")
    if eid == 'no_arm_match':
        return ("The selected armature does not match the skeleton" +
                " in the file you are trying to import.\n" +
                "Make sure the bone names match and 'Deform' is checked on those bones.")

    return "Unhandled error"

def check_weighting(obj, bm, bones):
    boneNames = [b.name for b in bones]
    allVertGroups = obj.vertex_groups[:]
    # weightGroups = [vg for vg in allVertGroups if vg.name in boneNames]
    weightGroupIndexes = [vg.index for vg in allVertGroups if vg.name in boneNames]
    weightData = bm.verts.layers.deform.active
    unweightedVerts = 0
    zeroWeightVerts = 0
    for v in bm.verts:
        influences = [
            wgi for wgi in weightGroupIndexes
            if wgi in v[weightData].keys()]
        if not influences:
            unweightedVerts += 1
        else:
            for wgi in influences:
                if v[weightData][wgi] < 0.0001: # changed lower
                    zeroWeightVerts += 1
                    print("Zero Weight %s ( checking against limit of 0.0001 )", v[weightData][wgi])
                    v.select_set(True)
    return (unweightedVerts, zeroWeightVerts)

def is_export_go(context, what, collection):
    deformOnly = context.window_manager.md5_prop.export_onlyDeformBones

    #support single mesh export.
    #only supported by commandline, normal export is still as documented.
    if what == 'mesh':
        if context.active_object.type != 'MESH':
            return ['no_selectedmesh', None]
        meshObjects = [context.active_object]
    else: # 'meshes' 'anim' 'batch'
        meshObjects = [
            o for o in bpy.data.collections[collection.name].objects
            if o.data in bpy.data.meshes[:] and o.find_armature() and not o.hide_get()]
        meshObjects.sort(key=lambda o: o.name) #sort by name. for skin files
    if not meshObjects:
        return ['no_deformables', None]
    armatures = [a.find_armature() for a in meshObjects]
    armature = armatures[0]
    if armatures.count(armature) < len(meshObjects):
        return ['multiple_armatures', None]
    if deformOnly:
        # bpy.types.Bone
        bones = [b for b in armature.data.bones if b.use_deform]
    else:
        bones = armature.data.bones
    if not bones:
        return ['layer_empty', None]
    rootBones = [i for i in bones if not i.parent]
    if len(rootBones) > 1:
        boneList = []
        for rb in rootBones:
            boneList.append("- " + str(rb.name))
        return ['orphans', boneList]
    abandonedBones = [
        i for i in bones
        if i.parent and i.parent not in bones[:]]
    if abandonedBones:
        boneList = []
        for ab in abandonedBones:
            boneList.append("- " + str(ab.name))
        return ['missing_parents', boneList]
    if what != 'anim':
        for mo in meshObjects:
            bm = bmesh.new()
            bm.from_mesh(mo.data)
            unweightedVerts, zeroWeightVerts = check_weighting(mo, bm, bones)
            uvLayer = bm.loops.layers.uv.active
            bm.free()
            if unweightedVerts > 0:
                return ['unweighted_verts', (mo.name, unweightedVerts)]
            if zeroWeightVerts > 0:
                return ['zero_weight_verts', (mo.name, zeroWeightVerts)]
            if not uvLayer:
                return ['no_uvs', mo.name]
    return ['ok', (armatures, bones, meshObjects)]

def export_validation(context, what): #, collection):
    ''' populate objects/armatures. return True or false
    '''
    md5_prop = context.window_manager.md5_prop

    ao = context.active_object

    if not ao or len(context.selected_objects) == 0:
        msgLines = md5_error_messages(context, 'no_selection', None)
        bpy.ops.message.md5_messagebox('INVOKE_DEFAULT', message=msgLines)
        md5_prop.errorMsg = "Nothing selected."
        return False # {'CANCELLED'}

    collection = ao.users_collection[0]
    md5_prop.collection_prefix = "("+collection.name+")_"

    checkResult = is_export_go(context, what, collection)
    if checkResult[0] == 'ok':
        md5_prop.addCollection(checkResult[-1])
        return True

    msgLines = md5_error_messages(context, checkResult[0], checkResult[1])
    bpy.ops.message.md5_messagebox('INVOKE_DEFAULT', message=msgLines)
    md5_prop.errorMsg = checkResult[0] + " see Console"
    return False # {'CANCELLED'}

def delete_action(context, prepend, actionNameToDelete):
    print("Deleting Action: {}".format(actionNameToDelete))
    bpy.data.actions.remove(bpy.data.actions[actionNameToDelete])
    return

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

# Operators

### Import UI

class ImportMD5Mesh(Operator, ImportHelper, MD5_Common_Prop):
    '''Import an .MD5mesh file as a new Collection'''
    bl_idname = "import_scene.import_md5mesh"
    bl_label = 'Import MD5 Mesh'
    bl_options = {'PRESET'}
    filename_ext = ".md5mesh"

    filter_glob = StringProperty(
        default="*.md5mesh",
        options={'HIDDEN'}
    )
    mergeVerticesCM = bpy.props.FloatProperty(
        name="Merge Vertices",
        description="Automatically weld near vertices (in centimetres). ( 0.0 = Default Disabled )",
        min=0.00, max=1.00,
        soft_min=0.00,
        soft_max=1.00,
        default=0.00
    )

    def execute(self, context):
        rotdeg = float(self.reorientDegrees)
        orientationTweak = Matrix.Rotation(math.radians(rotdeg), 4, 'Z')

        scaleTweak = Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak @ scaleTweak

        read_md5mesh(
            context, self.filepath, correctionMatrix,
            self.mergeVerticesCM * 0.01)
        self.report({'INFO'}, "Mesh Imported")
        return {'FINISHED'}


class ImportMD5Anim(Operator, ImportHelper, MD5_Common_Prop):
    ''' Import .MD5anim files '''
    bl_idname = "import_scene.import_md5anim"
    bl_label = 'Import MD5 Anim'
    bl_options = {'PRESET'}
    bl_description = (
        "Import one or more .MD5anim files into dopesheet actions associated" +
        " with the collection of the active object")
    filename_ext = ".md5anim"

    prepend = BoolProperty(
        name="Prepend action name",
        description=(
            "Prepend the collection name to the animation name in the action editor.\n" +
            " Necessary to bulk export actions and identify which actions are associated with which collection."),
        default=True,
    )
    filter_glob = StringProperty(
        default="*.md5anim",
        options={'HIDDEN'}
    )
    files = CollectionProperty(
        name="MD5Anim files",
        type=OperatorFileListElement,
    )
    directory = StringProperty(
        subtype='DIR_PATH'
    )
    md5_export_helpText = [
        "Select one or more .md5anim files",
        "  to import as Blender actions.",
        "Actions are available",
        "  in the dopesheet action editor.",
        "Note: Load a valid mesh first."
    ]

    def draw(self, context):
        layout = self.layout
        row = layout.column_flow(columns=1, align=False)
        row.use_property_split = True
        row.prop(self, "path_mode", )
        row.prop(self, "reorientDegrees")
        row.prop(self, "scaleFactor")
        row.prop(self, "prepend")

        # help window
        box = layout.box()
        row = box.column_flow(columns=1, align=True)
        for s in self.__class__.md5_export_helpText:
            row.label(text=s)

    def invoke(self, context, event):

        #check the active object first
        ao = context.active_object
        if ao and ao.type == 'ARMATURE' and ao.data.bones[:]:
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}

        #if the active object isn't a valid armature, get it's collection and check

        if ao:
            collection = ao.users_collection[0]
            print("Using armature")
        else:
            collection = context.view_layer.active_layer_collection

        print(collection)
        meshObjects = [
            o for o in bpy.data.collections[collection.name].objects
            if o.data in bpy.data.meshes[:] and o.find_armature()]
        if len(meshObjects):
            armatures = [a.find_armature() for a in meshObjects]
            armature = armatures[0]
            if armature.data.bones[:]:
                context.view_layer.objects.active = armature
                context.window_manager.fileselect_add(self)
                return {'RUNNING_MODAL'}

        # no valid armature selected or in the active collection
        msg = md5_error_messages(context, "no_arm")
        bpy.ops.message.md5_messagebox('INVOKE_DEFAULT', message=msg)
        # print(msg)
        # self.report({'ERROR'}, msg)
        return {'CANCELLED'}

    def execute(self, context):
        errors = 0
        successes = 0
        errorString = ""
        res = []

        rotdeg = float(self.reorientDegrees)
        orientationTweak = Matrix.Rotation(math.radians(rotdeg), 4, 'Z')
        scaleTweak = Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak @ scaleTweak

        ao = context.active_object
        if not ao.animation_data:
            ao.animation_data_create()

        for newAnim in self.files:
            actionName = newAnim.name
            fullAnimPath = os.path.join(self.directory, actionName)
            print("Importing: " + fullAnimPath)

            if self.prepend:
                prefix = "(" + ao.name.replace("_MD5_Armature", "") + ")_"
                actionName = prefix + actionName

            action = bpy.data.actions.new(name=actionName) # new action
            actionName = action.name # get unique name

            try:
                msg, res = read_md5anim(
                    context, fullAnimPath, correctionMatrix, ao, action)
                if res == {'CANCELLED'}:
                    self.report({'ERROR'}, msg)
                    print(msg)
                    errors = errors + 1
                    errorString = errorString + newAnim.name + ","
                    delete_action(context, self.prepend, actionName)
                else:
                    successes = successes + 1
                    # set active anim
                    ao.animation_data.action = action
            except:
                errors = errors + 1
                errorString = errorString + newAnim.name + ","
                delete_action(context, self.prepend, actionName)
                continue

        msg = str(successes) + " .md5anim files successfully imported as actions.\n"

        if errors >= 1:
            msg = msg + "ERROR! The following " + str(errors) + " imports failed - see System Console for details.\n" + errorString

        bpy.ops.message.md5_messagebox('INVOKE_DEFAULT', message=msg)

        if res:
            if errors == 0:
                self.report({'INFO'}, "Animation Imported")
            return res
        else:
            return {'CANCELLED'}


class VIEW3D_PT_MD5Panel(Panel):
    """ MD5 parameters panel in the scene context of the properties editor """
    bl_label = "MD5 Models"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS' if check_version(2, 80, 0) < 0 else 'UI'
    bl_category = 'MD5-Models'

    def draw(self, context):
        md5_props = context.window_manager.md5_prop
        layout = self.layout

        # import buttons
        box = layout.box()
        col1 = box.column()
        col1.label(text="Import:")
        col2 = box.column()
        col2.scale_y = 1.4
        col2.operator_context = 'INVOKE_DEFAULT'
        col2.operator("import_scene.import_md5mesh", text="MD5 Mesh", icon='OUTLINER_OB_MESH')
        col2.operator("import_scene.import_md5anim", text="MD5 Anim", icon='OUTLINER_OB_ARMATURE')

        # export buttons
        box = layout.box()
        col1 = box.column()
        col1.label(text="Export:")
        col1.prop(md5_props, "export_onlySelectedMesh")
        col1.prop(md5_props, "export_onlyDeformBones")
        col2 = box.column()
        col2.scale_y = 1.4
        col2.operator_context = 'INVOKE_DEFAULT'
        col2.operator("export_scene.export_md5mesh", text="MD5 Mesh", icon='OUTLINER_OB_MESH')
        col2.operator("export_scene.export_md5anim", text="MD5 Animations", icon='OUTLINER_OB_ARMATURE')
        col2.operator("export_scene.export_md5batch", text="MD5 Mesh+Anim", icon='PACKAGE')

### Export UI

class ExportMD5Mesh(Operator, ExportHelper, MD5_Common_Prop, MD5_CommonMesh_Prop):
    '''Export All objects in the parent collection of the active object as an .MD5mesh.'''
    bl_idname = "export_scene.export_md5mesh"
    bl_label = 'Export MD5 Mesh'
    # bl_options = {'PRESET'}
    filename_ext = ".md5mesh"

    filter_glob = StringProperty(
        default="*.md5mesh",
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        md5_prop = context.window_manager.md5_prop
        md5_prop.clear()

        if md5_prop.export_onlySelectedMesh:
            exp_type = 'mesh'
        else:
            exp_type = 'meshes'

        if not export_validation(context, exp_type): # 'mesh'
            self.report({'ERROR'}, md5_prop.errorMsg)
            return {'CANCELLED'}

        ao = context.active_object
        collection = ao.users_collection[0]
        self.filepath = collection.name
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        md5_prop = context.window_manager.md5_prop

        orientationTweak = Matrix.Rotation(math.radians(float(self.reorientDegrees)), 4, 'Z')
        scaleTweak = Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak @ scaleTweak
        write_md5mesh(
            self.filepath, md5_prop.getBoneMesh(),
            correctionMatrix, self.fixWindings)
        self.report({'INFO'}, "Mesh Exported")
        return {'FINISHED'}


class ExportMD5Anim(Operator, ExportHelper, MD5_Common_Prop, MD5_CommonAnim_Prop):
    '''Export the action currently associated with the active object as an .MD5anim'''
    bl_idname = "export_scene.export_md5anim"
    bl_label = 'Export MD5 Animations'
    filename_ext = ".md5anim"

    filter_glob = StringProperty(
        default="*.md5anim",
        options={'HIDDEN'},
    )
    useTimeline = BoolProperty(
        name="Use timeline",
        description=(
            "Only export frames indicated by timeline preview 'Start' and 'End' frames values.\n" +
            "Disabled: all action frames will be exported."),
        default=False,
    )

    def invoke(self, context, event):
        md5_prop = context.window_manager.md5_prop
        md5_prop.clear()

        if not export_validation(context, 'anim'):
            self.report({'ERROR'}, md5_prop.errorMsg)
            return {'CANCELLED'}

        # get filepath
        armatures = md5_prop.getArmature()
        self.filepath = remove_prefix(
            armatures[0].animation_data.action.name,
            md5_prop.collection_prefix)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        md5_prop = context.window_manager.md5_prop
        armatures, bones, meshObjects = md5_prop.getCollection()

        orientationTweak = Matrix.Rotation(math.radians(float(self.reorientDegrees)), 4, 'Z')
        scaleTweak = Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak @ scaleTweak

        write_md5anim(
            context, self.filepath,
            (bones, meshObjects),
            correctionMatrix,
            self.useTimeline,
            armatures[0].animation_data.action.frame_range,
            self.baseframeTPose)
        self.report({'INFO'}, "Animation Exported")
        return {'FINISHED'}


class ExportMD5Batch(Operator, ExportHelper, MD5_Common_Prop, MD5_CommonMesh_Prop, MD5_CommonAnim_Prop):
    '''Export all objects in the parent collection of the active object as an .MD5mesh.
    Export the active action or all actions as .MD5anim files'''
    bl_idname = "export_scene.export_md5batch"
    bl_label = 'Export MD5 Mesh+Anim'
    filename_ext = ".md5mesh"

    filter_glob = StringProperty(
        default="*.md5mesh",
        options={'HIDDEN'},
    )
    stripPrepend = BoolProperty(
        name="Save Without Prefix",
        description="Strip the prepended collection name from exported action names.",
        default=True,
    )
    onlyPrepend = BoolProperty(
        name="Only Prefixed Actions",
        description="Only export actions prepended with the collection name.",
        default=False,
    )

    def invoke(self, context, event):
        md5_prop = context.window_manager.md5_prop
        md5_prop.clear()

        if md5_prop.export_onlySelectedMesh:
            exp_type = 'mesh'
        else:
            exp_type = 'batch'

        if not export_validation(context, exp_type):
            self.report({'ERROR'}, md5_prop.errorMsg)
            return {'CANCELLED'}

        ao = context.active_object
        collection = ao.users_collection[0]
        self.filepath = collection.name
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        md5_prop = context.window_manager.md5_prop
        armatures, bones, meshObjects = md5_prop.getCollection()

        orientationTweak = Matrix.Rotation(math.radians(float(self.reorientDegrees)), 4, 'Z')
        scaleTweak = Matrix.Scale(self.scaleFactor, 4)
        correctionMatrix = orientationTweak @ scaleTweak
        batch_directory = os.path.dirname(self.filepath)
        collection_Prefix = md5_prop.collection_prefix  #"("+collection.name+")_"
        armature = armatures[0]
        #write the mesh
        write_md5mesh(self.filepath, (bones, meshObjects), correctionMatrix, self.fixWindings)

        # write all frames for all actions
        oldAction = armature.animation_data.action
        for exportAction in bpy.data.actions:
            name = exportAction.name
            frame_range = exportAction.frame_range
            print("Checking action name " + name + " to see if in collection " + collection_Prefix)
            if not self.onlyPrepend or name.startswith(collection_Prefix):
                #export this action
                armature.animation_data.action = exportAction
                if self.stripPrepend:
                    name = remove_prefix(name, collection_Prefix)
                if not name.endswith(".md5anim"):
                    name = name + ".md5anim"
                self.filepath = os.path.join(batch_directory, name)
                print("Exporting animation "+self.filepath)
                write_md5anim(context, self.filepath, (bones, meshObjects), correctionMatrix,
                                False, frame_range, self.baseframeTPose)

        armature.animation_data.action = oldAction
        self.report({'INFO'}, "Mesh+Anims Exported")
        return {'FINISHED'}


class MD5_MessageBox(Operator):
    bl_idname = "message.md5_messagebox"
    bl_label = "INFO:"

    message = bpy.props.StringProperty(
        name="message",
        description="message",
        default=""
    )
    message2 = bpy.props.StringProperty(
        name="message2",
        description="message2",
        default=""
    )

    def execute(self, context):
        # self.report({'INFO'}, self.message)
        # print(self.message)
        return {'FINISHED'}

    def invoke(self, context, event):
        print(self.message)
        return context.window_manager.invoke_props_dialog(self, width=600)

    def chunkstring(self, string, length):
        outList = [""]
        words = string.split(' ')
        i = 0
        for word in words:
            if (len(outList[i]) + len(word)) < length:
                outList[i] = outList[i] + word +" "
            else:
                outList.append(word + " ")
                i += 1
        return outList

    def draw(self, context):
        layout = self.layout
        row = layout.column_flow(columns=1, align=False)

        lines = self.message.split('\n')
        for i in range(len(lines)):
            for chunk in self.chunkstring(lines[i], 80):
                row.label(text=chunk)


def menu_func_import_mesh(self, context):
    self.layout.operator(
        ImportMD5Mesh.bl_idname, text="MD5 Mesh (.md5mesh)")
def menu_func_import_anim(self, context):
    self.layout.operator(
        ImportMD5Anim.bl_idname, text="MD5 Animation(s) (.md5anim)")

def menu_func_export_mesh(self, context):
    self.layout.operator(
        ExportMD5Mesh.bl_idname, text="MD5 Mesh (.md5mesh)")
def menu_func_export_anim(self, context):
    self.layout.operator(
        ExportMD5Anim.bl_idname, text="MD5 Animation (.md5anim)")
def menu_func_export_batch(self, context):
    self.layout.operator(
        ExportMD5Batch.bl_idname, text="MD5 Mesh and Animation(s)")

classes = (
    ImportMD5Mesh,
    ImportMD5Anim,
    ExportMD5Mesh,
    ExportMD5Anim,
    ExportMD5Batch,
    MD5_MessageBox,
    MD5_GlobalProps,
    VIEW3D_PT_MD5Panel,
)


def register():
    make_annotations(MD5_Common_Prop)
    make_annotations(MD5_CommonMesh_Prop)
    make_annotations(MD5_CommonAnim_Prop)

    for cls in classes:
        make_annotations(cls)
        register_class(cls)

    WindowManager.md5_prop = PointerProperty(type=MD5_GlobalProps)
    TOPBAR_MT_file_import.append(menu_func_import_mesh)
    TOPBAR_MT_file_import.append(menu_func_import_anim)
    TOPBAR_MT_file_export.append(menu_func_export_mesh)
    TOPBAR_MT_file_export.append(menu_func_export_anim)
    TOPBAR_MT_file_export.append(menu_func_export_batch)

def unregister():
    for cls in reversed(classes):
        unregister_class(cls)

    del WindowManager.md5_prop
    TOPBAR_MT_file_import.remove(menu_func_import_mesh)
    TOPBAR_MT_file_import.remove(menu_func_import_anim)
    TOPBAR_MT_file_export.remove(menu_func_export_mesh)
    TOPBAR_MT_file_export.remove(menu_func_export_anim)
    TOPBAR_MT_file_export.remove(menu_func_export_batch)

if __name__ == "__main__":
    register()
