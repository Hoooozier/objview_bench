# -*- coding: utf-8 -*-
"""
Blender 4.2 batch GLB -> standardized OBJ converter for benchmark assets.

Standardized export policy:
1) Import GLB
2) Flatten parent hierarchy influence into current mesh-object transforms
3) Apply transforms
4) FORCE join all mesh objects into one mesh per asset
5) Triangulate
6) Preserve source UV for reading original textures
7) FORCE create a dedicated BakeUV atlas for bake target/output
8) FORCE bake one basecolor texture to a fixed filename: texture.png
9) Replace all materials with one baked material
10) Export OBJ + MTL + texture.png
11) Save blender_export_info.json

Design choice:
- DO NOT apply explicit export axis conversion
- Keep exported OBJ geometry coordinates as close as possible to Blender current mesh coordinates
- Explicitly export normals and UVs
- Always standardize texture representation to one baked texture for stable downstream PyTorch3D use

Important implementation detail:
- Source/original textures should continue reading from the ORIGINAL/source UV
- BakeUV is used only as the OUTPUT UV for the baked texture
"""

import bpy
import os
import sys
import json
import shutil
import argparse
import traceback
from pathlib import Path


# --------------------------------------------------
# args / utils
# --------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Expected '--' in Blender argv.")
    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Blender 4.2 batch GLB -> standardized OBJ")
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--single_glb", type=str, default=None, help="Optional single GLB path to process.")
    parser.add_argument("--uid_filter", type=str, default=None, help="Optional single uid (stem) to process from input_dir.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--bake_resolution", type=int, default=2048)
    parser.add_argument("--bake_margin", type=int, default=16)
    parser.add_argument("--bake_device", type=str, default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--unwrap_method", type=str, default="smart_project", choices=["smart_project"])
    parser.add_argument("--smart_project_angle_limit", type=float, default=1.15192)
    parser.add_argument("--smart_project_island_margin", type=float, default=0.03)

    # index slicing after sorted file list
    parser.add_argument("--start_idx", type=int, default=0, help="Start index after sorting (inclusive).")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index after sorting (exclusive). -1 means all remaining.")

    args = parser.parse_args(argv)

    if args.single_glb is None and args.input_dir is None:
        raise RuntimeError("Either --input_dir or --single_glb must be provided.")

    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def purge_orphans():
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass


def ensure_object_mode():
    obj = bpy.context.object
    if obj is not None:
        try:
            if obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass


def clear_scene():
    ensure_object_mode()

    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    datablock_names = [
        "meshes",
        "materials",
        "images",
        "textures",
        "cameras",
        "lights",
        "armatures",
        "actions",
        "collections",
        "node_groups",
        "curves",
        "metaballs",
        "grease_pencils",
        "grease_pencils_v3",
    ]

    for name in datablock_names:
        collection = getattr(bpy.data, name, None)
        if collection is None:
            continue
        for block in list(collection):
            try:
                if block.users == 0:
                    collection.remove(block)
            except Exception:
                pass

    purge_orphans()


def clean_asset_output_dir(asset_dir: Path):
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)


def list_mesh_objects(objs=None):
    if objs is None:
        objs = bpy.context.scene.objects
    return [obj for obj in objs if obj.type == 'MESH']


def select_only(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    if objs:
        bpy.context.view_layer.objects.active = objs[0]


def activate_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def snapshot_object_names():
    return set(bpy.data.objects.keys())


def collect_new_objects(before_names):
    after_names = set(bpy.data.objects.keys())
    new_names = after_names - before_names
    return [bpy.data.objects[name] for name in new_names if name in bpy.data.objects]


def get_object_diagnostics(obj):
    if obj is None or obj.type != 'MESH':
        return {}

    mesh = obj.data
    uv_names = [uv.name for uv in mesh.uv_layers]
    active_uv_name = None
    active_render_uv_name = None

    try:
        if mesh.uv_layers.active is not None:
            active_uv_name = mesh.uv_layers.active.name
    except Exception:
        active_uv_name = None

    try:
        for uv in mesh.uv_layers:
            if getattr(uv, "active_render", False):
                active_render_uv_name = uv.name
                break
    except Exception:
        active_render_uv_name = None

    return {
        "object_name": obj.name,
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "polygon_count": len(mesh.polygons),
        "uv_layer_names": uv_names,
        "active_uv_name": active_uv_name,
        "active_render_uv_name": active_render_uv_name,
        "material_slot_names": [
            slot.material.name if slot.material is not None else None
            for slot in obj.material_slots
        ],
        "unique_material_count": get_unique_material_count(obj),
    }


def write_error_info(asset_dir: Path, uid: str, glb_path: Path, exc: Exception, extra=None):
    err = {
        "uid": uid,
        "source_glb": str(glb_path),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "blender_version": ".".join(map(str, bpy.app.version)),
    }
    if extra is not None:
        err["diagnostics"] = extra
    save_json(err, asset_dir / "blender_export_error.json")


# --------------------------------------------------
# import / preprocess
# --------------------------------------------------

def import_glb(glb_path: str):
    before = snapshot_object_names()
    bpy.ops.import_scene.gltf(filepath=glb_path)
    imported_objs = collect_new_objects(before)
    return imported_objs


def apply_transforms(objs):
    if not objs:
        raise RuntimeError("No mesh objects to transform.")

    ensure_object_mode()

    for obj in objs:
        if obj.type == 'MESH' and obj.data is not None and obj.data.users > 1:
            obj.data = obj.data.copy()

    select_only(objs)

    # Flatten parent hierarchy while preserving current visual transforms.
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # Bake current object transforms into mesh data.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def force_join_all_meshes(mesh_objs):
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after import.")

    ensure_object_mode()

    if len(mesh_objs) == 1:
        activate_object(mesh_objs[0])
        return mesh_objs[0], False, 1

    select_only(mesh_objs)
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    if obj is None or obj.type != "MESH":
        raise RuntimeError("Failed to join mesh objects.")

    return obj, True, len(mesh_objs)


def triangulate_active_mesh():
    obj = bpy.context.view_layer.objects.active
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Active object is not a mesh.")

    ensure_object_mode()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def get_material_slot_count(obj):
    if obj is None or obj.type != "MESH":
        return 0
    return len(obj.material_slots)


def get_unique_material_count(obj):
    if obj is None or obj.type != "MESH":
        return 0

    mats = [slot.material for slot in obj.material_slots if slot.material is not None]
    return len({m.name_full for m in mats})


# --------------------------------------------------
# UV / material helpers
# --------------------------------------------------

def set_active_uv(obj, uv_name: str):
    """
    Explicitly set a UV layer as:
    - active_index
    - active
    - active_render
    """
    if obj.type != "MESH":
        raise RuntimeError("set_active_uv expects a mesh object.")

    mesh = obj.data
    uv = mesh.uv_layers.get(uv_name)
    if uv is None:
        raise RuntimeError(f"UV layer not found: {uv_name}")

    for idx, layer in enumerate(mesh.uv_layers):
        if layer.name == uv_name:
            mesh.uv_layers.active_index = idx
            break

    try:
        mesh.uv_layers.active = uv
    except Exception:
        pass

    for layer in mesh.uv_layers:
        try:
            layer.active_render = (layer.name == uv_name)
        except Exception:
            pass


def get_preferred_source_uv_name(obj):
    """
    Choose the UV map that should be used to READ original textures/materials.
    Priority:
    1) current active_render UV
    2) current active UV
    3) first existing UV
    4) create a SourceUV via smart_project if none exists
    """
    if obj is None or obj.type != "MESH":
        raise RuntimeError("get_preferred_source_uv_name expects a mesh object.")

    mesh = obj.data

    # 1) active_render
    try:
        for uv in mesh.uv_layers:
            if getattr(uv, "active_render", False):
                return uv.name
    except Exception:
        pass

    # 2) active
    try:
        if mesh.uv_layers.active is not None:
            return mesh.uv_layers.active.name
    except Exception:
        pass

    # 3) first existing
    if len(mesh.uv_layers) > 0:
        return mesh.uv_layers[0].name

    # 4) create one if nothing exists
    uv_name = "SourceUV"
    mesh.uv_layers.new(name=uv_name)
    set_active_uv(obj, uv_name)

    activate_object(obj)
    ensure_object_mode()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.03)
    bpy.ops.object.mode_set(mode='OBJECT')

    set_active_uv(obj, uv_name)
    return uv_name


def attach_source_uv_to_image_texture_nodes(obj, source_uv_name: str):
    """
    Force source image textures to read from the ORIGINAL/source UV map,
    instead of accidentally falling back to the newly created BakeUV.

    Strategy:
    - For each material on the object
    - For each ShaderNodeTexImage
    - If its Vector input is NOT linked, attach a UV Map node pointing to source_uv_name
    - Keep existing Vector links untouched
    """
    if obj is None or obj.type != "MESH":
        raise RuntimeError("attach_source_uv_to_image_texture_nodes expects a mesh object.")

    mesh = obj.data
    if mesh.uv_layers.get(source_uv_name) is None:
        raise RuntimeError(f"Source UV not found: {source_uv_name}")

    patch_records = []

    for slot_idx, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat is None or not mat.use_nodes or mat.node_tree is None:
            continue

        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links

        mat_record = {
            "material_name": mat.name,
            "patched_image_texture_nodes": [],
        }

        for node in nodes:
            if node.type != 'TEX_IMAGE':
                continue

            vec_input = node.inputs.get("Vector")
            if vec_input is None:
                continue

            if vec_input.is_linked:
                continue

            uv_node = nodes.new(type='ShaderNodeUVMap')
            uv_node.name = f"UVMap_Source_{source_uv_name}"
            uv_node.uv_map = source_uv_name
            uv_node.location = (node.location.x - 220, node.location.y)

            try:
                links.new(uv_node.outputs["UV"], vec_input)
                mat_record["patched_image_texture_nodes"].append(node.name)
            except Exception:
                try:
                    nodes.remove(uv_node)
                except Exception:
                    pass

        if len(mat_record["patched_image_texture_nodes"]) > 0:
            patch_records.append(mat_record)

    return patch_records


def create_bake_uv_map(obj, uv_name="BakeUV", angle_limit=1.15192, island_margin=0.03):
    """
    ALWAYS create or refresh a dedicated bake UV atlas.
    This is the final export UV used by the benchmark pipeline.
    """
    if obj.type != "MESH":
        raise RuntimeError("create_bake_uv_map expects a mesh object.")

    mesh = obj.data

    existing = mesh.uv_layers.get(uv_name)
    if existing is not None:
        mesh.uv_layers.remove(existing)

    mesh.uv_layers.new(name=uv_name)
    set_active_uv(obj, uv_name)

    activate_object(obj)
    ensure_object_mode()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    bpy.ops.object.mode_set(mode='OBJECT')

    set_active_uv(obj, uv_name)
    return uv_name


# --------------------------------------------------
# export validation
# --------------------------------------------------

def export_obj_blender42(obj_path: str, obj):
    """
    Blender 4.2 OBJ export operator.
    """
    ensure_dir(os.path.dirname(obj_path))
    select_only([obj])

    bpy.ops.wm.obj_export(
        filepath=obj_path,
        export_selected_objects=True,
        export_animation=False,
        export_materials=True,
        export_triangulated_mesh=False,
        export_normals=True,
        export_uv=True,
        path_mode='STRIP',
    )


def _parse_map_statement_path(rest: str):
    option_specs = {
        '-blendu': 1, '-blendv': 1, '-boost': 1,
        '-mm': 2, '-o': 3, '-s': 3, '-t': 3,
        '-texres': 1, '-clamp': 1, '-bm': 1,
        '-imfchan': 1, '-type': 1,
    }

    parts = rest.split()
    if not parts:
        return None

    i = 0
    while i < len(parts):
        token = parts[i]
        if token in option_specs:
            i += 1 + option_specs[token]
            continue
        return " ".join(parts[i:]).strip()
    return None


def parse_mtl_texture_refs(mtl_path: Path):
    texture_refs = []
    if not mtl_path.exists():
        return texture_refs

    valid_prefixes = {
        "map_Kd",
        "map_Ka",
        "map_Ks",
        "map_d",
        "bump",
        "map_Bump",
        "disp",
        "decal",
        "refl",
        "norm",
    }

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split(maxsplit=1)
            if len(parts) < 2:
                continue

            key = parts[0]
            rest = parts[1]
            if key in valid_prefixes:
                tex_path = _parse_map_statement_path(rest)
                if tex_path:
                    texture_refs.append(tex_path)

    return texture_refs


def resolve_texture_paths(asset_dir: Path, texture_refs):
    resolved = []
    for ref in texture_refs:
        candidates = []

        p = Path(ref)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((asset_dir / ref).resolve())
            candidates.append((asset_dir / Path(ref).name).resolve())

        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    resolved.append(str(c))
                    break
            except Exception:
                pass
    return resolved


def validate_textured_obj_trio(asset_dir: Path, uid: str):
    obj_path = asset_dir / f"{uid}.obj"
    mtl_path = asset_dir / f"{uid}.mtl"
    standardized_texture = asset_dir / "texture.png"

    info = {
        "obj_exists": obj_path.exists(),
        "mtl_exists": mtl_path.exists(),
        "standardized_texture_exists": standardized_texture.exists(),
        "texture_refs": [],
        "resolved_textures": [],
        "has_resolved_texture": False,
        "is_valid_textured_trio_for_pipeline": False,
    }

    if not info["obj_exists"] or not info["mtl_exists"]:
        return False, info

    refs = parse_mtl_texture_refs(mtl_path)
    resolved = resolve_texture_paths(asset_dir, refs)

    info["texture_refs"] = refs
    info["resolved_textures"] = resolved
    info["has_resolved_texture"] = len(resolved) > 0
    info["is_valid_textured_trio_for_pipeline"] = (
        len(resolved) > 0 and standardized_texture.exists()
    )

    return info["is_valid_textured_trio_for_pipeline"], info


# --------------------------------------------------
# bake standardized export
# --------------------------------------------------

def ensure_cycles(device: str):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    actual = 'CPU'
    try:
        scene.cycles.device = device
        actual = getattr(scene.cycles, 'device', 'UNKNOWN')
    except Exception:
        try:
            scene.cycles.device = 'CPU'
            actual = getattr(scene.cycles, 'device', 'CPU')
        except Exception:
            actual = 'UNKNOWN'

    return {
        "render_engine": scene.render.engine,
        "cycles_device_requested": device,
        "cycles_device_actual": actual,
    }


def build_bake_target_materials(obj, bake_img):
    """
    Attach an image node as the active bake target on ALL materials of the object.
    Also temporarily disable Metallic / Transmission contributions on Principled
    nodes to avoid black or unstable areas in diffuse-color bake.
    """
    activate_object(obj)

    if len(obj.material_slots) == 0:
        mat = bpy.data.materials.new(name=f"{obj.name}_DummyMat")
        mat.use_nodes = True
        obj.data.materials.append(mat)

    bake_nodes_info = []

    for slot_idx, slot in enumerate(obj.material_slots):
        mat = slot.material

        if mat is None:
            mat = bpy.data.materials.new(name=f"{obj.name}_AutoMat_{slot_idx}")
            mat.use_nodes = True
            slot.material = mat

        mat.use_nodes = True
        nt = mat.node_tree
        nodes = nt.nodes

        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                if 'Metallic' in node.inputs:
                    node.inputs['Metallic'].default_value = 0.0
                    while node.inputs['Metallic'].is_linked:
                        try:
                            nt.links.remove(node.inputs['Metallic'].links[0])
                        except Exception:
                            break

                if 'Transmission' in node.inputs:
                    node.inputs['Transmission'].default_value = 0.0
                    while node.inputs['Transmission'].is_linked:
                        try:
                            nt.links.remove(node.inputs['Transmission'].links[0])
                        except Exception:
                            break

        bake_node = nodes.new(type='ShaderNodeTexImage')
        bake_node.name = "BAKE_IMAGE_NODE"
        bake_node.image = bake_img
        bake_node.location = (-500, 0)

        for n in nodes:
            n.select = False
        bake_node.select = True
        nodes.active = bake_node

        bake_nodes_info.append((mat, bake_node))

    return bake_nodes_info


def prune_to_single_baked_material(obj, baked_img_path: str):
    activate_object(obj)

    baked_mat = bpy.data.materials.new(name="BakedMaterial")
    baked_mat.use_nodes = True
    nt = baked_mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for n in list(nodes):
        nodes.remove(n)

    tex = nodes.new(type='ShaderNodeTexImage')
    tex.location = (-400, 0)
    img = bpy.data.images.load(baked_img_path, check_existing=True)
    img.colorspace_settings.name = 'sRGB'
    tex.image = img
    tex.interpolation = 'Linear'

    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (-100, 0)
    try:
        principled.inputs["Metallic"].default_value = 0.0
    except Exception:
        pass

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)

    links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(baked_mat)

    for poly in mesh.polygons:
        poly.material_index = 0


def bake_basecolor_to_texture(
    obj,
    asset_dir: Path,
    uid: str,
    bake_resolution: int,
    bake_margin: int,
    bake_device: str,
    angle_limit: float,
    island_margin: float,
    source_uv_name: str,
):
    render_diag = ensure_cycles(bake_device)

    # Ensure source textures keep reading from ORIGINAL/source UV.
    source_uv_patch_info = attach_source_uv_to_image_texture_nodes(obj, source_uv_name)

    # Create dedicated target bake UV.
    bake_uv_name = create_bake_uv_map(
        obj,
        uv_name="BakeUV",
        angle_limit=angle_limit,
        island_margin=island_margin,
    )

    # BakeUV is for output only.
    set_active_uv(obj, bake_uv_name)

    baked_img_name = "texture.png"
    baked_img_path = str((asset_dir / baked_img_name).resolve())

    existing_img = bpy.data.images.get(baked_img_name)
    if existing_img is not None:
        try:
            bpy.data.images.remove(existing_img)
        except Exception:
            pass

    bake_img = bpy.data.images.new(
        name=baked_img_name,
        width=bake_resolution,
        height=bake_resolution,
        alpha=False,
        float_buffer=False,
    )
    bake_img.generated_color = (0.5, 0.5, 0.5, 1.0)
    bake_img.filepath_raw = baked_img_path
    bake_img.file_format = 'PNG'
    bake_img.colorspace_settings.name = 'sRGB'

    bake_nodes_info = build_bake_target_materials(obj, bake_img)

    scene = bpy.context.scene
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = bake_margin
    scene.render.bake.target = 'IMAGE_TEXTURES'

    activate_object(obj)
    set_active_uv(obj, bake_uv_name)
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})

    bake_img.save()

    for mat, b_node in bake_nodes_info:
        try:
            if mat and mat.node_tree and b_node in mat.node_tree.nodes:
                mat.node_tree.nodes.remove(b_node)
        except Exception:
            pass

    prune_to_single_baked_material(obj, baked_img_path)
    set_active_uv(obj, bake_uv_name)

    bake_diag = {
        "source_uv_name": source_uv_name,
        "bake_uv_name": bake_uv_name,
        "source_uv_patch_info": source_uv_patch_info,
        **render_diag,
    }

    return baked_img_name, baked_img_path, bake_diag


# --------------------------------------------------
# per-asset processing
# --------------------------------------------------

def process_one_glb(
    glb_path: Path,
    output_root: Path,
    overwrite: bool,
    bake_resolution: int,
    bake_margin: int,
    bake_device: str,
    smart_project_angle_limit: float,
    smart_project_island_margin: float,
):
    uid = glb_path.stem
    asset_dir = output_root / uid
    obj_path = asset_dir / f"{uid}.obj"
    mtl_path = asset_dir / f"{uid}.mtl"

    if obj_path.exists() and not overwrite:
        trio_ok, trio_info = validate_textured_obj_trio(asset_dir, uid)
        expected_texture = asset_dir / "texture.png"

        if trio_ok and expected_texture.exists():
            info = {
                "uid": uid,
                "source_glb": str(glb_path),
                "blender_version": ".".join(map(str, bpy.app.version)),
                "force_join_all_meshes": True,
                "triangulated": True,
                "uv_created_for_export": False,
                "did_join_multiple": None,
                "mesh_object_count_before_join": None,
                "export_mode": "standardized_baked_skip_existing",
                "bake_used": True,
                "bake_resolution": int(bake_resolution),
                "bake_margin": int(bake_margin),
                "bake_device_requested": str(bake_device),
                "material_slot_count": None,
                "unique_material_count": None,
                "force_bake_due_to_multi_material": True,
                "obj_path": str(obj_path),
                "mtl_path": str(mtl_path),
                "texture_files": [Path(p).name for p in trio_info["resolved_textures"]],
                "baked_texture_name": "texture.png",
                "baked_texture_path": str(expected_texture),
                "validation": trio_info,
            }
            save_json(info, asset_dir / "blender_export_info.json")
            return info

    clean_asset_output_dir(asset_dir)
    clear_scene()

    imported_objs = import_glb(str(glb_path))
    mesh_objs = list_mesh_objects(imported_objs)
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after GLB import.")

    pre_join_diag = {
        "imported_object_count": len(imported_objs),
        "imported_mesh_object_count": len(mesh_objs),
        "imported_mesh_names": [obj.name for obj in mesh_objs],
    }

    apply_transforms(mesh_objs)

    active_obj, did_join_multiple, mesh_object_count_before_join = force_join_all_meshes(mesh_objs)
    activate_object(active_obj)
    triangulate_active_mesh()

    source_uv_name = get_preferred_source_uv_name(active_obj)

    material_slot_count = get_material_slot_count(active_obj)
    unique_material_count = get_unique_material_count(active_obj)

    post_join_diag_before_bake = get_object_diagnostics(active_obj)
    post_join_diag_before_bake["source_uv_name"] = source_uv_name

    baked_texture_name, baked_texture_path, bake_diag = bake_basecolor_to_texture(
        obj=active_obj,
        asset_dir=asset_dir,
        uid=uid,
        bake_resolution=bake_resolution,
        bake_margin=bake_margin,
        bake_device=bake_device,
        angle_limit=smart_project_angle_limit,
        island_margin=smart_project_island_margin,
        source_uv_name=source_uv_name,
    )

    set_active_uv(active_obj, "BakeUV")
    export_obj_blender42(str(obj_path), active_obj)

    trio_ok, trio_info = validate_textured_obj_trio(asset_dir, uid)
    trio_info["material_slot_count"] = material_slot_count
    trio_info["unique_material_count"] = unique_material_count
    trio_info["standardized_export"] = True
    trio_info["expected_texture_name"] = "texture.png"

    if not trio_ok:
        raise RuntimeError(
            "OBJ export invalid after standardized bake export. "
            f"validation={json.dumps(trio_info, ensure_ascii=False)}"
        )

    texture_files = [Path(p).name for p in trio_info["resolved_textures"]]
    post_join_diag_after_bake = get_object_diagnostics(active_obj)

    info = {
        "uid": uid,
        "source_glb": str(glb_path),
        "blender_version": ".".join(map(str, bpy.app.version)),
        "force_join_all_meshes": True,
        "mesh_object_count_before_join": int(mesh_object_count_before_join),
        "did_join_multiple": bool(did_join_multiple),
        "triangulated": True,
        "uv_created_for_export": False,
        "export_mode": "standardized_baked",
        "bake_used": True,
        "bake_resolution": int(bake_resolution),
        "bake_margin": int(bake_margin),
        "bake_device_requested": str(bake_device),
        "material_slot_count": material_slot_count,
        "unique_material_count": unique_material_count,
        "force_bake_due_to_multi_material": True,
        "obj_path": str(obj_path),
        "mtl_path": str(mtl_path),
        "texture_files": texture_files,
        "baked_texture_name": baked_texture_name,
        "baked_texture_path": str(baked_texture_path),
        "validation": trio_info,
        "diagnostics": {
            "pre_join": pre_join_diag,
            "post_join_before_bake": post_join_diag_before_bake,
            "post_join_after_bake": post_join_diag_after_bake,
            "bake": bake_diag,
        },
    }

    save_json(info, asset_dir / "blender_export_info.json")
    purge_orphans()
    return info


# --------------------------------------------------
# main
# --------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.single_glb is not None:
        single_path = Path(args.single_glb)
        if not single_path.exists() or not single_path.is_file():
            raise RuntimeError(f"--single_glb not found: {single_path}")
        glb_files = [single_path]
        print(f"[INFO] Single-file mode: 1 file -> {single_path}", flush=True)

    else:
        input_dir = Path(args.input_dir)
        glb_files_all = sorted([p for p in input_dir.glob("*.glb") if p.is_file()])
        print(f"[INFO] Found {len(glb_files_all)} GLB files in {input_dir}", flush=True)

        if args.uid_filter is not None:
            glb_files_all = [p for p in glb_files_all if p.stem == args.uid_filter]
            print(f"[INFO] UID filter: {args.uid_filter} -> {len(glb_files_all)} files", flush=True)

        if len(glb_files_all) == 0:
            print("[WARN] No GLB files found after filtering.", flush=True)
            return

        start_idx = max(0, args.start_idx)
        end_idx = len(glb_files_all) if args.end_idx < 0 else min(args.end_idx, len(glb_files_all))

        if start_idx >= end_idx:
            print(
                f"[WARN] Empty slice after indexing: start_idx={start_idx}, end_idx={end_idx}, total={len(glb_files_all)}",
                flush=True,
            )
            return

        glb_files = glb_files_all[start_idx:end_idx]

        print(
            f"[INFO] Processing sorted slice [{start_idx}:{end_idx}) -> {len(glb_files)} files",
            flush=True,
        )

    success = 0
    fail = 0

    for idx, glb_path in enumerate(glb_files, 1):
        uid = glb_path.stem
        asset_dir = output_dir / uid

        try:
            info = process_one_glb(
                glb_path=glb_path,
                output_root=output_dir,
                overwrite=args.overwrite,
                bake_resolution=args.bake_resolution,
                bake_margin=args.bake_margin,
                bake_device=args.bake_device,
                smart_project_angle_limit=args.smart_project_angle_limit,
                smart_project_island_margin=args.smart_project_island_margin,
            )
            success += 1
            print(
                f"[INFO] ({idx}/{len(glb_files)}) Done: {uid} | "
                f"export_mode={info['export_mode']} | "
                f"texture={info.get('baked_texture_name', 'texture.png')} | "
                f"did_join={info['did_join_multiple']} | "
                f"mesh_count_before_join={info['mesh_object_count_before_join']} | "
                f"material_slots={info['material_slot_count']} | "
                f"unique_materials={info['unique_material_count']} | "
                f"textures={len(info.get('texture_files', []))}",
                flush=True,
            )
        except Exception as e:
            fail += 1
            ensure_dir(asset_dir)
            active = bpy.context.view_layer.objects.active
            extra_diag = {
                "active_object": get_object_diagnostics(active) if active else None,
                "scene_object_count": len(bpy.context.scene.objects),
            }
            write_error_info(asset_dir, uid, glb_path, e, extra=extra_diag)
            print(f"[ERROR] ({idx}/{len(glb_files)}) Failed: {uid} | {e}", flush=True)

    print("[INFO] Finished.", flush=True)
    print(f"[INFO] Success: {success}", flush=True)
    print(f"[INFO] Failed: {fail}", flush=True)


if __name__ == "__main__":
    main()

"""
Example:

# single uid from a directory
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b -P "glb_blender_obj.py" -- --input_dir "geometry_sampled/glb" --output_dir "geometry_sampled/obj" --overwrite --bake_device GPU --uid_filter 03e08488bf4d49cfa92b2b41a575ae72

# single explicit file
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b -P "glb_blender_obj.py" -- --single_glb "geometry_sampled/glb/03e08488bf4d49cfa92b2b41a575ae72.glb" --output_dir "geometry_sampled/obj" --overwrite --bake_device GPU

# sliced batch 1
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b -P "glb_blender_obj.py" -- --input_dir "geometry_sampled/glb" --output_dir "geometry_sampled/obj" --overwrite --bake_device GPU --start_idx 0 --end_idx 7000

# sliced batch 2
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b -P "glb_blender_obj.py" -- --input_dir "geometry_sampled/glb" --output_dir "geometry_sampled/obj" --overwrite --bake_device GPU --start_idx 7000 --end_idx 12000
"""
