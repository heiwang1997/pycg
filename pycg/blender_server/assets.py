"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import bpy
import uuid
import logging
from pathlib import Path
import bpy_types
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, \
    EnumProperty
import numpy as np
import functools
import bgl
import bmesh
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader


ANNOTATE_PALETTE = np.asarray([
    [0, 0, 0],          # Black
    [31, 119, 180],     # Blue
    [255, 127, 14],     # Orange
    [44, 160, 44],      # Green
    [214, 39, 40],      # Red
    [148, 103, 189],    # Purple
    [140, 86, 75],      # Brown
    [227, 119, 194],    # Pink
]).astype(np.float32) / 255.


@functools.lru_cache(maxsize=None)
def get_shader(shader_name):
    shader_base = Path(__file__).parent / "shaders"
    with (shader_base / shader_name).open("r") as f:
        return f.read()


class AssetData:
    def __init__(self, obj):
        self.object = obj
        self.shader = None
        self.data = None
        self.batch = None
        self.attributes = {}
        self.pc_edit_proxy = None
        self.textures = []


class GlobalProperty(bpy.types.PropertyGroup):
    """
    Global status.
    """
    def update_labeling_update(self, context):
        if context.object is None or context.object.parent is None:
            return
        if self.labeling_update:
            context.object.parent.pycg_asset.shader_type = 'LABEL'
        else:
            context.object.parent.pycg_asset.shader_type = 'FLAT'

    labeling_update: BoolProperty(name="Whether update will give labels", default=False, update=update_labeling_update)

    @classmethod
    def register(cls):
        bpy.types.Scene.pycg_globals = PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        del bpy.types.Scene.pycg_globals


class AssetProperty(bpy.types.PropertyGroup):
    """
    These properties can be saved along with blender.
    You should use this using your best rather than AssetData.
    """
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    geometry_type: EnumProperty(name="Type", items=[
        ('MESH', "Triangle Mesh", ""),
        ('PC', "Point Cloud", ""),
        ('LINESET', "Line Set", "")
    ], default='MESH', description="Geometry type")

    def update_shader_type(self, context):
        if self.uuid not in AssetManager.data.keys():
            logging.warning("update_shader_type: the object is not registered.")
            return

        # Should change new shader.
        if self.geometry_type not in ['PC', 'LINESET']:
            logging.warning(f"Error changing shader type. Current geometry type {self.geometry_type} not allowed.")
            return

        obj_data = AssetManager.data[self.uuid]
        if self.shader_type == "FLAT":
            shader = GPUShader(get_shader("pc_flat.vert"), get_shader("pc_flat.frag"))
            batch = batch_for_shader(shader, 'POINTS', {"position": obj_data.data["xyz"],
                                                        "color": obj_data.data["rgb"]})
        elif self.shader_type == "NORMAL":
            shader = GPUShader(get_shader("pc_normal.vert"), get_shader("pc_normal.frag"))
            batch = batch_for_shader(shader, 'POINTS', {"position": obj_data.data["xyz"],
                                                        "normal": obj_data.data["normal"]})
        elif self.shader_type == "LABEL":
            pseudo_color = np.ones((obj_data.data["xyz"].shape[0], 4), dtype=np.float32)
            pseudo_color[:, 0:3] = ANNOTATE_PALETTE[obj_data.data["label"]]
            shader = GPUShader(get_shader("pc_flat.vert"), get_shader("pc_flat.frag"))
            batch = batch_for_shader(shader, 'POINTS', {"position": obj_data.data["xyz"],
                                                        "color": pseudo_color})
        else:
            raise NotImplementedError

        obj_data.shader = shader
        obj_data.batch = batch

    shader_type: EnumProperty(name="Shader", items=[
        ('NONE', "No shader", "For non-opengl assets"),
        ('FLAT', "Flat Shading", ""),
        ('NORMAL', "Normal mapping", ""),
        ('LABEL', "Labels", "")
    ], default='NONE', update=update_shader_type)

    def update_material_type(self, context):
        if self.uuid not in AssetManager.data.keys():
            logging.warning("update_material_type: the object is not registered.")
            return

        if self.geometry_type not in ['MESH']:
            logging.warning(f"Error changing material type. Current geometry type {self.geometry_type} not allowed.")
            return

        import style
        if self.material_type == "ORIGIN":
            style.set_origin_material(self.uuid)
        else:
            raise NotImplementedError

    material_type: EnumProperty(name="Material", items=[
        ('ORIGIN', "Derived from give attributes", ""),
        ('METAL', "Metallic look", "")
    ], default='ORIGIN', update=update_material_type)

    point_size: IntProperty(name="Size", default=1, min=1, max=10, subtype='PIXEL', description="Point size", )
    alpha_radius: FloatProperty(name="Radius", default=1.0, min=0.001, max=1.0, precision=3, subtype='FACTOR',
                                description="Adjust point circular discard radius", )
    global_alpha: FloatProperty(name="Alpha", default=1.0, min=0.0, max=1.0, precision=2, subtype='FACTOR',
                                description="Adjust alpha of points displayed", )

    @classmethod
    def register(cls):
        bpy.types.Object.pycg_asset = PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        del bpy.types.Object.pycg_asset


class AssetManager:
    """
    Usage:
        This manager will automatically setup initial attribute of the empty object for controlling.
    """

    data = {}

    @staticmethod
    def check_shape(arr: np.ndarray, dims: list):
        if arr is None:
            return True
        if arr.ndim != len(dims):
            logging.warning(f"Dim check not match: {arr.shape} vs. {dims}")
            return False
        for di in range(len(dims)):
            if dims[di] != -1 and dims[di] != arr.shape[di]:
                logging.warning(f"Dim check not match at dim {di}: {arr.shape} vs. {dims}")
                return False
        return True

    @classmethod
    def handler(cls):
        for data_uuid in list(cls.data.keys()):
            datum = cls.data[data_uuid]
            if 'invalid>' in str(datum.object):
                del cls.data[data_uuid]
                logging.info(f"Object {data_uuid} deleted")
            elif datum.object.visible_get():
                cls.render(data_uuid)
            # TODO: Unify delete semantics.

    @classmethod
    def render(cls, uuidx):
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnable(bgl.GL_BLEND)

        shader = cls.data[uuidx].shader
        batch = cls.data[uuidx].batch
        obj = cls.data[uuidx].object
        attr = obj.pycg_asset

        if shader is not None and batch is not None:
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", obj.matrix_world)
            shader.uniform_float("point_size", attr.point_size)
            shader.uniform_float("alpha_radius", attr.alpha_radius)
            shader.uniform_float("global_alpha", attr.global_alpha)
            batch.draw(shader)

        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        bgl.glDisable(bgl.GL_BLEND)

    @classmethod
    def clear_all(cls):
        for datum in cls.data.values():
            if datum.object.active_material is not None:
                bpy.data.materials.remove(datum.object.active_material, do_unlink=True)
            if datum.object.data is None:
                bpy.data.objects.remove(datum.object, do_unlink=True)
            else:
                bpy.data.meshes.remove(datum.object.data, do_unlink=True)
            if datum.pc_edit_proxy is not None:
                bpy.data.meshes.remove(datum.pc_edit_proxy.data, do_unlink=True)
            for tex_img in datum.textures:
                bpy.data.images.remove(tex_img, do_unlink=True)

    @classmethod
    def create_entity(cls, kwargs):
        from inspect import signature

        geom_method = {
            'PC': cls.create_pointcloud,
            'MESH': cls.create_triangle_mesh
        }[kwargs['geometry_type']]
        method_kwargs = signature(geom_method).parameters.keys()
        new_kwargs = {'new_uuid': kwargs.get('uuid', str(uuid.uuid1()))}
        for k, v in kwargs.items():
            if k in method_kwargs:
                new_kwargs[k] = v
        return geom_method(**new_kwargs)

    @classmethod
    def get_entity(cls, uuidx):
        assert uuidx in cls.data.keys()
        tpe = cls.data[uuidx].object.pycg_asset.geometry_type
        return {
            'PC': cls.__get_pointcloud,
            'MESH': cls.__get_triangle_mesh
        }[tpe](uuidx)

    @classmethod
    def create_triangle_mesh(cls, new_uuid: str, vert: np.ndarray, faces: np.ndarray, vert_colors: np.ndarray,
                             triangle_uvs: np.ndarray, textures: list, attributes: dict):
        if not cls.check_shape(vert, [-1, 3]) or not cls.check_shape(faces, [-1, 3]) \
                or not cls.check_shape(vert_colors, [-1, 4]) or not cls.check_shape(triangle_uvs, [-1, 2]):
            return None

        new_mesh = bpy.data.meshes.new(name=f"TriangleMesh-{new_uuid}")
        new_mesh.vertices.add(vert.shape[0])
        new_mesh.loops.add(3 * faces.shape[0])
        new_mesh.polygons.add(faces.shape[0])

        new_mesh.vertices.foreach_set("co", vert.flatten().astype(np.float32))
        new_mesh.polygons.foreach_set("loop_total", np.full((faces.shape[0], ), 3, dtype=np.int32))
        new_mesh.polygons.foreach_set("loop_start", np.arange(faces.shape[0]).astype(np.int32) * 3)
        new_mesh.polygons.foreach_set("vertices", faces.ravel())
        new_mesh.update(calc_edges=True, calc_edges_loose=False)

        if vert_colors is not None:
            mesh_color_layer = new_mesh.vertex_colors.new()
            mesh_color_layer.data.foreach_set("color", vert_colors[faces.ravel()].flatten())

        if triangle_uvs is not None:
            uv_layer = new_mesh.uv_layers.new()
            uv_layer.data.foreach_set("uv", triangle_uvs.ravel())

        new_object = bpy.data.objects.new(f"TriangleMesh-{new_uuid}", new_mesh)
        bpy.context.scene.collection.objects.link(new_object)
        new_object.pycg_asset.uuid = new_uuid

        new_data = AssetData(new_object)
        new_data.attributes = attributes
        cls.data[new_uuid] = new_data

        # TODO: Maybe we can re-use the texture.
        for tex_id, tex in enumerate(textures):
            new_tex = bpy.data.images.new(name=f"Tex{tex_id}-{new_uuid}", width=tex.shape[1], height=tex.shape[0],
                                          alpha=True, float_buffer=False)
            new_tex.pixels = tex.ravel().astype(np.float16)
            new_data.textures.append(new_tex)

        new_object.pycg_asset.geometry_type = 'MESH'
        new_object.pycg_asset.material_type = 'ORIGIN'
        new_object.pycg_asset.update_material_type(None)

        return new_object

    @classmethod
    def __get_triangle_mesh(cls, uuidx):
        cur_obj = cls.data[uuidx].object
        n_point = len(cur_obj.data.vertices)

        if len(cur_obj.data.uv_layers) > 0:
            uv_layer = cur_obj.data.uv_layers[0]
            uv_data = np.empty(len(uv_layer.data) * 2, dtype=np.float32)
            uv_layer.data.foreach_get("uv", uv_data)
            uv_data = uv_data.reshape(-1, 2)
        else:
            uv_data = None

        return {
            "vert": n_point,
            "uv": uv_data,
            "pose": [*cur_obj.location[:], *cur_obj.rotation_quaternion[:]]
        }

    @classmethod
    def create_pointcloud(cls, new_uuid: str, xyz: np.ndarray, rgb: np.ndarray = None, normal: np.ndarray = None,
                          label: np.ndarray = None):
        if not cls.check_shape(xyz, [-1, 3]) or not cls.check_shape(rgb, [-1, 4]) or \
                not cls.check_shape(normal, [-1, 3]) or not cls.check_shape(label, [-1]):
            return None

        if rgb is None:
            rgb = np.zeros((xyz.shape[0], 4), dtype=np.float32)
            rgb[:, 3] = 1.0
        if normal is None:
            normal = np.zeros_like(xyz)
            normal[:, 0] = 1.0
        if label is None:
            label = np.zeros((xyz.shape[0]), dtype=int)

        new_object = bpy.data.objects.new(f"PointCloud-{new_uuid}", None)
        bpy.context.scene.collection.objects.link(new_object)
        new_object.pycg_asset.uuid = new_uuid

        new_data = AssetData(new_object)
        new_data.data = {"xyz": xyz, "rgb": rgb, "normal": normal, "label": label}
        cls.data[new_uuid] = new_data

        new_object.pycg_asset.geometry_type = 'PC'
        new_object.pycg_asset.shader_type = 'FLAT'

        return new_object

    @classmethod
    def __get_pointcloud(cls, uuidx):
        cur_data = cls.data[uuidx]
        return {
            "xyz": np.copy(cur_data.data['xyz']),
            "rgb": np.copy(cur_data.data['rgb']),
            "normal": np.copy(cur_data.data['normal']),
            "label": np.copy(cur_data.data['label'])
        }


class PYCG_PT_AssetPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = "pycg Asset"

    @classmethod
    def poll(cls, context):
        if context.object is None:
            return False
        if context.object.pycg_asset.uuid in AssetManager.data.keys():
            return True
        # Proxy mesh selected for editing.
        if context.object.parent and context.object.parent.pycg_asset.uuid in AssetManager.data.keys():
            return True
        return False

    def draw(self, context):
        cur_obj = context.object
        sub = self.layout.column()

        if cur_obj.pycg_asset.uuid != '':
            # Normal Mode
            # Type Display:
            r = sub.row()
            s = r.split(factor=0.33)
            s.label(text='Type')
            s = s.split(factor=1.0)
            s.label(text=cur_obj.pycg_asset.geometry_type)
            sub.separator()

            # Shader Selector
            sub.prop(cur_obj.pycg_asset, "shader_type", expand=True)
            sub.prop(cur_obj.pycg_asset, "point_size")

            sub.operator('pycg.pc_edit_start')

            sub.template_palette(context.tool_settings.image_paint, "palette", color=True)
        else:
            # Edit Mode
            sub.label(text='Hint')
            sub.label(text='- B: Box Select')
            sub.label(text='- C: Circle Select')
            sub.label(text='- Ctrl + RMB: Lasso Select')
            sub.operator('pycg.pc_edit_update')
            sub.prop(context.scene.pycg_globals, "labeling_update")
            sub.operator('pycg.pc_edit_end')

            sub.template_palette(context.tool_settings.image_paint, "palette", color=True)


class PYCG_OT_pc_edit_start(bpy.types.Operator):
    bl_idname = "pycg.pc_edit_start"
    bl_label = "PC Edit Start"
    bl_description = "Create helper object and switch to it"

    @classmethod
    def poll(cls, context):
        if context.object is None:
            return False
        if context.object.pycg_asset.uuid not in AssetManager.data.keys():
            return False
        if AssetManager.data[context.object.pycg_asset.uuid].pc_edit_proxy is not None:
            return False
        return context.object.pycg_asset.geometry_type == 'PC'

    def execute(self, context):
        cur_obj = context.object
        cur_data = AssetManager.data[cur_obj.pycg_asset.uuid]
        cur_xyz = cur_data.data['xyz']

        # Create proxy object and copy point cloud data.
        proxy_name = f'proxy_pc_{cur_obj.pycg_asset.uuid}'
        proxy_mesh = bpy.data.meshes.new(proxy_name)
        proxy_mesh.vertices.add(cur_xyz.shape[0])
        proxy_mesh.vertices.foreach_set('co', cur_xyz.ravel())
        proxy_mesh.vertex_layers_int.new(name="idx")
        proxy_mesh.vertex_layers_int["idx"].data.foreach_set('value', np.arange(cur_xyz.shape[0]))
        proxy_obj = bpy.data.objects.new(proxy_name, proxy_mesh)
        bpy.context.scene.collection.objects.link(proxy_obj)
        proxy_obj.parent = cur_obj
        proxy_obj.matrix_world = cur_obj.matrix_world.copy()
        proxy_obj.data.vertices.foreach_set('select', np.zeros(cur_xyz.shape[0], dtype=bool))
        cur_data.pc_edit_proxy = proxy_obj

        # ... and set to edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        proxy_obj.select_set(True)
        bpy.context.view_layer.objects.active = proxy_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.context.space_data.shading.type = 'WIREFRAME'
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT', )

        return {'FINISHED'}


class PYCG_OT_pc_edit_update(bpy.types.Operator):
    bl_idname = "pycg.pc_edit_update"
    bl_label = "PC Edit Update (Alt+C)"
    bl_description = "Update displayed cloud from edited mesh"

    @classmethod
    def poll(cls, context):
        if context.object is None or context.object.parent is None:
            return False
        p_obj = context.object.parent
        if p_obj.pycg_asset.uuid not in AssetManager.data.keys():
            return False
        return p_obj.pycg_asset.geometry_type == 'PC'

    def execute(self, context):
        cur_obj = context.object.parent
        cur_data = AssetManager.data[cur_obj.pycg_asset.uuid]

        # The mesh data can only be accessed and updated in object mode.
        bpy.ops.object.mode_set(mode='OBJECT')

        # Get current edited data
        proxy_obj = cur_data.pc_edit_proxy
        new_npoint = len(proxy_obj.data.vertices)
        new_xyz = np.empty(new_npoint * 3)
        proxy_obj.data.vertices.foreach_get('co', new_xyz)
        new_xyz = new_xyz.reshape((new_npoint, 3)).astype(np.float32)
        new_idx = np.empty(new_npoint, dtype=int)
        proxy_obj.data.vertex_layers_int["idx"].data.foreach_get('value', new_idx)

        # Update
        cur_data.data['xyz'] = new_xyz
        cur_data.data['rgb'] = cur_data.data['rgb'][new_idx]
        cur_data.data['normal'] = cur_data.data['normal'][new_idx]

        if context.scene.pycg_globals.labeling_update:
            selected_mask = np.empty(new_npoint, dtype=bool)
            proxy_obj.data.vertices.foreach_get('select', selected_mask)
            selected_color = context.tool_settings.image_paint.palette.colors.active.color
            # print(np.asarray(selected_color))
            cur_data.data['label'][selected_mask] = 1
            # Clear selection for convenience.
            proxy_obj.data.vertices.foreach_set('select', np.zeros(new_npoint, dtype=bool))

        # Trigger shader/batch update to reflect changes.
        cur_obj.pycg_asset.update_shader_type(None)

        # As we are now using new_xyz, the indices should be refreshed.
        proxy_obj.data.vertex_layers_int["idx"].data.foreach_set('value', np.arange(new_xyz.shape[0]))

        # Change back to edit mode.
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}


class PYCG_OT_pc_edit_end(bpy.types.Operator):
    bl_idname = "pycg.pc_edit_end"
    bl_label = "End"
    bl_description = "Update displayed cloud from edited mesh, stop edit mode and remove helper object"

    @classmethod
    def poll(cls, context):
        return PYCG_OT_pc_edit_update.poll(context)

    def execute(self, context):
        cur_obj = context.object.parent
        cur_data = AssetManager.data[cur_obj.pycg_asset.uuid]
        proxy_obj = cur_data.pc_edit_proxy

        # Delete proxy obj and associated mesh.
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.data.meshes.remove(proxy_obj.data, do_unlink=True)
        cur_data.pc_edit_proxy = None
        cur_obj.select_set(True)
        bpy.context.view_layer.objects.active = cur_obj

        return {'FINISHED'}


def register_assets():
    bpy.utils.register_class(AssetProperty)
    bpy.utils.register_class(GlobalProperty)
    bpy.utils.register_class(PYCG_PT_AssetPanel)

    bpy.utils.register_class(PYCG_OT_pc_edit_start)
    bpy.utils.register_class(PYCG_OT_pc_edit_update)
    bpy.data.window_managers[0].keyconfigs.active.keymaps['3D View'].keymap_items.new('pycg.pc_edit_update', value='PRESS', type='C',
                                                                                   alt=True)
    bpy.utils.register_class(PYCG_OT_pc_edit_end)

    AssetManager.handle = bpy.types.SpaceView3D.draw_handler_add(AssetManager.handler, (), 'WINDOW', 'POST_VIEW')

    # Annotation Palette
    annotation_palette = bpy.data.palettes.new("Annotation")
    for pi, pcolor in enumerate(ANNOTATE_PALETTE):
        mcolor = annotation_palette.colors.new()
        mcolor.color = (pcolor[0], pcolor[1], pcolor[2])
        if pi == 0:
            annotation_palette.colors.active = mcolor

    bpy.context.tool_settings.image_paint.palette = annotation_palette
