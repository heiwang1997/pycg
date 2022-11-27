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
from assets import AssetManager

class EnvironmentProperty(bpy.types.PropertyGroup):
    pass


class PYCG_PT_EnvironmentPanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = "pycg Environment"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        cur_obj = context.object
        sub = self.layout.column()
        sub.operator('pycg.host_notify')


def set_origin_material(uuidx):
    target_obj = AssetManager.data[uuidx].object
    mat_name = f"Material-{uuidx}-origin"
    mesh_attributes = AssetManager.data[uuidx].attributes

    old_mat = bpy.data.materials.get(mat_name)
    if old_mat:
        target_obj.active_material = old_mat
    else:
        mat = bpy.data.materials.new(name=mat_name)  # If exist, should use get method
        mat.use_nodes = True
        # Create nodes
        # It is also possible to create a node group.
        nodes = mat.node_tree.nodes
        nodes.clear()

        input_uv_node = nodes.new(type='ShaderNodeUVMap')
        links = mat.node_tree.links

        checker_attr = mesh_attributes.get("material.checker", {"on": False})
        normal_attr = mesh_attributes.get("material.normal", {"on": False})
        uniform_color_attr = mesh_attributes.get("uniform_color", None)

        if uniform_color_attr is not None:
            input_color_node = nodes.new(type='ShaderNodeRGB')
            input_color_node.outputs[0].default_value = uniform_color_attr
        elif checker_attr["on"]:
            input_color_node = nodes.new(type='ShaderNodeTexChecker')
            mat.node_tree.links.new(input_uv_node.outputs[0], input_color_node.inputs[0])
            input_color_node.inputs[1].default_value = checker_attr.get("color_a", (0.8, 0.29, 0.14, 1.0))
            input_color_node.inputs[2].default_value = checker_attr.get("color_b", (1.0, 0.808, 0.416, 1.0))
            input_color_node.inputs[3].default_value = checker_attr.get("scale", 5.0)
        elif normal_attr["on"]:
            input_normal_node = nodes.new(type='ShaderNodeNormalMap')
            normal_t_node1 = nodes.new(type='ShaderNodeVectorMath')
            normal_t_node1.operation = 'ADD'
            normal_t_node1.inputs[1].default_value = (1.0, 1.0, 1.0)
            input_color_node = nodes.new(type='ShaderNodeVectorMath')
            input_color_node.operation = 'MULTIPLY'
            input_color_node.inputs[1].default_value = (0.5, 0.5, 0.5)
            mat.node_tree.links.new(input_normal_node.outputs[0], normal_t_node1.inputs[0])
            mat.node_tree.links.new(normal_t_node1.outputs[0], input_color_node.inputs[0])
        elif len(AssetManager.data[uuidx].textures) > 0:
            input_color_node = nodes.new(type='ShaderNodeTexImage')
            mat.node_tree.links.new(input_uv_node.outputs[0], input_color_node.inputs[0])
            input_color_node.image = AssetManager.data[uuidx].textures[0]
        else:
            input_color_node = nodes.new(type='ShaderNodeAttribute')
            input_color_node.attribute_name = "Col"

        # Adding AO on top of input_color_node to increase contrast
        ao_attr = mesh_attributes.get("material.ao", {"on": False})
        if ao_attr["on"]:
            ao_node = nodes.new(type='ShaderNodeAmbientOcclusion')
            ao_node.inputs[1].default_value = 10.0  # Distance
            gamma_node = nodes.new(type='ShaderNodeGamma')
            gamma_node.inputs[1].default_value = ao_attr.get("gamma", 0.0)
            mix_rgb_node = nodes.new(type='ShaderNodeMixRGB')
            mix_rgb_node.blend_type = 'MULTIPLY'
            mix_rgb_node.inputs[0].default_value = ao_attr.get("strength", 0.5)
            links.new(input_color_node.outputs[0], ao_node.inputs[0])
            links.new(ao_node.outputs[0], mix_rgb_node.inputs[1])
            links.new(ao_node.outputs[1], gamma_node.inputs[0])
            links.new(gamma_node.outputs[0], mix_rgb_node.inputs[2])
            input_color_node = mix_rgb_node

        main_shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        # Main material attribute
        main_shader_node.inputs[4].default_value = mesh_attributes.get("material.metallic", 0.0)
        main_shader_node.inputs[5].default_value = mesh_attributes.get("material.specular", 0.0)
        main_shader_node.inputs[7].default_value = mesh_attributes.get("material.roughness", 0.5)
        # Link nodes
        links.new(input_color_node.outputs[0], main_shader_node.inputs[0])
        links.new(main_shader_node.outputs[0], output_node.inputs[0])
        target_obj.data.materials.append(mat)
        target_obj.active_material = mat

        # Alpha rendering by mixing a transparent shader.
        # if mesh_attributes.get("alpha", 1.0) < 0.99:
            # transparent_shader_node = nodes.new(type='ShaderNodeBsdfTransparent')
            # node_mix_shader = nodes.new(type='ShaderNodeMixShader')
            # node_mix_shader.inputs[0].default_value = mesh_attributes["alpha"]
            # links.new(transparent_shader_node.outputs[0], node_mix_shader.inputs[1])
            # links.new(main_shader_node.outputs[0], node_mix_shader.inputs[2])
            # links.new(node_mix_shader.outputs[0], output_node.inputs[0])
        main_shader_node.inputs[18].default_value = mesh_attributes.get("alpha", 1.0)

    # Visibility layers.
    if not mesh_attributes.get("cycles_visibility.camera", True):
        target_obj.cycles_visibility.camera = False
    if not mesh_attributes.get("cycles_visibility.shadow", True):
        target_obj.cycles_visibility.shadow = False
    if not mesh_attributes.get("cycles_visibility.diffuse", True):
        target_obj.cycles_visibility.diffuse = False

    # Other Attributes.
    # target_obj.cycles.is_shadow_catcher = mesh_attributes.get("cycles.is_shadow_catcher", False)
    target_obj.is_shadow_catcher = mesh_attributes.get("cycles.is_shadow_catcher", False)
    if mesh_attributes.get("smooth_shading", False):
        for p in target_obj.data.polygons:
            p.use_smooth = True

    # If render wireframe. Create the new object (but share the same mesh data)
    wireframe_attr = mesh_attributes.get("material.wireframe", {"on": False})
    if wireframe_attr["on"]:
        # TODO: delete this object when switching materials or deleting the parent.
        wireframe_obj_name = f"{uuidx}-wireframe"
        wireframe_mat_name = f"Material-{uuidx}-origin-wireframe"
        wireframe_obj = bpy.data.objects.get(wireframe_obj_name)
        if not wireframe_obj:
            wireframe_obj = target_obj.copy()
            wireframe_obj.name = wireframe_obj_name
            bpy.context.scene.collection.objects.link(wireframe_obj)
            wireframe_obj.parent = target_obj
            wireframe_modifier = wireframe_obj.modifiers.new("Wireframe", "WIREFRAME")
            wireframe_modifier.thickness = wireframe_attr.get("thickness", 0.01)
            wireframe_modifier.material_offset = 1
            wireframe_modifier.use_even_offset = False
        old_mat = bpy.data.materials.get(wireframe_mat_name)
        if not old_mat:
            mat = bpy.data.materials.new(name=wireframe_mat_name)
            mat.diffuse_color = (0, 0, 0, 1)
            target_obj.data.materials.append(mat)       # Materials are defined on meshes.


def register_style():
    bpy.utils.register_class(PYCG_PT_EnvironmentPanel)
