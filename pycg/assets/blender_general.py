# Script executed by blender.

import argparse
import glob
import numpy as np
import shutil
import bpy
import sys
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ws', type=str, help='path to the workspace.')
parser.add_argument('--quality', type=int, help='Cycles sampling count')
parser.add_argument('--save', type=str, default="", help='blender project save path. if this is empty, will do rendering.')

argv = sys.argv[sys.argv.index("--") + 1:]  # get all args after "--"
args = parser.parse_args(argv)

base_workspace = Path(args.ws)
artifact_path = base_workspace / "artifacts"
output_path = base_workspace / "output"
output_path.mkdir(parents=True, exist_ok=True)

with (artifact_path / "scene.json").open() as f:
    scene_def = json.load(f)

# Delete Cube
bpy.ops.object.delete({"selected_objects": [bpy.data.objects["Cube"], bpy.data.objects["Light"]]})

# Switch to cycles and sRGB color management
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.view_settings.view_transform = 'Standard'

# Change light setting.
for light_id, light_def in enumerate(scene_def['lights']):
    # light_def's format is ('SUN', kwargs)
    light_name = f"Light-{light_id}"
    light_data = bpy.data.lights.new(name=light_name, type=light_def[0])
    light_data.energy = light_def[1]['energy']

    if light_def[0] == 'SUN':
        light_data.angle = light_def[1]['angle']
    elif light_def[0] == 'POINT':
        light_data.shadow_soft_size = light_def[1]['radius']
    elif light_def[0] == 'AREA':
        light_data.size = light_def[1]['size']

    light_object = bpy.data.objects.new(name=light_name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    # bpy.context.view_layer.objects.active = light_object
    light_object.location = light_def[1]['pos']
    light_object.rotation_mode = 'QUATERNION'
    light_object.rotation_quaternion = light_def[1]['rot']

# Ambient light
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = \
    (scene_def['ambient'][0], scene_def['ambient'][1], scene_def['ambient'][2], 1)
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = scene_def['ambient'][3]

# Add a new plane and set to shadow catcher (Now added via script)
# bpy.ops.mesh.primitive_plane_add(size=8, enter_editmode=False, location=(0.0, 0.0, 0.0))
# plane_obj = bpy.data.objects["Plane"]
# plane_obj.cycles.is_shadow_catcher = True
# New plane lies in X-Y plane, optionally rotate it to be X-Z plane
bpy.ops.object.select_all(action='DESELECT')

# Set the film to transparent.
bpy.context.scene.render.film_transparent = scene_def['film_transparent']

blender_camera = bpy.data.objects['Camera']
# Load in objects.
for mi, (mesh_name, mesh_attributes) in enumerate(scene_def["objects"]):
    # Import object
    bpy.ops.import_mesh.ply(filepath=str(artifact_path / mesh_name))
    # Current selection has plane and mesh
    obj_object = bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action='DESELECT')

    # Set pass id for index rendering
    obj_object.pass_index = mi

    # Setup material using nodes
    # https://blender.stackexchange.com/questions/23436/control-cycles-material-nodes-and-material-properties-in-python
    # https://blender.stackexchange.com/questions/23433/how-to-assign-a-new-material-to-an-object-in-the-scene-from-python
    mat = bpy.data.materials.new(name="Material-" + mesh_name)  # If exist, should use get method
    mat.use_nodes = True
    # Create nodes
    # It is also possible to create a node group.
    nodes = mat.node_tree.nodes
    nodes.clear()
    node_attribute = nodes.new(type='ShaderNodeAttribute')
    node_attribute.attribute_name = "Col"       # Use stored Ply color
    node_main_shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    # Main material attribute
    node_main_shader.inputs[4].default_value = mesh_attributes.get("material.metallic", 0.0)
    node_main_shader.inputs[5].default_value = mesh_attributes.get("material.specular", 0.0)
    node_main_shader.inputs[7].default_value = mesh_attributes.get("material.roughness", 0.5)
    # Link nodes
    links = mat.node_tree.links
    link0 = links.new(node_attribute.outputs[0], node_main_shader.inputs[0])
    link1 = links.new(node_main_shader.outputs[0], node_output.inputs[0])
    obj_object.data.materials.append(mat)

    # Alpha rendering by mixing a transparent shader.
    if mesh_attributes.get("alpha", 1.0) < 0.99:
        node_transparent_shader = nodes.new(type='ShaderNodeBsdfTransparent')
        node_mix_shader = nodes.new(type='ShaderNodeMixShader')
        node_mix_shader.inputs[0].default_value = mesh_attributes["alpha"]
        links.new(node_transparent_shader.outputs[0], node_mix_shader.inputs[1])
        links.new(node_main_shader.outputs[0], node_mix_shader.inputs[2])
        links.new(node_mix_shader.outputs[0], node_output.inputs[0])

    # Visibility layers.
    if not mesh_attributes.get("cycles_visibility.camera", True):
        obj_object.cycles_visibility.camera = False
    if not mesh_attributes.get("cycles_visibility.shadow", True):
        obj_object.cycles_visibility.shadow = False
    if not mesh_attributes.get("cycles_visibility.diffuse", True):
        obj_object.cycles_visibility.diffuse = False

    # Other Attributes.
    obj_object.cycles.is_shadow_catcher = mesh_attributes.get("cycles.is_shadow_catcher", False)
    if mesh_attributes.get("smooth_shading", False):
        for p in obj_object.data.polygons:
            p.use_smooth = True


# Setup camera.
cam_ext = np.asarray(scene_def['camera_pose'])
cam_intr = np.asarray(scene_def['camera_intrinsic'])
blender_camera.location = cam_ext[0:3]
blender_camera.rotation_mode = 'QUATERNION'
blender_camera.rotation_quaternion = cam_ext[3:7]
bpy.data.scenes['Scene'].render.resolution_x = cam_intr[0]
bpy.data.scenes['Scene'].render.resolution_y = cam_intr[1]
bpy.data.cameras['Camera'].shift_x = cam_intr[2]
bpy.data.cameras['Camera'].shift_y = cam_intr[3]
bpy.data.cameras['Camera'].angle = cam_intr[4]

# Render Quality
bpy.context.scene.cycles.samples = args.quality

if not args.save:
    # Start render.
    bpy.context.scene.render.filepath = str(output_path / "rgb.png")
    bpy.ops.render.render(write_still=True)
else:
    # For debug purpose, write blend file.
    bpy.ops.wm.save_as_mainfile(filepath=args.save)
