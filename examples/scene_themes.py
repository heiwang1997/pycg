from pycg.isometry import Isometry
from pycg import vis, render, image
import open3d as o3d


if __name__ == '__main__':
    # Load in the mesh and crop the room
    example_mesh = vis.from_file("/home/huangjh/nas/ours/matterport-test/vis50k/mesh-trimmed/000046.ply")
    max_bound = example_mesh.get_max_bound()
    max_bound[1] -= 0.1
    bound = o3d.geometry.AxisAlignedBoundingBox(example_mesh.get_min_bound(), max_bound)
    example_mesh = example_mesh.crop(bound)

    # Build a scene
    scene = render.Scene()
    scene.up_axis = '+Y'
    scene.add_object(example_mesh, name='room')
    scene.quick_camera(w=800, h=800)

    # Specify theme
    theme = render.IndoorRoomTheme()
    theme.apply_to(scene)

    # This allows user to change sun direction: Press L to confirm.
    scene.preview(use_new_api=True)

    image.write(scene.render_blender(), "out/theme_room.png")
