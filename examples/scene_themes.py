from pycg.isometry import Isometry
from pycg import vis, render, image, color


def demo_render_bunny():
    scene = render.Scene(up_axis='+Y').add_object(bunny_y)
    scene.quick_camera(w=600, h=600, plane_angle=280.0)
    scene.preview(use_new_api=True)

    # Render using NKSR style
    render.ThemeNKSR(need_plane=True).apply_to(scene)
    nksr_rendering = scene.render_blender()
    nksr_rendering = image.alpha_compositing(
        image.gamma_transform(nksr_rendering, alpha_only=True, gamma=3.0),
        image.solid(nksr_rendering.shape[1], nksr_rendering.shape[0]))

    # Render using Diffusing style
    render.ThemeDiffuseShadow(
        base_color=color.map_quantized_color(0), sun_tilt_right=20.0, sun_tilt_back=30.0).apply_to(scene)
    ds_rendering = scene.render_blender()
    ds_rendering = image.alpha_compositing(
        ds_rendering, image.solid(ds_rendering.shape[1], ds_rendering.shape[0]))

    image.show(nksr_rendering, ds_rendering)


if __name__ == '__main__':
    # Load example meshes
    bunny_y = vis.from_file("assets/bunny.obj")
    chair_y = vis.from_file("assets/chair.ply")
    chair_z = Isometry.from_axis_angle('+X', 90.0) @ chair_y
    demo_render_bunny()

    # Render a room mesh using Angela Style
    room_y = vis.colored_mesh(vis.from_file("assets/room.ply"), ucid=0)
    #   This show an alternative way of creating a scene.
    room_scene = vis.show_3d([room_y], show=False).preview(use_new_api=True)
    render.ThemeAngela().apply_to(room_scene)

    angela_rendering = room_scene.render_blender()
    angela_rendering = image.alpha_compositing(
        angela_rendering, image.solid(angela_rendering.shape[1], angela_rendering.shape[0]))
    image.show(angela_rendering)
