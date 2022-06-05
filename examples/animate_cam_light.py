from pycg import animation, render, vis, image
from pycg.animation import InterpType
from pycg.isometry import Isometry


if __name__ == '__main__':
    bunny_geom = vis.from_file("assets/bunny.obj")
    bunny_geom = vis.colored_mesh(bunny_geom, ucid=0)

    # Create a scene with a bunny inside
    scene = render.Scene()
    scene.viewport_shading = 'LIT'

    # Setting the up axis, so the default configurations could fit well
    scene.up_axis = '+Y'
    scene.add_object(bunny_geom, name='bunny', attributes={"smooth_shading": False})
    scene.add_object(vis.wireframe(bunny_geom), name='bunny-wireframe')

    # Use the preset camera
    scene.quick_camera(w=800, h=800, no_override=True)

    # Now we add a camera animation
    camera_animator = animation.FreePoseAnimator(InterpType.BEZIER)
    pose_1 = Isometry.copy(scene.relative_camera_pose)
    pose_2 = Isometry.copy(pose_1)
    pose_2.t[0] -= 0.1
    camera_animator.set_keyframe(0, pose_1)
    camera_animator.set_keyframe(100, pose_2)
    scene.animator.set_relative_camera(camera_animator)
    # (optional) if not set, this will automatically clamp to the keyframes.
    scene.animator.set_range(0, 200)

    # By default, there is no sun-light in the scene -- only ambient light (indirect light exists)
    #   To render shadows, we can add a sun (and also animate it using the animation system!)
    scene.add_light_sun('sun')
    sun_animator = animation.FreePoseAnimator(InterpType.LINEAR)
    scene.animator.set_sun_pose(sun_animator)

    # Use native open3d preview
    scene.animator.set_frame(100)
    scene.preview(use_new_api=True)

    # Lastly, render out:
    for t, img in scene.render_opengl_animation():
        image.write(img, f'out/{t:04d}.png')

    # Export to blender to see if that aligns
    for t, img in scene.render_blender_animation(do_render=True):
        image.write(img, f"out/{t:04d}-blend.png")
