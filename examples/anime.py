from pycg import animation, render, vis, image
from pycg.animation import InterpType
from pycg.isometry import Isometry


if __name__ == '__main__':
    bunny_geom = vis.from_file("assets/bunny.obj")
    bunny_geom = vis.colored_mesh(bunny_geom, ucid=0)

    # Create a scene with a bunny inside
    scene = render.Scene()

    # Setting the up axis, so the default configurations could fit well
    scene.up_axis = '+Y'
    scene.add_object(bunny_geom, name='bunny')

    # Use the preset camera (if a camera path is already provided, you'll have to enable 'force_override')
    scene.quick_camera(w=800, h=800)

    # Preview
    # scene.preview()

    # Now we add a camera animation
    camera_animator = animation.FreePoseAnimator(InterpType.BEZIER)
    pose_1 = Isometry.copy(scene.camera_pose)
    pose_2 = Isometry.copy(pose_1)
    pose_2.t[0] -= 0.1
    camera_animator.set_keyframe(0, pose_1)
    camera_animator.set_keyframe(20, pose_2)
    scene.animator.set_relative_camera(camera_animator)

    # camera_animator.interp_x.visualize()

    # Use native open3d preview
    scene.animator.set_frame(19)
    scene.preview(with_animation=True)

    # Lastly, render out:
    # for t, img in scene.render_opengl_animation():
    #     image.write(img, f'out/{t:04d}.png')

    # Export to blender to see if that aligns
    for t, img in scene.render_blender_animation(do_render=False):
        image.write(img, f"out/{t:04d}-blend.png")
