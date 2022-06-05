import numpy as np
from pycg import animation, render, vis, image
from pycg.animation import InterpType
from pycg.isometry import Isometry


if __name__ == '__main__':
    # Create scene
    scene = render.Scene(cam_path="assets/animate_rotating.bin")
    scene.up_axis = '+Y'

    # Our scene will be composed of many (4x5) colorful arrows
    for i in range(20):
        arrow_dir = np.random.random() * 2 * np.pi
        arrow_obj = vis.arrow(np.zeros(3),
                              np.array([np.cos(arrow_dir), 0.0, np.sin(arrow_dir)]),
                              color_id=i, cmap='metro')
        scene.add_object(arrow_obj, pose=Isometry(t=np.array([i % 4 * 2, 0.0, i // 4 * 2])),
                         name=f"arrow-{i}", attributes={"smooth_shading": False})

    # The quick_camera will automatically set camera_base to the center of scene.
    #   In most case this should be not sensible.
    scene.quick_camera(w=800, h=800, no_override=True)

    # Rotate the camera around the scene
    cambase_animator = animation.SpinPoseAnimator(
        InterpType.LINEAR, spin_axis=scene.up_axis, center=scene.camera_base.t)
    cambase_animator.set_keyframe(0, -0.2 * np.pi)
    cambase_animator.set_keyframe(100, 0.2 * np.pi)
    scene.animator.set_camera_base(cambase_animator, no_override=True)

    # Allow user to change camera whenever possible: simply add keyframes and the rotating effects will be overlayed.
    cam_animator = animation.FreePoseAnimator(InterpType.BEZIER)
    scene.animator.set_relative_camera(cam_animator, no_override=True)

    # Rotate also the arrows
    for i in range(20):
        arrow_animator = animation.SpinPoseAnimator(
            InterpType.LINEAR, spin_axis=scene.up_axis, center=scene.objects[f"arrow-{i}"].pose.t)
        arrow_animator.set_keyframe(0, 0.0)
        arrow_animator.set_keyframe(100, 5 * np.pi)
        scene.animator.set_object_pose(f"arrow-{i}", arrow_animator)

    # Use native open3d preview
    scene.preview(use_new_api=True)

    # Lastly, render out:
    for t, img in scene.render_opengl_animation():
        image.write(img, f'out/{t:04d}.png')

    # Export to blender to see if that aligns
    for t, img in scene.render_blender_animation(do_render=True):
        image.write(img, f"out/{t:04d}-blend.png")
