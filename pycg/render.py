"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
import copy
import functools
import os
import pdb
import pickle
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pyquaternion import Quaternion

import pycg.blender_client as blender
import pycg.image as image
import pycg.o3d as o3d
import pycg.vis as vis
from pycg import get_assets_path
from pycg.camera import CameraIntrinsic
from pycg.exp import logger
from pycg.interfaces import Renderable

from .animation import SceneAnimator
from .isometry import Isometry, ScaledIsometry

gui = o3d.visualization.gui

class GLEngineWrapper:
    def __init__(self, engine):
        self.engine = engine
        self.displayed_geometries = {}


class VisualizerNewAPIClosures:
    """
    We create a separate scope here to store the data needed by the closure (i.e. <locals>)
    So we do not need strange partials any more.
    Note that changing arguments outside (such as all_windows) ...
        still have impact on closure behaviours because args are anyway references.
    https://stackoverflow.com/questions/14413946/what-exactly-is-contained-within-a-obj-closure
    """
    @staticmethod
    def keyboard_closures(cur_window, cur_scene, cur_pose_change_callback, cur_hists, all_windows):

        temp_cam_keys = [gui.KeyName.F1, gui.KeyName.F2, gui.KeyName.F3, gui.KeyName.F4, gui.KeyName.F5,
                         gui.KeyName.F6, gui.KeyName.F7, gui.KeyName.F8, gui.KeyName.F9]
        wireframe_on = False

        def on_key(e):
            if e.type == gui.KeyEvent.DOWN:
                if e.key == gui.KeyName.COMMA:
                    cur_window.show_message_box(
                        "Help",
                        "Mouse Control (Arcball Mode):\n"
                        " + Left button: rotate\n"
                        " + Double left click: set rotation center to pointed position\n"
                        " + Shift + Left: high-precision dolly.\n"
                        " + (Ctrl + Shift)/(Meta) + Left: in-screen-plane rotation (RotateZ).\n"
                        " + Ctrl + Left: Pan\n"
                        " + Right button: Pan\n"
                        " + Wheel: low-precision dolly.\n"
                        " + Shift + Wheel: high-precision fov.\n"
                        "Keyboard Shortcuts:\n"
                        " + e(X)it: raise a KeyboardInterrupt exception to end the program directly.\n"
                        " + (S)etting: show settings sidebar.\n"
                        " + (A)xes: show the axes.\n"
                        " + (H)istogram: show histograms if available.\n"
                        " + (R)ecord: record the current camera poses (and its animation if existing).\n"
                        " + (L)ight: record the sun light direction (but not saved to file).\n"
                        " + (O)bject: record the object pose (but not saved to file)."
                        " + (B)ackface: control whether to cull backface\n"
                        " + (W)ireframe: Turn on/off the wireframe mode\n"
                        " + anno(T)ation: save annotation to scene\n"
                        " + F1-F12: Jump to temporary camera locations. Ctrl modifier to set.\n"
                        " + +/-: increase/decrease the size of point cloud.\n"
                        " + [/]: increase/decrease the fov.")
                elif e.key == gui.KeyName.ONE:
                    print("View Control set to rotate camera")
                    cur_window.mouse_mode = gui.SceneWidget.Controls.ROTATE_CAMERA
                elif e.key == gui.KeyName.TWO:
                    print("View Control set to pick geometry... Use control+click to pick.")
                    cur_window.mouse_mode = gui.SceneWidget.Controls.PICK_GEOMETRY
                elif e.key == gui.KeyName.THREE:
                    print("View Control set to rotate model")
                    cur_window.mouse_mode = gui.SceneWidget.Controls.ROTATE_MODEL
                elif e.key == gui.KeyName.X:
                    raise KeyboardInterrupt
                elif e.key == gui.KeyName.Q:
                    for w_cur in all_windows:
                        w_cur.close()
                elif e.key == gui.KeyName.S:
                    cur_window.show_settings = not cur_window.show_settings
                elif e.key == gui.KeyName.A:
                    cur_window.show_axes = not cur_window.show_axes
                elif e.key == gui.KeyName.H:
                    for hst in cur_hists:
                        hst.visible = not hst.visible
                elif e.key == gui.KeyName.W:
                    nonlocal wireframe_on
                    wireframe_on = not wireframe_on
                    cur_window.enable_wireframe_mode(wireframe_on)
                elif e.key == gui.KeyName.R:
                    if cur_pose_change_callback is not None:
                        # When output current pose, need to change back this.
                        # if scale != 1.0:
                        #     cur_scene.camera_intrinsic = cur_scene.camera_intrinsic.scale(1.0 / scale)
                        if cur_scene.animator.is_enabled():
                            cam_animator = cur_scene.animator.get_relative_camera()
                            if cam_animator is not None and cam_animator.get_last_t() is not None:
                                # cur_scene.animator.set_current_frame()
                                # cur_window.scene_widget.force_redraw()
                                # cur_window.post_redraw()
                                cur_window.show_message_box(
                                    "Reminder",
                                    "Animation is saved. But the current camera can be only added via keyframes.")
                        cur_pose_change_callback(cur_window)
                        # if scale != 1.0:
                        #     cur_scene.camera_intrinsic = cur_scene.camera_intrinsic.scale(scale)
                elif e.key == gui.KeyName.L:
                    if "sun" in cur_scene.lights.keys():
                        if cur_scene.animator.is_enabled():
                            sun_animator = cur_scene.animator.get_sun_pose()
                            if sun_animator is not None and sun_animator.get_last_t() is not None:
                                cur_window.show_message_box(
                                    "Reminder",
                                    "Sun can be saved. But it seems that it is actually controlled by an animator."
                                )
                        print("Light information written to scene.")
                        sun_dir = cur_window.scene.scene.get_sun_light_direction()
                        cur_scene.lights["sun"].pose = Isometry.look_at(source=np.zeros((3,)),
                                                                        target=-sun_dir)
                elif e.key == gui.KeyName.O:
                    for obj_name, obj in cur_scene.objects.items():
                        if cur_window.scene.has_geometry(obj_name):
                            new_transform = cur_window.scene.get_geometry_transform(obj_name)
                            obj.pose = Isometry.from_matrix(new_transform)
                            print(f"Object {obj_name}'s pose written to scene")
                elif e.key == gui.KeyName.B:
                    print("Culling settings saved.")
                    cur_scene.backface_culling = not cur_scene.backface_culling
                    for mesh_name, mesh_obj in cur_scene.objects.items():
                        cur_window.scene.set_geometry_double_sided(mesh_name, not cur_scene.backface_culling)
                    cur_window.scene_widget.force_redraw()
                    cur_window.post_redraw()
                elif e.key == gui.KeyName.EQUALS:  # + point size
                    target_size = cur_window.point_size + 1
                    for w_cur in all_windows:
                        w_cur.point_size = target_size
                        w_cur.scene_widget.force_redraw()
                        w_cur.post_redraw()
                elif e.key == gui.KeyName.MINUS:  # - point size
                    target_size = cur_window.point_size - 1
                    for w_cur in all_windows:
                        w_cur.point_size = target_size
                        w_cur.scene_widget.force_redraw()
                        w_cur.post_redraw()
                elif e.key == gui.KeyName.LEFT_BRACKET:
                    pose, intr = VisualizerManager.get_window_camera_info(cur_window)
                    intr.fov_x += np.deg2rad(5.0)
                    cur_window.setup_camera(intr.to_open3d_intrinsic(), pose.inv().matrix)
                elif e.key == gui.KeyName.RIGHT_BRACKET:
                    pose, intr = VisualizerManager.get_window_camera_info(cur_window)
                    intr.fov_x -= np.deg2rad(5.0)
                    cur_window.setup_camera(intr.to_open3d_intrinsic(), pose.inv().matrix)
                elif e.key == gui.KeyName.T:
                    cur_scene.selection_sets = cur_window.get_selection_sets()
                    print("Annotation selection set saved.")
                elif e.key in temp_cam_keys:
                    f_idx = temp_cam_keys.index(e.key) + 1
                    cam_path = Path(f"/tmp/pycg-cam{f_idx}.bin")
                    if cur_window.get_mouse_mods() == 2:    # Control pressed
                        camera_pose, camera_intrinsic = VisualizerManager.get_window_camera_info(cur_window)
                        print(f"Temp Camera {f_idx} saved!")
                        with cam_path.open("wb") as f:
                            pickle.dump([camera_pose, camera_intrinsic], f)
                    else:
                        if cam_path.exists():
                            with cam_path.open("rb") as f:
                                camera_pose, camera_intrinsic = pickle.load(f)
                            print(f"Temp Camera {f_idx} loaded!")
                            cur_window.setup_camera(camera_intrinsic.to_open3d_intrinsic(), camera_pose.inv().matrix)
            return 0

        return on_key

    @staticmethod
    def mouse_closures(cur_window, all_windows):
        def on_mouse(e):
            if e.type == gui.MouseEvent.DRAG or e.type == gui.MouseEvent.WHEEL:
                cam_pose, cam_intrinsic = VisualizerManager.get_window_camera_info(cur_window)
                for w_others in all_windows:
                    if id(w_others) != id(cur_window):
                        w_others.setup_camera(cam_intrinsic.to_open3d_intrinsic(), cam_pose.inv().matrix)
                        w_others.scene_widget.force_redraw()
                        w_others.post_redraw()
            return 0
        return on_mouse

    @staticmethod
    def set_keyframer_closures(keyframer, cur_scene, cur_window):

        all_targets = []
        current_target = 0
        CAM_NAME, CAM_ATTR = "relative_camera", "pose"
        SUN_NAME, SUN_ATTR = "sun", "pose"

        from .animation import FreePoseAnimator

        for obj_uuid, obj_attribs in cur_scene.animator.events.items():
            for attrib_name, attrib_interp in obj_attribs.items():
                if obj_uuid == CAM_NAME and attrib_name == CAM_ATTR:
                    current_target = len(all_targets)
                all_targets.append([obj_uuid, attrib_name])

        # Update all target attributes.
        # Although we could only control relative_camera.pose, we could at least see the keyframes of other attributes...
        def update_all_targets():
            current_name, current_attr = all_targets[current_target]
            keyframer.set_available_targets([f"{o}.{a}" for (o, a) in all_targets], current_target)
            frame_start, frame_end = cur_scene.animator.get_range()
            keyframer.set_keyframes(frame_start, frame_end, o3d.utility.IntVector(
                cur_scene.animator.events[current_name][current_attr].ordered_times()
            ))
            keyframer.set_current_frame(cur_scene.animator.current_frame)

        update_all_targets()

        def on_frame_changed(new_t):
            cur_scene.animator.set_frame(new_t)
            cur_scene._update_filament_engine(cur_window)
            cur_window.scene_widget.force_redraw()
            cur_window.post_redraw()

        def on_keyframe_added(new_t):
            current_name, current_attr, = all_targets[current_target]
            animator = cur_scene.animator.events[current_name][current_attr]

            if current_name == CAM_NAME and current_attr == CAM_ATTR and isinstance(animator, FreePoseAnimator):
                camera_pose, _ = VisualizerManager.get_window_camera_info(cur_window)
                base_pose = cur_scene.camera_base
                new_value = base_pose.inv() @ camera_pose
            elif current_name == SUN_NAME and current_attr == SUN_ATTR and isinstance(animator, FreePoseAnimator):
                sun_dir = cur_window.scene.scene.get_sun_light_direction()
                new_value = Isometry.look_at(source=np.zeros((3,)), target=-sun_dir)
            else:
                new_value = animator.get_value(new_t, raw=True)

            animator.set_keyframe(new_t, new_value)
            update_all_targets()

        def on_keyframe_removed(new_t):
            current_name, current_attr, = all_targets[current_target]
            animator = cur_scene.animator.events[current_name][current_attr]
            animator.remove_keyframe(new_t)
            update_all_targets()

        def on_keyframe_moved(old_t, new_t):
            current_name, current_attr, = all_targets[current_target]
            animator = cur_scene.animator.events[current_name][current_attr]
            kf_value = animator.get_value(old_t, raw=True)
            animator.remove_keyframe(old_t)
            animator.set_keyframe(new_t, kf_value)
            cur_scene.animator.set_frame(new_t)
            cur_scene._update_filament_engine(cur_window)
            update_all_targets()

        def on_play_status_changed(is_play):
            cur_window.is_animating = is_play

        def on_target_changed(new_target):
            nonlocal current_target
            current_target = new_target
            update_all_targets()

        keyframer.set_on_frame_changed(on_frame_changed)
        keyframer.set_on_keyframe_added(on_keyframe_added)
        keyframer.set_on_keyframe_removed(on_keyframe_removed)
        keyframer.set_on_keyframe_moved(on_keyframe_moved)
        keyframer.set_on_play_status_changed(on_play_status_changed)
        keyframer.set_on_target_changed(on_target_changed)

    @staticmethod
    def set_animation_closures(keyframer, cur_scene, cur_window):
        has_renderable = any([obj.renderable is not None for obj in cur_scene.objects.values()])

        def on_animation(e, t):

            if keyframer is not None:
                cur_scene.animator.set_frame(cur_scene.animator.current_frame + 1)
                cur_scene._update_filament_engine(e)
                keyframer.set_current_frame(cur_scene.animator.current_frame)

            if has_renderable:
                camera_pose, camera_intrinsic = VisualizerManager.get_window_camera_info(e)

                canvas = None
                for obj in cur_scene.objects.values():
                    if obj.renderable is not None:
                        assert canvas is None, "Only one renderable object is allowed."
                        canvas = obj.renderable.render(camera_pose, camera_intrinsic)

                    if canvas is not None:
                        canvas_data = o3d.geometry.Image(canvas)
                        cur_window.set_background((1.0, 1.0, 1.0, 1.0), canvas_data)

        if keyframer is not None or has_renderable:
            cur_window.is_animating = True
            cur_window.animation_frame_delay = 0.05     # seconds
            cur_window.set_on_animation_frame(on_animation)

class VisualizerManager:
    """
    This class is used to manage the rendering of scenes.

    use_new_api: 
        True: use o3d.visualization.O3DVisualizer powered by filament.
        False: use Open3D's original OpenGL-based visualization
    """
    TITLE_HEIGHT_FIX = 24

    def __init__(self):
        self.scenes = []
        self.scene_titles = []
        self.pose_change_callbacks = []
        self.all_engines = []
        self.use_new_api = None
        self.reset()

    def reset(self):
        self.scenes = []
        self.scene_titles = []
        self.pose_change_callbacks = []
        self.all_engines = []
        self.use_new_api = None

    def add_scene(self, scene, title=None, pose_change_callback=None):
        self.scenes.append(scene)
        self.scene_titles.append(title)
        self.pose_change_callbacks.append(pose_change_callback)

    @staticmethod
    def get_window_camera_info(window):
        pos = window.scene.camera.get_position()
        forward = window.scene.camera.get_forward_vector()
        up = window.scene.camera.get_up_vector()

        projection = window.scene.camera.get_projection_matrix()
        width = window.content_rect.width
        height = window.content_rect.height

        fx = projection[0, 0] * width / 2.
        fy = projection[1, 1] * height / 2.
        cx = (1.0 - projection[0, 2]) * width / 2.
        cy = (1.0 + projection[1, 2]) * height / 2.

        return Isometry.look_at(pos, pos + forward, up), CameraIntrinsic(width, height, fx, fy, cx, cy)

    @staticmethod
    def convert_histogram(data):
        from scipy import stats
        r_min = np.nanmin(data)
        r_max = np.nanmax(data)
        pos = np.linspace(r_min, r_max, 200)

        kernel = stats.gaussian_kde(data[np.isfinite(data)])
        return r_min, r_max, kernel(pos)

    def build_engines(self, use_new_api=False, key_bindings=None, max_cols=-1):
        self.use_new_api = use_new_api

        assert len(self.scenes) > 0, "No scene to show."
        if max_cols == -1:
            max_cols = len(self.scenes)

        if use_new_api:
            # Please refer to Open3D/python/open3d/visualization/draw.py
            gui.Application.instance.initialize(str(o3d.get_resource_path()))

            all_windows = []
            for scene_idx, (cur_scene, cur_title) in enumerate(zip(self.scenes, self.scene_titles)):
                if cur_title is None:
                    cur_title = f"Scene-{scene_idx:03d}"

                c_param = o3d.visualization.O3DVisualizer.ConstructParams()
                c_param.set_scene_widget(gui.SceneWidget())
                other_widgets = []

                # Analyse and add histogram window if necessary.
                histogram_widgets = []
                for mesh_name, mesh_obj in cur_scene.objects.items():
                    if mesh_obj.annotations is None:
                        continue
                    if isinstance(mesh_obj.geom, o3d.geometry.PointCloud):
                        h_min, h_max, h_values = self.convert_histogram(mesh_obj.annotations[0])
                        hist_widget = gui.Histogram(20, 50 + len(histogram_widgets) * 100, 400, 100,
                                                    f"PC-{mesh_name[:6]}")
                        HISTOGRAM_COLORMAP = {
                            'viridis': gui.Color.Colormap.VIRIDIS, 'plasma': gui.Color.Colormap.PLASMA,
                            'jet': gui.Color.Colormap.JET, 'spectral': gui.Color.Colormap.SPECTRAL
                        }
                        hist_widget.set_value(h_min, h_max, h_values, HISTOGRAM_COLORMAP[mesh_obj.annotations[1]])
                        histogram_widgets.append(hist_widget)
                other_widgets += histogram_widgets

                # Add keyframer if necessary.
                if cur_scene.animator.is_enabled():
                    keyframer = gui.Keyframer(
                        (cur_scene.camera_intrinsic.w - 600) // 2, int(cur_scene.camera_intrinsic.h * 0.8),
                        600, 100, "Keyframer (Top row: time, Bottom row: keyframe mover, Press Tab to enter number)")
                    other_widgets.append(keyframer)
                else:
                    keyframer = None

                c_param.other_widgets = other_widgets
                w = o3d.visualization.O3DVisualizer(title=cur_title, width=cur_scene.camera_intrinsic.w,
                                                    height=cur_scene.camera_intrinsic.h + self.TITLE_HEIGHT_FIX,
                                                    param=c_param)
                all_windows.append(w)
                cur_scene._build_filament_engine(w)

                # -- Define and set callbacks --

                # Set animation and edits callback
                if keyframer is not None:
                    VisualizerNewAPIClosures.set_keyframer_closures(keyframer, cur_scene, w)

                # Set refreshing callback
                VisualizerNewAPIClosures.set_animation_closures(keyframer, cur_scene, w)

                # You can also do w.scene_widget.set_on_key, but on panels it won't respond.
                w.set_on_key(VisualizerNewAPIClosures.keyboard_closures(
                    w, cur_scene, self.pose_change_callbacks[scene_idx], histogram_widgets, all_windows))

                if len(self.scenes) > 1:
                    # Sync camera.
                    w.scene_widget.set_on_mouse(VisualizerNewAPIClosures.mouse_closures(w, all_windows))

                gui.Application.instance.add_window(w)

                if len(self.scenes) > 1:
                    w.os_frame = gui.Rect(w.os_frame.width * scene_idx, 0, w.os_frame.width, w.os_frame.height)
            self.all_engines = all_windows

        else:
            all_engines = []

            axis_object = vis.frame()
            axis_shown = False

            for scene_idx, (cur_scene, cur_title) in enumerate(zip(self.scenes, self.scene_titles)):
                if cur_title is None:
                    cur_title = f"Scene-{scene_idx:03d}"

                pos_x = cur_scene.camera_intrinsic.w * (scene_idx % max_cols)
                pos_y = 50 + (cur_scene.camera_intrinsic.h + 50) * (scene_idx // max_cols)

                gl_engine = cur_scene._build_gl_engine(cur_title, True, pos=(pos_x, pos_y))
                engine = gl_engine.engine

                def interrupt(vis):
                    raise KeyboardInterrupt

                engine.register_key_callback(key=ord("X"), callback_func=interrupt)

                def toggle_axis(vis):
                    nonlocal axis_shown
                    for eng in all_engines:
                        if not axis_shown:
                            eng.add_geometry(axis_object, reset_bounding_box=False)
                        else:
                            eng.remove_geometry(axis_object, reset_bounding_box=False)
                    axis_shown = not axis_shown

                engine.register_key_callback(key=ord("A"), callback_func=toggle_axis)

                if key_bindings is not None:
                    for _key, _func in key_bindings.items():
                        engine.register_key_callback(key=ord(_key.upper()), callback_func=_func)

                if self.pose_change_callbacks[scene_idx] is not None:
                    new_callback = self.pose_change_callbacks[scene_idx]
                    engine.register_key_callback(key=ord("R"), callback_func=new_callback)

                # Temp camera locations (FX - Use, Ctrl+FX - Set)
                def temp_camera_callback(vis, key_action, mod_action, idx):
                    if key_action == 1:         # Press
                        cam_path = Path(f"/tmp/pycg-cam{idx}.bin")
                        if mod_action == 2:     # Control (Set)
                            cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                            camera_pose = Isometry.from_matrix(cam_param.extrinsic).inv()
                            camera_intrinsic = CameraIntrinsic.from_open3d_intrinsic(cam_param.intrinsic)
                            print(f"Temp Camera {idx} saved!")
                            with cam_path.open("wb") as f:
                                pickle.dump([camera_pose, camera_intrinsic], f)
                        else:
                            if cam_path.exists():
                                with cam_path.open("rb") as f:
                                    camera_pose, camera_intrinsic = pickle.load(f)
                                print(f"Temp Camera {idx} loaded!")
                                vis.get_view_control().convert_from_pinhole_camera_parameters(
                                    camera_intrinsic.get_pinhole_camera_param(camera_pose, fix_bug=True), allow_arbitrary=True
                                )
                for fx in range(10):
                    engine.register_key_action_callback(
                        key=289 + fx, callback_func=functools.partial(temp_camera_callback, idx=fx))
                        # key=ord("5") + fx, callback_func=functools.partial(temp_camera_callback, idx=fx))

                if len(self.scenes) > 1:
                    # Sync camera and lighting settings.
                    def on_view_refresh(vis):
                        cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                        cam_pose = cam_param.extrinsic
                        camera_pose = Isometry.from_matrix(cam_pose).inv()
                        camera_intrinsic = CameraIntrinsic.from_open3d_intrinsic(cam_param.intrinsic)
                        for eng in all_engines:
                            if eng == vis:
                                continue

                            eng.get_view_control().convert_from_pinhole_camera_parameters(
                                camera_intrinsic.get_pinhole_camera_param(camera_pose, fix_bug=True), allow_arbitrary=True
                            )
                            eng.get_render_option().point_size = vis.get_render_option().point_size
                            eng.get_render_option().mesh_show_back_face = vis.get_render_option().mesh_show_back_face
                            eng.get_render_option().mesh_show_wireframe = vis.get_render_option().mesh_show_wireframe
                            # self.gl_render_options.save_to_json("/tmp/ro.json")
                            # engine.get_render_option().load_from_json("/tmp/ro.json")

                    if o3d.is_custom_build:
                        engine.register_view_refresh_callback(on_view_refresh)
                all_engines.append(engine)
            self.all_engines = all_engines

    def run(self, use_new_api=False, key_bindings=None, max_cols=-1, scale=1.0):
        self.build_engines(use_new_api, key_bindings=key_bindings, max_cols=max_cols)
        if self.use_new_api:
            gui.Application.instance.run()
        else:
            if len(self.all_engines) < 2:
                # For one window, use the faster solution.
                self.all_engines[0].run()
            else:
                while True:
                    can_stop = False
                    for eng in self.all_engines:
                        if not eng.poll_events():
                            can_stop = True
                        eng.update_renderer()
                    if can_stop:
                        break
            for eng in self.all_engines:
                eng.destroy_window()
        self.reset()

    def run_step(self, update_from_scene: bool = False):
        if self.use_new_api is None or len(self.all_engines) == 0:
            print("You have to call build_engines first in order to run step by step.")
        elif self.use_new_api is True:
            if update_from_scene:
                for scene, w in zip(self.scenes, self.all_engines):
                    scene._update_filament_engine(w)
                    w.scene_widget.force_redraw()
                    w.post_redraw()
            can_stop = not gui.Application.instance.run_one_tick()
            if can_stop:
                self.reset()
        else:
            if update_from_scene:
                for scene, w in zip(self.scenes, self.all_engines):
                    scene._update_gl_engine(w)
            can_stop = False
            for eng in self.all_engines:
                if not eng.poll_events():
                    can_stop = True
                    break
                eng.update_renderer()
            if can_stop:
                for eng in self.all_engines:
                    eng.destroy_window()
                self.reset()

    def get_scene_engine(self, scene):
        for s, engine in zip(self.scenes, self.all_engines):
            if id(s) == id(scene):
                return engine
        return None


# Global manager.
vis_manager = VisualizerManager()


def render_scene_batches(scene_list, n_cols: int):
    rendered_imgs = []
    for cur_scene in scene_list:
        cur_img = cur_scene.render_opengl()
        rendered_imgs.append(cur_img)
    pic_w, pic_h = rendered_imgs[0].shape[1], rendered_imgs[0].shape[0]

    row_imgs, col_imgs = [], []
    for pic_id, pic_img in enumerate(rendered_imgs):
        col_imgs.append(pic_img)
        if (pic_id + 1) % n_cols == 0:
            row_img = image.hlayout_images(col_imgs, [pic_w] * n_cols)
            col_imgs = []
            row_imgs.append(row_img)
    final_img = image.vlayout_images(row_imgs, [pic_h] * len(row_imgs), )
    final_img = (final_img * 255).astype(np.uint8)

    return final_img


def sanitize_usd_prim_path(origin_path: str):
    path_comp = origin_path.split('/')
    new_path_comp = []
    for pc in path_comp:
        if len(pc) > 0 and not pc[0].isalpha():
            pc = 't' + pc
        pc = pc.replace('-', '_')
        new_path_comp.append(pc)
    new_path = '/'.join(new_path_comp)
    if new_path != origin_path:
        logger.warning(f"Invalid USD prim path: {origin_path}, sanitized to {new_path}!")
    return new_path


class SceneObject:
    """
    A class that represents a renderable object in a 3D scene. It wraps geometry with transformation and rendering attributes.

    Key components:
    - geom: The underlying geometry (e.g. mesh, point cloud) from Open3D
    - pose: Isometry transformation representing object's position and orientation 
    - scale: Uniform scale factor applied to the geometry
    - attributes: Dictionary of rendering attributes like materials, colors, etc.
    - annotations: Optional annotations associated with the geometry
    - renderable: Optional custom renderer for the object
    
    Attributes dictionary can include:
    - alpha: Transparency value (0.0-1.0)
    - material.specular: Specular reflection (0.0-1.0) 
    - material.metallic: Metallic material property (0.0-1.0)
    - material.roughness: Surface roughness (0.0-1.0)
    - material.checker: Checkerboard texture settings {"on": True/False, "color_a": (rgba), "color_b": (rgbd), "scale": 1.0}
    - material.normal: Normal shading settings {"on": True/False}
    - material.ao: Ambient occlusion settings {"on": True/False, "gamma": 0.0 (the larger the more contrast), "strength": 0.5 (0~1)}
    - cycles_visibility.camera: True/False
    - cycles_visibility.shadow: True/False
    - cycles_visibility.diffuse: True/False
    - cycles.is_shadow_catcher: True/False
    - smooth_shading: True/False
    - uniform_color: Override vertex colors with uniform RGBA color
    """

    USD_RAW_ATTR_NAME = "pycg_raw_attributes"

    def __init__(self, geom, pose: Isometry = Isometry(), scale: float = 1.0, attributes: dict = None):
        if attributes is None:
            attributes = {}
        if isinstance(geom, vis.AnnotatedGeometry):
            self.geom = geom.geom
            self.annotations = geom.annotations
            attributes.update(geom.attributes)
        else:
            self.geom, self.annotations = geom, None

        if isinstance(geom, Renderable):
            self.renderable = self.geom
            self.geom = self.geom.get_proxy_geometry()
        
        else:
            self.renderable = None

        self.pose = pose
        self.scale = scale
        self.visible = True
        self.attributes = attributes

    @property
    def scaled_iso(self):
        return ScaledIsometry.from_inner_form(self.pose, self.scale)

    def get_extent(self):
        # Note this is only a rough extent!
        if 'text_pos' in self.attributes:
            ps = np.asarray(self.attributes['text_pos'])
            return ps, ps + 0.01
            
        bound_points = self.geom.get_axis_aligned_bounding_box().get_box_points()
        if hasattr(bound_points, "numpy"):
            bound_points = bound_points.numpy()
        else:
            bound_points = np.asarray(bound_points)
        bound_points = self.scaled_iso @ bound_points
        return np.min(bound_points, axis=0), np.max(bound_points, axis=0)

    def get_transformed(self):
        """Get a transformed copy of the geometry using the stored pose and scale.
        """
        # Apply scaled isometry transformation to geometry
        return self.scaled_iso @ self.geom

    def get_filament_material(self, scene_shader: str, scene_point_size: float) -> o3d.visualization.rendering.MaterialRecord:
        """Get a material record for rendering with Open3D's Filament renderer.

        This function creates and configures a MaterialRecord based on the geometry type and attributes.
        It handles different shader types, transparency, material properties like metallic/roughness,
        and texture mapping.

        Args:
            scene_shader: str
                The shader type to use. Must be one of: 'LIT', 'UNLIT', or 'NORMAL'
            scene_point_size: float
                Size of points when rendering point clouds

        Returns:
            o3d.visualization.rendering.MaterialRecord:
                Configured material record for use with Filament renderer
        """
        # Create base material record
        mat = o3d.visualization.rendering.MaterialRecord()
        
        # Map shader type to Filament shader name
        mat.shader = {
            'LIT': "defaultLit", 'UNLIT': "defaultUnlit", 'NORMAL': 'normals'
        }[scene_shader]
        mat.point_size = scene_point_size
        mat.line_width = 1

        # Use line shader for LineSet geometries
        if isinstance(self.geom, o3d.geometry.LineSet):
            mat.shader = 'unlitLine'

        # override scene_shader to use transparency
        if "alpha" in self.attributes:
            mat.base_color = (1.0, 1.0, 1.0, self.attributes["alpha"])
            if mat.shader == 'defaultLit':
                mat.shader = "defaultLitTransparency"
            elif mat.shader == 'defaultUnlit':
                mat.shader = "defaultUnlitTransparency"

        # Configure material properties from attributes
        if "material.specular" in self.attributes:
            mat.base_reflectance = self.attributes["material.specular"]
        else:
            mat.base_reflectance = 0.0

        if "material.metallic" in self.attributes:
            mat.base_metallic = self.attributes["material.metallic"]
        else:
            mat.base_metallic = 0.0

        if "material.roughness" in self.attributes:
            mat.base_roughness = self.attributes["material.roughness"]
        else:
            mat.base_roughness = 1.0

        # override scene_shader to use normal shading
        if "material.normal" in self.attributes:
            if self.attributes["material.normal"]["on"]:
                mat.shader = 'normals'

        # if smooth shading is enabled, use vertex normals, otherwise use triangle normals
        if "smooth_shading" in self.attributes and isinstance(self.geom, o3d.geometry.TriangleMesh):
            if self.attributes["smooth_shading"]:
                if not self.geom.has_vertex_normals():
                    self.geom.compute_vertex_normals()
            else:
                if self.geom.has_vertex_normals():
                    self.geom.vertex_normals.clear()
                if not self.geom.has_triangle_normals():
                    self.geom.compute_triangle_normals()

        # Set base color to uniform color if specified
        if "uniform_color" in self.attributes:
            mat.base_color = self.attributes["uniform_color"]

        # Handle texture mapping
        if hasattr(self.geom, "textures"):
            if len(self.geom.textures) == 1:
                tex_img = np.asarray(self.geom.textures[0])[::-1]  # Flip texture vertically
                tex_img = np.ascontiguousarray(tex_img)
                mat.albedo_img = o3d.geometry.Image(tex_img)
            elif len(self.geom.textures) > 1:
                logger.warning(f"More than one texture found for {self.geom}, texture will not be displayed.")

        return mat

    def add_usd_prim(self, stage, prim_path):
        # https://raw.githubusercontent.com/NVIDIAGameWorks/kaolin/master/kaolin/io/usd.py
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        prim_path = sanitize_usd_prim_path(prim_path)
        if isinstance(self.geom, o3d.geometry.TriangleMesh):
            usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
            usd_mesh.AddTransformOp().Set(Gf.Matrix4d(self.pose.matrix.T))
            if self.geom.has_triangles():
                face_data = np.asarray(self.geom.triangles)
                usd_mesh.GetFaceVertexCountsAttr().Set(np.full((face_data.shape[0], ), 3, dtype=int))
                usd_mesh.GetFaceVertexIndicesAttr().Set(face_data)
            if self.geom.has_vertices():
                vert_data = np.asarray(self.geom.vertices)
                usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vert_data))
            if self.geom.has_vertex_colors():
                # This will be suppressed by material. But this serves as a save slot.
                vert_color_data = np.asarray(self.geom.vertex_colors)
                usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray.FromNumpy(vert_color_data))
                usd_mesh.GetPrimvar('displayColor').SetInterpolation('vertex')

            mesh_material = UsdShade.Material.Define(stage, f"{prim_path}/material")
            mesh_material.GetPrim().CreateAttribute(self.USD_RAW_ATTR_NAME, Sdf.ValueTypeNames.String).Set(
                str(self.attributes))
            mesh_shader = UsdShade.Shader.Define(stage, f"{prim_path}/material/shader")
            mesh_shader.CreateIdAttr("UsdPreviewSurface")

            # https://graphics.pixar.com/usd/release/spec_usdpreviewsurface.html
            if "alpha" in self.attributes:
                mesh_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(self.attributes["alpha"])
            if "material.metallic" in self.attributes:
                mesh_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(self.attributes["material.metallic"])
            if "material.roughness" in self.attributes:
                mesh_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(self.attributes["material.roughness"])

            u_color = None
            if "uniform_color" in self.attributes:
                u_color = self.attributes["uniform_color"]
            elif self.geom.has_vertex_colors():
                u_color = vert_color_data[0]
            if u_color is not None:
                mesh_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(u_color.tolist()))

            mesh_material.CreateSurfaceOutput().ConnectToSource(mesh_shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(usd_mesh).Bind(mesh_material)

        elif isinstance(self.geom, o3d.geometry.PointCloud):
            # xyz = np.asarray(self.geom.points)
            # from tqdm import tqdm
            # for idx, pts in enumerate(tqdm(xyz)):
            #     # print(idx)
            #     usd_pcd = UsdGeom.Sphere.Define(stage, prim_path + f"/s{idx}")
            #     usd_pcd.AddTransformOp().Set(Gf.Matrix4d(Isometry(t=pts).matrix.T))
            #     usd_pcd.GetRadiusAttr().Set(0.05)

            usd_pcd = UsdGeom.Points.Define(stage, prim_path)
            usd_pcd.AddTransformOp().Set(Gf.Matrix4d(self.pose.matrix.T))
            if self.geom.has_points():
                bounds = self.geom.get_max_bound() - self.geom.get_min_bound()
                min_bound = min(bounds)
                scale = (min_bound / len(self.geom.points) ** (1 / 3)).item()
                scales = np.full((len(self.geom.points), ), scale, dtype=float)
                usd_pcd.GetPointsAttr().Set(np.asarray(self.geom.points))
                usd_pcd.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(scales))
            if self.geom.has_colors():
                usd_pcd.GetDisplayColorAttr().Set(np.asarray(self.geom.colors))
        else:
            raise NotImplementedError


class SceneLight:
    """
    A class that represents a light source in a 3D scene.

    This class handles different types of lights (SUN, POINT, SPOT, AREA) and their properties
    for rendering in various engines like Filament, Blender, etc.

    Key components:
    - type: The type of light source (SUN, POINT, SPOT, AREA)
    - pose: Isometry transformation representing light's position and orientation
    - attributes: Dictionary of light properties specific to each type

    Supported light types and their attributes:
    - SUN: 
        - energy: Light intensity (in Blender watts)
        - angle: Angular diameter of sun (in radians)
    - POINT:
        - energy: Light intensity (in Blender watts) 
        - radius: Size of point light
    - SPOT:
        - energy: Light intensity (in Blender watts)
        - radius: Size of spotlight source
    - AREA:
        - energy: Light intensity (in Blender watts)
        - size: Size of area light surface

    Light direction conventions:
    - SUN & SPOT lights shoot in the -Z direction in their local coordinates

    """

    # We just use this sun to view shape for now (x100).
    PREVIEW_LIGHT_COLOR = [1., 1., 1.]
    FILAMENT_INTENSITY_MULT = 20000

    def __init__(self, mtype, pose: Isometry = Isometry(), attributes: dict = None):
        self.type = mtype
        self.pose = pose
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def setup_filament_scene(
        self,
        name,
        fengine: o3d.visualization.rendering.Scene,
        scene,
        update: bool = False
    ):
        """
        Setup the light in the open3d scene for filament.
        """
        if name == "sun":
            fengine.set_sun_light(
                -self.pose.q.rotation_matrix[:, 2], self.PREVIEW_LIGHT_COLOR,
                self.attributes["energy"] * self.FILAMENT_INTENSITY_MULT)
        elif self.type == 'SUN':
            sun_light_direction = -self.pose.q.rotation_matrix[:, 2]
            sun_light_intensity = self.attributes["energy"] * self.FILAMENT_INTENSITY_MULT
            if not update:
                # light_name, color, direction, intensity, cast_shadows
                fengine.add_directional_light(name, self.PREVIEW_LIGHT_COLOR, sun_light_direction,
                                              sun_light_intensity, True)
            else:
                fengine.update_light_direction(name, sun_light_direction)
                fengine.update_light_intensity(name, sun_light_intensity)
        elif self.type == 'POINT':
            point_light_intensity = self.attributes["energy"] * self.FILAMENT_INTENSITY_MULT
            if not update:
                # name, color, position, intensity, falloff, cast_shadows
                fengine.add_point_light(name, self.PREVIEW_LIGHT_COLOR, self.pose.t,
                                        point_light_intensity, 1000, True)
            else:
                fengine.update_light_position(name, self.pose.t)
                fengine.update_light_intensity(name, point_light_intensity)
        elif self.type == 'SPOT':
            spot_light_position = self.pose.t
            spot_light_direction = -self.pose.q.rotation_matrix[:, 2]
            spot_light_intensity = self.attributes["energy"] * self.FILAMENT_INTENSITY_MULT
            if not update:
                # name, color, position, direction, intensity, falloff, inner_cone_angle, outer_cone_angle, cast_shadows
                fengine.add_spot_light(name, self.PREVIEW_LIGHT_COLOR, spot_light_position, spot_light_direction,
                                       spot_light_intensity, 1000, 0.1, 0.2, True)
            else:
                fengine.update_light_position(name, spot_light_position)
                fengine.update_light_direction(name, spot_light_direction)
                fengine.update_light_intensity(name, spot_light_intensity)
        else:
            if not update:
                print(f"Warning: light {name} of type {self.type} is not yet supported in filament!")

    def add_usd_prim(self, stage, prim_path):
        from pxr import Gf, UsdGeom, UsdLux, Vt

        prim_path = sanitize_usd_prim_path(prim_path)

        if self.type == 'POINT':
            usd_light = UsdLux.SphereLight.Define(stage, prim_path)
            usd_light.AddTransformOp().Set(Gf.Matrix4d(self.pose.matrix.T))
            usd_light.GetRadiusAttr().Set(self.attributes["radius"])
            usd_light.GetIntensityAttr().Set(self.attributes["energy"] * 0.01)
            usd_light.GetColorAttr().Set(Gf.Vec3f(self.PREVIEW_LIGHT_COLOR))
        elif self.type == 'SUN':
            usd_light = UsdLux.DistantLight.Define(stage, prim_path)
            usd_light.AddTransformOp().Set(Gf.Matrix4d(self.pose.matrix.T))
            usd_light.GetAngleAttr().Set(self.attributes["angle"] / np.pi * 180.0)
            usd_light.GetIntensityAttr().Set(self.attributes["energy"])
            usd_light.GetColorAttr().Set(Gf.Vec3f(self.PREVIEW_LIGHT_COLOR))
        elif self.type == 'AREA':
            usd_light = UsdLux.RectLight.Define(stage, prim_path)
            usd_light.AddTransformOp().Set(Gf.Matrix4d(self.pose.matrix.T))
            usd_light.GetHeightAttr().Set(self.attributes["size"])
            usd_light.GetWidthAttr().Set(self.attributes["size"])
            usd_light.GetIntensityAttr().Set(self.attributes["energy"] * 0.01)
            usd_light.GetColorAttr().Set(Gf.Vec3f(self.PREVIEW_LIGHT_COLOR))
        else:
            print(f"Warning: light {prim_path} of type {self.type} is not yet supported in USD!")
            usd_light = UsdGeom.Xform.Define(stage, prim_path)
            # usd_light.GetPrim().CreateAttribute('render_width', Sdf.ValueTypeNames.Int).Set(self.w)


class Scene:
    """
    A comprehensive 3D scene representation that manages geometry, lighting, cameras and rendering.
    
    The Scene class serves as the main container and controller for 3D visualization, supporting
    multiple rendering backends (OpenGL, Filament, Blender, PyRender) and providing a unified
    interface for scene manipulation and rendering.

    Key Components:
    - objects: Dictionary of SceneObject instances representing 3D geometries
    - lights: Dictionary of SceneLight instances
    - camera_pose: Combined camera pose (camera_base @ relative_camera_pose)
    - camera_intrinsic: Camera intrinsic parameters
    - animator: SceneAnimator instance for handling animations
    - viewport_shading: Rendering mode ('LIT', 'UNLIT', 'NORMAL')
    
    """
    def __init__(self, cam_path: str = None, up_axis: str = '+Y'):
        self.objects = {}
        self.lights = {}

        self.cached_pyrenderer = None
        self.selection_sets = []

        # camera_pose = camera_base @ rel_camera_pose
        self.camera_base = Isometry()
        self.relative_camera_pose = Isometry()      # This only allows for quantized animation.

        self.camera_intrinsic = CameraIntrinsic(300, 300, 150, 150, 150, 150)
        self.cam_path = Path(cam_path) if cam_path else cam_path
        # These are old-api specific settings
        self.gl_render_options = o3d.visualization.RenderOption()
        self.gl_render_options.point_show_normal = False
        self.filament_show_skybox = False

        # Ambient color / Environment texture behave differently with different engines:
        #   - Blender: if env_map is None, then use ambient_color as world node, with the last element being the strength
        #              otherwise, ambient_color is simply ignored.
        #   - Filament: env_map is ignored. ambient_color is used to determine background color and IBL strength.
        #              env_map_filament is for choosing the IBL that lights the surface
        #     (how to use new env map in filament) https://google.github.io/filament/webgl/tutorial_redball.html
        #     Filament uses *_ibl.ktx and *_skybox.ktx as the format for env-map. The tool that converts from HDR
        #   Equilateral map to ktx cubemap is named 'cmgen', located at: Open3D/build/filament/src/ext_filament/bin
        #   Usage: cmgen -x [OUT_DIR] --format=ktx --size=256 [BLUR OPTIONS] [IN.png / IN.hdr]
        #     After this, put the ktx files into Open3D's installation directory under Resource/
        self.ambient_color = [1.0, 1.0, 1.0, 1.0]
        self.env_map = None
        self.env_map_rotation = 'auto'

        self._env_map_filament = ""
        self.set_filament_env_map("default")

        self.film_transparent = False
        self.background_image = None
        self.background_updated = False     # Should change outside.
        self.point_size = 5.0
        self.viewport_shading = 'LIT'       # Supported: LIT, UNLIT, NORMAL
        self.up_axis = up_axis
        self.backface_culling = False

        # We no longer add default light: this is now controlled by themes.
        self.additional_blender_commands = ""

        # Animation controls.
        self.animator = SceneAnimator(self)

        try:
            self.load_camera()
        except FileNotFoundError:
            pass
        except AttributeError:
            pass

    @property
    def camera_pose(self) -> Isometry:
        """Get the absolute camera pose by composing the camera base pose with the relative camera pose.

        The camera pose is computed by multiplying the camera base pose with the relative camera pose.
        This allows for hierarchical camera control where the camera can move relative to a base frame.

        Returns:
            Isometry: The absolute camera pose in world coordinates, computed as camera_base @ relative_camera_pose
        """
        # Compose base pose with relative pose to get absolute camera pose
        return self.camera_base @ self.relative_camera_pose

    @camera_pose.setter
    def camera_pose(self, value: Isometry) -> None:
        """Set the absolute camera pose by decomposing it into relative pose.

        This setter takes an absolute camera pose in world coordinates and converts it
        to a relative camera pose by removing the camera base transformation. This allows
        the camera to maintain its hierarchical relationship with the base frame while
        being positioned absolutely in world space.

        Args:
            value (Isometry): The desired absolute camera pose in world coordinates

        Returns:
            None
        """
        # Convert absolute pose to relative by removing base transformation
        self.relative_camera_pose = self.camera_base.inv() @ value

    def set_filament_env_map(self, name: str):
        if (o3d.get_resource_path() / f"{name}_ibl.ktx").exists():
            self._env_map_filament = str(o3d.get_resource_path() / name)
        elif (get_assets_path() / "envmaps" / f"{name}_ibl.ktx").exists():
            self._env_map_filament = str(get_assets_path() / "envmaps" / name)
        else:
            raise FileNotFoundError

    def export(self, path: Union[str, Path]):
        with Path(path).open('wb') as scene_f:
            pickle.dump(self, scene_f)

    @staticmethod
    def build_from(path: Union[str, Path]):
        pass

    def load_camera(self) -> None:
        """Load camera parameters and animation data from a saved file.

        This function loads camera parameters including extrinsic/intrinsic matrices and animation data
        from a previously saved camera configuration pickle file. The data is used to restore camera pose,
        intrinsic parameters, and animation states.

        The function expects the following data in the saved file:
        - 'extrinsic': Isometry representing the relative camera pose
        - 'intrinsic': Camera intrinsic parameters
        - 'base': Isometry representing the camera base pose
        - 'relative_animator': Optional relative camera animation data
        - 'base_animator': Optional base camera animation data

        Returns:
            None
        """
        # Load camera data from file
        with self.cam_path.open('rb') as cam_f:
            camera_data: dict = pickle.load(cam_f)

        # Set core camera parameters
        self.relative_camera_pose = camera_data['extrinsic']
        self.camera_intrinsic = camera_data['intrinsic']

        # Get optional animation and base transformation data
        relative_animator = camera_data.get('relative_animator', None)
        base_animator = camera_data.get('base_animator', None)
        camera_base = camera_data.get('base', None)

        # Apply optional parameters if they exist
        if camera_base is not None:
            self.camera_base = camera_base
        if relative_animator is not None:
            self.animator.set_relative_camera(relative_animator)
        if base_animator is not None:
            self.animator.set_camera_base(base_animator)

    def save_camera(self):
        self.cam_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cam_path.open('wb') as scene_f:
            pickle.dump({
                'extrinsic': self.relative_camera_pose,
                'intrinsic': self.camera_intrinsic,
                'base': self.camera_base,
                'relative_animator': self.animator.get_relative_camera(),
                'base_animator': self.animator.get_camera_base()
            }, scene_f)

    def quick_camera(self, pos: Optional[Union[np.ndarray, list]] = None, look_at: Optional[Union[np.ndarray, list]] = None, 
                     w: int = 1024, h: int = 768, fov: float = 60.0, up_axis: Optional[Union[str, np.ndarray, list]] = None,
                     fill_percent: float = 0.5, plane_angle: float = 45.0, pitch_angle: float = 20.0, 
                     no_override: bool = False):
        """Set up a quick camera configuration for the scene.

        This function configures the camera parameters either automatically based on the scene extent
        or using provided position and look-at points. It handles camera positioning, orientation,
        and intrinsic parameters.

        Args:
            pos: Camera position as a 3D numpy array. If None, position is calculated automatically.
            look_at: Point the camera looks at as a 3D numpy array. If None, calculated from scene center.
            w: Width of the camera viewport in pixels.
            h: Height of the camera viewport in pixels.
            fov: Field of view angle in degrees.
            up_axis: Up direction for camera orientation. Can be string (e.g. '+Y') or 3D numpy array.
                    If None, uses scene's default up axis.
            fill_percent: Target percentage of viewport to be filled by scene objects (for auto positioning).
            plane_angle: Horizontal rotation angle in degrees (for auto positioning).
            pitch_angle: Vertical tilt angle in degrees (for auto positioning).
            no_override: If True and camera file exists, keep existing camera settings.

        Returns:
            self: Returns the scene object itself.
        """
        # Check if should load existing camera settings
        if self.cam_path is not None and self.cam_path.exists() and no_override:
            return self

        # Process up axis input
        if up_axis is not None:
            if isinstance(up_axis[0], str):
                self.up_axis = up_axis
                up_axis = Isometry._str_to_axis(up_axis)
            else:
                up_axis = np.array(up_axis)
        else:
            up_axis = Isometry._str_to_axis(self.up_axis)

        # Calculate camera position and orientation if not provided
        if pos is None or look_at is None:
            # Calculate camera plane vectors
            if abs(up_axis[0]) < 0.01 and abs(up_axis[1]) < 0.01:
                cplane_x = np.array([0.0, up_axis[2], -up_axis[1]])
            else:
                cplane_x = np.array([up_axis[1], -up_axis[0], 0.0])
            cplane_y = np.cross(up_axis, cplane_x)
            cplane_y = cplane_y / np.linalg.norm(cplane_y)

            # Handle empty scene case
            if len(self.objects) == 0:
                logger.warning("Please add objects before using quick-camera!")
                pos = np.array([1.0, 0.0, 0.0])
                look_at = np.array([0.0, 0.0, 0.0])
            else:
                # Calculate auto camera position based on scene extent
                min_extent, max_extent = self.get_scene_extent()
                view_ratio = np.max(max_extent - min_extent) / fill_percent / 2
                distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
                look_at = (min_extent + max_extent) / 2. # look_at is the center of the scene
                plane_deg = plane_angle / 180.0 * np.pi
                pitch_deg = pitch_angle / 180.0 * np.pi
                pos = look_at + distance * (np.cos(plane_deg) * np.cos(pitch_deg) * cplane_x +
                                            np.sin(plane_deg) * np.cos(pitch_deg) * cplane_y +
                                            np.sin(pitch_deg) * up_axis)

        # Set final camera parameters
        self.camera_base = Isometry(t=look_at).validified() # camera_base is the center of the scene
        self.camera_pose = Isometry.look_at(np.asarray(pos), np.asarray(look_at), up_axis).validified()
        self.camera_intrinsic = CameraIntrinsic.from_fov(w, h, np.deg2rad(fov))
        return self

    def add_object(self, geom: Union[SceneObject, o3d.geometry.Geometry3D], name: str = None, 
                  pose: Isometry = None, attributes: dict = None, return_name: bool = False):
        """Add a new object to the scene.

        This function adds a geometry object to the scene with specified name, pose and attributes.
        If no name is provided, a random UUID will be generated. The object can be either a 
        SceneObject or a geometry from Open3D.

        Args:
            geom: The geometry to add. Can be either a SceneObject or an Open3D geometry like PointCloud, TriangleMesh, etc.
            name: Optional name for the object. If None, a random UUID will be generated
            pose: Optional pose (position/orientation) for the object. Defaults to identity pose
            attributes: Optional dictionary of rendering attributes for the SceneObject, see SceneObject.attributes.
            return_name: If True, returns the name of added object instead of self

        Returns:
            Union[Scene, str]: Returns self if return_name is False, otherwise returns the name of the added object.
        """
        # Generate random UUID if no name provided
        if name is None:
            new_name = "obj" + str(uuid.uuid1())[:8]
        else:
            # Ensure name doesn't already exist
            assert name not in self.objects.keys()
            new_name = name

        # Use identity pose if none provided
        if pose is None:
            pose = Isometry()

        # Handle SceneObject vs raw geometry
        if isinstance(geom, SceneObject):
            new_obj = copy.deepcopy(geom)
        else:
            new_obj = SceneObject(geom, pose=pose, attributes=attributes)

        # Add to scene objects dictionary
        self.objects[new_name] = new_obj

        return self if not return_name else new_name

    def remove_object(self, name: str, non_exist_ok: bool = False):
        """Remove an object from the scene by its name.

        This function removes an object from the scene's objects dictionary. If the object
        doesn't exist and non_exist_ok is False, it will raise a KeyError.

        Args:
            name: Name/ID of the object to remove
            non_exist_ok: If True, silently ignore when trying to remove non-existent object.
                         If False, raise KeyError when object doesn't exist. Defaults to False.

        Returns:
            Scene: Returns self for method chaining
        """
        # Try to remove object from scene
        try:
            del self.objects[name]
        except KeyError:
            # Raise error if object doesn't exist and non_exist_ok is False
            if not non_exist_ok:
                raise
        return self

    def set_object_attribute(self, name: str, kwargs: dict):
        """Set or update attributes for an object in the scene.

        This function updates the attributes dictionary of a scene object with new key-value pairs.
        If the attribute already exists, it will be updated with the new value.

        Args:
            name: Name of the object whose attributes should be updated
            kwargs: Dictionary containing the attributes to set/update

        Returns:
            Scene: Returns self for method chaining
        """
        # Check if object exists in scene
        assert name in self.objects.keys()
        # Update object's attributes with new key-value pairs
        self.objects[name].attributes.update(kwargs)
        return self

    def add_light(self, light: Union[str, SceneLight], name: Optional[str] = None, 
                 pose: Isometry = Isometry(), attributes: Optional[dict] = None):
        """Add a light source to the scene.

        This function adds a new light source to the scene, either by creating a new SceneLight
        instance or by copying an existing one. If no name is provided, a random UUID is generated.

        Args:
            light: Either a string specifying the light type ('SUN', 'POINT', 'SPOT', 'AREA') 
                  or an existing SceneLight instance to copy
            name: Optional name/ID for the light. If None, generates random UUID. 
                 Must be unique among existing lights.
            pose: Isometry transformation for light position/orientation. Defaults to identity.
            attributes: Optional dictionary of light-specific attributes (energy, radius, etc.)

        Returns:
            Scene: Returns self for method chaining
        """
        # Generate random UUID if no name provided
        if name is None:
            new_name = str(uuid.uuid1())[:8]
        else:
            # Ensure light name doesn't already exist
            assert name not in self.lights.keys()
            new_name = name

        # Create new light - either copy existing or create from parameters
        if isinstance(light, SceneLight):
            new_light = copy.deepcopy(light)
        else:
            new_light = SceneLight(light, pose, attributes)

        # Add light to scene's lights dictionary
        self.lights[new_name] = new_light
        return self

    def remove_light(self, name, non_exist_ok: bool = False):
        try:
            del self.lights[name]
        except KeyError:
            if not non_exist_ok:
                raise
        return self
    
    def add_light_sun(self, name: Optional[str] = None, 
                      light_look_at: Optional[Union[np.ndarray, list]] = None,
                      light_dir: Optional[Union[np.ndarray, list]] = None, 
                      light_energy: float = 5.0,
                      angle: float = 0.1745):
        """Add a directional sun light to the scene.

        This function creates a directional sun light source that illuminates the scene from a 
        specified direction. The light direction can be specified either directly via light_dir
        or by providing a target point that the light should look at.

        Args:
            name: Optional name for the light. If None and "sun" not in scene, uses "sun".
                 Must be unique among existing lights.
            light_look_at: starting from (0, 0, 0), look at the target point to determine light direction
            light_dir: Optional explicit quaternion vector for light. Takes precedence over light_look_at.
            light_energy: Intensity/brightness of the light in watts. Defaults to 5.0.
            angle: Angular diameter of sun disk in radians, Defaults to 0.1745 

        Returns:
            Scene: Returns self for method chaining.
        """
        # Use "sun" as default name if available
        if name is None and "sun" not in self.lights:
            name = "sun"

        # Set light orientation - either from explicit direction or look-at target
        if light_dir is not None:
            light_iso = Isometry(q=Quaternion(light_dir))
        else:
            if light_look_at is None:
                # Default to negative of up axis if no target specified
                light_look_at = -Isometry._str_to_axis(self.up_axis)
            light_iso = Isometry.look_at(light_look_at, np.zeros(3)) # light_look_at -> (0, 0, 0)

        # Create and add the sun light
        return self.add_light('SUN', name, light_iso, {'energy': light_energy, 'angle': angle})

    def add_light_point(self, name=None, energy=100, radius=0.1, pos=None):
        if pos is None:
            pos = (0.0, 0.0, 0.0)
        return self.add_light('POINT', name, Isometry(t=pos, q=Quaternion([1.0, 0.0, 0.0, 0.0])),
                              {'energy': energy, 'radius': radius})

    def add_light_area(self, name=None, energy=100, size=0.5, pos=None, lookat=None):
        if pos is None:
            pos = (0.0, 1.0, 0.0)
        if lookat is None:
            lookat = (0.0, 0.0, 0.0)
        rot_q = (Isometry.look_at(np.asarray(pos), np.asarray(lookat)).q *
                    Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)).q
        return self.add_light('AREA', name, Isometry(t=(0.0, 0.0, 0.0), q=rot_q),
                              {'energy': energy, 'size': size})

    def add_light_spot(self, name=None, energy=100, radius=0.1, pos=None, lookat=None):
        if pos is None:
            pos = (0.0, 1.0, 0.0)
        if lookat is None:
            lookat = (0.0, 0.0, 0.0)
        rot_q = (Isometry.look_at(np.asarray(pos), np.asarray(lookat)).q *
                    Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)).q
        return self.add_light('SPOT', name, Isometry(t=(0.0, 0.0, 0.0), q=rot_q),
                              {'energy': energy, 'radius': radius})

    def auto_plane(self, config: Optional[str] = None, dist_ratio: float = 0.1, scale: float = 10.0):
        """Automatically create and add a ground plane to the scene based on scene extent.

        Creates a plane aligned with the scene's up axis and positioned relative to the scene bounds.
        The plane is sized proportionally to the scene extent and can be offset from the scene center.

        Args:
            config: Optional[str]. Axis configuration string (e.g. '+Y', '-Z'). If None, uses scene's up_axis.
                   First char indicates sign ('+'/'-'), second char indicates axis ('X'/'Y'/'Z').
            dist_ratio: float. Ratio to offset plane from scene center along up axis. Default 0.1.
            scale: float. Scale factor for plane size relative to scene extent. Default 10.0.

        Returns:
            Scene: Returns self for method chaining.
        """
        # Get scene bounds
        min_extent, max_extent = self.get_scene_extent()
        scene_center = (min_extent + max_extent) / 2.

        # Set plane orientation based on config
        if config is None:
            config = self.up_axis
        else:
            self.up_axis = config       # Update scene up axis to match plane

        # Calculate plane normal direction
        axis_idx = ['X', 'Y', 'Z'].index(config[1])  # Get index of axis (0=X, 1=Y, 2=Z)
        symbol_mult = 1 if config[0] == '-' else -1  # Get sign multiplier
        plane_normal = np.zeros((3,))
        plane_normal[axis_idx] = -symbol_mult

        # Calculate plane center position
        plane_center = np.copy(scene_center)
        scene_extent = max_extent - min_extent
        plane_center[axis_idx] += symbol_mult * scene_extent[axis_idx] * (0.5 + dist_ratio)
        scene_extent[axis_idx] = 0.0

        # Create and add plane to scene
        my_plane = vis.plane(plane_center, plane_normal, scale=np.linalg.norm(scene_extent) * scale)
        my_plane.compute_vertex_normals()
        return self.add_object(my_plane, name='auto_plane')

    def get_scene_extent(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the axis-aligned bounding box that encompasses all objects in the scene.
        
        Computes the minimum and maximum extents across all objects by:
        1. Getting bounding boxes for each object
        2. Finding the minimum of all minimum corners
        3. Finding the maximum of all maximum corners

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - min_extent: 3D numpy array of minimum x,y,z coordinates
                - max_extent: 3D numpy array of maximum x,y,z coordinates
        """
        # Get bounding boxes for all objects as (min_corner, max_corner) pairs
        all_bbox = [obj.get_extent() for obj in self.objects.values()]
        
        # Transpose list to group min corners and max corners
        all_bbox = list(zip(*all_bbox))
        
        # Find global min/max extents across all objects
        min_extent = np.asarray(all_bbox[0]).min(axis=0)  # Minimum of all min corners
        max_extent = np.asarray(all_bbox[1]).max(axis=0)  # Maximum of all max corners
        
        return min_extent, max_extent

    def center_geometries(self):
        # Compute the center of geometries.
        min_extent, max_extent = self.get_scene_extent()
        center = (min_extent + max_extent) / 2.
        for obj in self.objects.values():
            obj.pose.t -= center
        return self

    def _build_raycasting_scene(self):
        scene = o3d.t.geometry.RaycastingScene()
        for obj in self.objects.values():
            scene.add_triangles(obj.geom)
        return scene

    def _build_gl_engine(self, window_name: str, visible: bool, pos: Tuple[int, int] = None):
        """Build and configure an Open3D visualization engine with OpenGL backend.
        
        This function creates a visualization window and configures it with the current scene settings,
        including camera parameters, rendering options, and all geometries in the scene.
        
        Args:
            window_name (str): Name of the visualization window
            visible (bool): Whether the window should be visible initially
            pos (Tuple[int, int], optional): Window position as (x,y) coordinates. Defaults to (50,50)
            
        Returns:
            GLEngineWrapper: A wrapped visualization engine containing the configured Open3D visualizer
                           and a dictionary of displayed geometries
        """
        # Set default window position if none provided
        if pos is None:
            pos = (50, 50)
            
        # Create visualization engine and wrapper
        engine = o3d.visualization.VisualizerWithKeyCallback()
        warpped_engine = GLEngineWrapper(engine)
        
        # Create visualization window with specified parameters.
        # when visible is False, the window will not be shown, but OpenGL context is still created.
        engine.create_window(window_name=window_name,
                             width=self.camera_intrinsic.w,
                             height=self.camera_intrinsic.h,
                             left=pos[0], top=pos[1],
                             visible=visible)
                             
        # Add all scene geometries to the engine
        for mesh_name, mesh_obj in self.objects.items():
            geom = mesh_obj.get_transformed() # apply pose and scale
            # Convert to legacy format if needed
            if hasattr(geom, "to_legacy"):
                geom = geom.to_legacy()
            engine.add_geometry(geom)
            warpped_engine.displayed_geometries[mesh_name] = geom
            
        # Configure camera parameters
        engine.get_view_control().convert_from_pinhole_camera_parameters(
            self.camera_intrinsic.get_pinhole_camera_param(self.camera_pose, fix_bug=True), allow_arbitrary=True
        )
        
        # Configure rendering options
        self.gl_render_options.point_size = self.point_size
        self.gl_render_options.save_to_json("/tmp/ro.json")
        engine.get_render_option().load_from_json("/tmp/ro.json")
        
        # Set viewport shading mode
        if self.viewport_shading == 'NORMAL':
            engine.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
            engine.get_render_option().point_color_option = o3d.visualization.PointColorOption.Normal
        elif self.viewport_shading == 'UNLIT':
            engine.get_render_option().light_on = False
            
        # Configure backface culling
        engine.get_render_option().mesh_show_back_face = not self.backface_culling
        
        return warpped_engine

    def _update_gl_engine(self, gl_engine: GLEngineWrapper) -> None:
        """Update the OpenGL visualization engine with current camera and scene objects. 

        This function handles updating the visualization when camera parameters or scene objects 
        change during animation. It updates camera parameters if camera animation events exist,
        and updates any scene objects that have animation events.

        Args:
            gl_engine: GLEngineWrapper
                The wrapped OpenGL visualization engine containing the Open3D visualizer
                and displayed geometries dictionary

        Returns:
            None
        """
        # Get the underlying Open3D visualizer engine
        engine = gl_engine.engine

        # Update camera if camera animation events exist
        if "relative_camera" in self.animator.events.keys() or "camera_base" in self.animator.events.keys():
            engine.get_view_control().convert_from_pinhole_camera_parameters(
                self.camera_intrinsic.get_pinhole_camera_param(self.camera_pose, fix_bug=True), allow_arbitrary=True
            )

        # Update any objects that have animation events
        for obj_uuid in self.animator.events.keys():
            if obj_uuid in gl_engine.displayed_geometries.keys():
                # Remove old geometry
                old_geom = gl_engine.displayed_geometries[obj_uuid]
                engine.remove_geometry(old_geom, reset_bounding_box=False)

                # Get updated geometry with new transform
                geom = self.objects[obj_uuid].get_transformed()
                
                # Convert to legacy format if needed
                if hasattr(geom, "to_legacy"):
                    geom = geom.to_legacy()

                # Add updated geometry and store reference
                engine.add_geometry(geom, reset_bounding_box=False)
                gl_engine.displayed_geometries[obj_uuid] = geom

    def _build_filament_engine(self, engine: Union[o3d.visualization.rendering.OffscreenRenderer, o3d.visualization.O3DVisualizer]):
        """Build and configure a Filament rendering engine for the scene. Set up scene objects,
        camera, lighting, background, and other rendering settings.
        It supports both offline rendering via OffscreenRenderer and interactive visualization via O3DVisualizer.

        Args:
            engine: Union[o3d.visualization.rendering.OffscreenRenderer, o3d.visualization.O3DVisualizer]
                The Filament rendering engine to configure. Can be either an OffscreenRenderer for 
                offline rendering or O3DVisualizer for interactive GUI visualization.

        Returns:
            None
        """
        # Initialize variables based on engine type
        if isinstance(engine, o3d.visualization.rendering.OffscreenRenderer):
            # -- Used for offline render (CPU rendering, which is introduced in https://www.open3d.org/2022/10/19/open3d-0-16-is-out/)
            visualizer = None               # (no gui)
            scene = engine.scene            # o3d.visualization.rendering.Open3DScene
            sv = scene                      # o3d.visualization.rendering.Open3DScene
        elif isinstance(engine, o3d.visualization.O3DVisualizer):
            # -- Used for GUI.
            visualizer = engine             # o3d.visualization.O3DVisualizer
            scene = visualizer.scene        # o3d.visualization.rendering.Open3DScene
            sv = visualizer                 # o3d.visualization.O3DVisualizer
        else:
            raise NotImplementedError

        # Using the shared API enables update of GUI.
        if self.background_image is not None:
            background_data = o3d.geometry.Image(self.background_image)
            sv.set_background((1.0, 1.0, 1.0, 1.0), background_data)
        elif self.film_transparent:
            sv.set_background((1.0, 1.0, 1.0, 1.0), None)
        else:
            sv.set_background(list(self.ambient_color[:3]) + [1.0], None)

        # Configure GUI-specific settings if using visualizer
        if visualizer is not None:
            # Although we already set material above, the following will make the UI correct.
            visualizer.point_size = int(self.point_size)
            visualizer.line_width = 1
            visualizer.scene_shader = {
                'LIT': visualizer.STANDARD, 'UNLIT': visualizer.UNLIT, 'NORMAL': visualizer.NORMALS
            }[self.viewport_shading]
            visualizer.enable_sun_follows_camera(False)

        # Add scene objects and configure their materials
        for mesh_name, mesh_obj in self.objects.items():
            mat = mesh_obj.get_filament_material(self.viewport_shading, int(self.point_size))
            if "text" in mesh_obj.attributes:
                if visualizer is not None:
                    sv.add_3d_label(mesh_obj.pose @ mesh_obj.attributes["text_pos"], mesh_obj.attributes["text"])
            else:
                sv.add_geometry(mesh_name, mesh_obj.geom, mat)
                pose_matrix = mesh_obj.scaled_iso.matrix
                scene.set_geometry_transform(mesh_name, pose_matrix)
                scene.set_geometry_double_sided(mesh_name, not self.backface_culling)

        # Configure rendering quality and shadow settings
        o3dr = o3d.visualization.rendering
        scene.view.set_color_grading(o3dr.ColorGrading(o3dr.ColorGrading.Quality.ULTRA,
                                                       o3dr.ColorGrading.ToneMapping.LINEAR))
        scene.view.set_shadowing(True, scene.view.ShadowType.VSM)

        # Configure environment lighting
        sv.show_skybox(self.filament_show_skybox)
        scene.scene.set_indirect_light(self._env_map_filament)
        scene.scene.set_indirect_light_intensity(self.ambient_color[-1] * 37500)
        
        # Handle environment map rotation
        if self.env_map_rotation == "auto":
            if self.up_axis == '+Z':
                scene.scene.set_indirect_light_rotation(Isometry.from_axis_angle('z', -90.0).matrix)
        else:
            scene.scene.set_indirect_light_rotation(Isometry.from_euler_angles(*self.env_map_rotation).matrix)

        # Configure scene lighting
        if "sun" not in self.lights.keys():
            scene.scene.enable_sun_light(False)
        for light_name, light_obj in self.lights.items():
            light_obj.setup_filament_scene(light_name, scene.scene, self, update=False)

        # Final visualizer setup if using GUI
        if visualizer is not None:
            engine.show_settings = False
            engine.reset_camera_to_default()  # important, because it correctly set up scene bounds

        # Configure camera parameters. same API for both GUI and offline render.
        engine.setup_camera(self.camera_intrinsic.to_open3d_intrinsic(), self.camera_pose.inv().matrix)

    def _update_filament_engine(self, engine: Union[o3d.visualization.rendering.OffscreenRenderer, o3d.visualization.O3DVisualizer]):
        """Updates the filament rendering engine with the current scene state.

        This function handles updating the filament engine for both offline rendering and GUI visualization.
        It updates camera parameters, object transforms, lighting, background and other rendering settings
        based on any changes in the scene state.

        Args:
            engine: The filament rendering engine to update. Can be either:
                   - OffscreenRenderer for offline rendering
                   - O3DVisualizer for GUI visualization

        Returns:
            None
        """
        if isinstance(engine, o3d.visualization.rendering.OffscreenRenderer):
            # -- Used for offline render
            visualizer = None               # (no gui)
            scene = engine.scene            # o3d.visualization.rendering.Open3DScene
            sv = scene                      # o3d.visualization.rendering.Open3DScene
        elif isinstance(engine, o3d.visualization.O3DVisualizer):
            # -- Used for GUI.
            visualizer = engine             # o3d.visualization.O3DVisualizer
            scene = visualizer.scene        # o3d.visualization.rendering.Open3DScene
            sv = visualizer                 # o3d.visualization.O3DVisualizer
        else:
            raise NotImplementedError

        # Update skybox 
        sv.show_skybox(self.filament_show_skybox)

        # Update camera if camera-related animation events exist
        if "relative_camera" in self.animator.events.keys() or "camera_base" in self.animator.events.keys():
            engine.setup_camera(self.camera_intrinsic.to_open3d_intrinsic(), self.camera_pose.inv().matrix)

        # Update transforms and lights for objects with animation events
        for obj_uuid in self.animator.events.keys():
            if scene.has_geometry(obj_uuid):
                pose_matrix = self.objects[obj_uuid].scaled_iso.matrix
                scene.set_geometry_transform(obj_uuid, pose_matrix)
            if obj_uuid in self.lights.keys():
                self.lights[obj_uuid].setup_filament_scene(obj_uuid, scene.scene, self, update=True)

        # Update background if changed
        if self.background_updated and self.background_image is not None:
            self.background_updated = False
            background_data = o3d.geometry.Image(self.background_image)
            sv.set_background((1.0, 1.0, 1.0, 1.0), background_data)

        if visualizer is not None:
            visualizer.mouse_mode = gui.SceneWidget.Controls.ROTATE_CAMERA

    def record_camera_pose(self, vis: Union[o3d.visualization.O3DVisualizer, o3d.visualization.Visualizer]) -> bool:
        """Records the current camera pose and intrinsic parameters from the visualizer.

        This function extracts and stores the camera pose and intrinsic parameters from either the new
        O3DVisualizer or legacy Visualizer. It handles both APIs by checking for the presence of
        get_view_control method. If a camera path is specified, it also saves the camera parameters to file.

        Args:
            vis: The Open3D visualizer instance, can be either O3DVisualizer (new API) or 
                Visualizer (legacy API)

        Returns:
            bool: Always returns False to indicate the visualization should continue
        """
        # Handle new O3DVisualizer API which doesn't have get_view_control
        if not hasattr(vis, "get_view_control"):
            self.camera_pose, self.camera_intrinsic = VisualizerManager.get_window_camera_info(vis)
        # Handle legacy Visualizer API
        else:
            cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam_pose = cam_param.extrinsic
            self.camera_pose = Isometry.from_matrix(cam_pose).inv()
            self.camera_intrinsic = CameraIntrinsic.from_open3d_intrinsic(cam_param.intrinsic)
        
        # Save camera parameters if path is specified
        if self.cam_path is not None:
            self.save_camera()
            
        logger.info(f"Camera parameter saved! w={self.camera_intrinsic.w}, h={self.camera_intrinsic.h}")
        return False

    def preview(self, allow_change_pose: bool = True, add_ruler: bool = True, title: str = "Render Preview", 
                use_new_api: bool = False, key_bindings: dict = None):
        """Preview the current scene in an interactive visualization window.

        This function displays the scene in a visualization window using Open3D's visualization tools.
        It handles camera pose changes, animation frames, and custom key bindings.

        Args:
            allow_change_pose (bool): Whether to allow camera pose changes via user interaction. 
                                    Defaults to True.
            add_ruler (bool): Not used.
            title (str): Title of the visualization window. Defaults to "Render Preview".
            use_new_api (bool): Whether to use O3DVisualizer powered by filament.
            key_bindings (dict): Custom key bindings mapping keys to callback functions. 
                                Defaults to None. e.g. {'r': self.record_camera_pose}

        Returns:
            Scene: Returns self for method chaining.
        """
        # Check and clear any existing buffered scenes
        if len(vis_manager.scenes) != 0:
            print("Warning: there are still buffered scenes in the manager which are not fully shown.")
            vis_manager.reset()

        # Update animation frame if animator is enabled
        if self.animator.is_enabled():
            self.animator.set_current_frame()

        # Add scene to visualization manager with optional camera pose recording callback
        vis_manager.add_scene(self, title, self.record_camera_pose if allow_change_pose else None)
        vis_manager.run(use_new_api=use_new_api, key_bindings=key_bindings)
        return self
    
    def preview_polyscope(self) -> None:
        """Preview the scene using Polyscope visualization library.

        This function displays point cloud objects from the scene in a Polyscope visualization window.
        Only point cloud geometries are supported. For point clouds with color information, the colors
        will be displayed as well.

        """
        # Import and initialize polyscope
        import polyscope as ps
        ps.init()

        # Iterate through scene objects
        for obj_name, obj in self.objects.items():
            # Only handle point cloud objects
            if isinstance(obj, o3d.geometry.PointCloud):
                # Register point cloud with polyscope
                ps_pcd = ps.register_point_cloud(obj_name, np.asarray(obj.points))
                
                # Add colors if available
                if obj.has_color():
                    ps_pcd.add_color_quantity("color", np.asarray(obj.colors))
        
        # Show the polyscope GUI
        ps.show()
    def _setup_blender_static(self) -> None:
        """Set up the static contents of the scene in Blender.

        This function configures the static scene elements in Blender, including:
        - Scene objects with their geometry, poses and attributes
        - Lights and their properties 
        - Camera setup with base pose and intrinsics
        - Environment map or background color
        - Film transparency settings
        - Any additional custom Blender commands

        The function sends commands to Blender to create and configure these elements.

        Args:
            None

        Returns:
            None
        """
        # Clear existing scene contents
        blender.send_clear()

        # Add scene objects (meshes, point clouds etc)
        for obj_uuid, obj_data in self.objects.items():
            blender.send_entity(obj_data.geom, obj_uuid, obj_data.pose, obj_data.attributes)

        # Add lights to the scene
        for light_uuid, light_data in self.lights.items():
            blender.send_light(light_data, light_uuid)

        # Set up camera base pose
        blender.send_entity_pose("camera_base", rotation_mode='QUATERNION', rotation_value=self.camera_base.q.q.tolist(),
                                 location_value=self.camera_base.t.tolist())
        
        # Configure camera parameters
        blender.send_camera(self.relative_camera_pose, self.camera_intrinsic)

        # Set up environment map or background color
        if self.env_map is not None:
            # Handle environment map rotation
            if self.env_map_rotation == 'auto':
                rotation = [0.0, 0.0, 0.0] if self.up_axis != '+Y' else [np.pi / 2., 0.0, 0.0]
            else:
                rotation = self.env_map_rotation
            blender.send_envmap(self.env_map, rotation=rotation)
        else:
            # Set background color if no environment map
            blender.send_eval(f"bg_node=bpy.data.worlds['World'].node_tree.nodes['Background'];"
                              f"bg_node.inputs[0].default_value=({self.ambient_color[0]},{self.ambient_color[1]},{self.ambient_color[2]},1);"
                              f"bg_node.inputs[1].default_value={self.ambient_color[3]}")

        # Configure film transparency
        blender.send_eval(f"bpy.context.scene.render.film_transparent={self.film_transparent}")

        # Execute any additional custom Blender commands
        if self.additional_blender_commands:
            blender.send_eval(self.additional_blender_commands)

    def setup_blender_and_detach(self, pause: bool = True, save_path: Optional[Path] = None):
        """
        Set up the Blender scene and detach from Python control, optionally saving the scene as .blend file.

        This function configures the Blender scene with the current scene settings and animations,
        then detaches Blender from Python control. This allows manual editing of the scene in 
        Blender's GUI. Optionally saves the Blender file before detaching.

        Args:
            pause (bool): If True, pause execution after detaching to keep Blender open. 
                         Default is True.
            save_path (Optional[Path]): Path to save the .blend file before detaching.
                                      If None, scene is not saved. Default is None.

        Returns:
            None
        """
        # Set up static scene elements (objects, lights, camera etc)
        self._setup_blender_static()

        # Configure animation if enabled
        if self.animator.is_enabled():
            self.animator.send_blender()

        # Save .blend file if path provided
        if save_path is not None:
            blender.send_save(Path(save_path))

        # Detach Blender from Python control
        blender.send_detach()

        # Optionally pause to keep Blender open
        if pause:
            logger.info("Blender exported. The process is no longer controlled by pycg.")
            pdb.set_trace()

    def render_blender(self, do_render: bool = True, save_path: Optional[str] = None, quality: int = 128) -> Optional[np.ndarray]:
        """Render the current scene using Blender.

        This function sets up and renders the scene in Blender. It can either save the rendered image
        to a file or return it as a numpy array. The rendering can be paused before execution if needed.

        Args:
            do_render (bool): If False, pauses before rendering to allow manual scene adjustments.
                            Default is True.
            save_path (Optional[str]): Path to save the rendered image. If None, the image is returned
                                     as a numpy array instead. Default is None.
            quality (int): Render quality/samples in Blender. Higher values give better quality but
                         slower rendering. Default is 128.

        Returns:
            Optional[np.ndarray]: If save_path is None, returns the rendered RGB image as a numpy array in uint8 format.
                                 If save_path is provided, returns None as the image is saved to file.
        """
        # Set up the static scene elements in Blender
        self._setup_blender_static()

        # Optionally pause before rendering if requested or env var is set
        if not do_render or 'PAUSE_BEFORE_RENDER' in os.environ.keys():
            # Wait for user to respond
            blender.poll_notified()

        if save_path is not None:
            # Render directly to specified save path
            blender.send_render(quality=quality, save_path=save_path)
            return None
        else:
            # Render to temporary file and read back the image
            with tempfile.TemporaryDirectory() as render_tmp_dir_p:
                # By default, blender will add '.png' to the input path if the suffix didn't exist.
                render_tmp_file = Path(render_tmp_dir_p) / "rgb.png"
                blender.send_render(quality=quality, save_path=str(render_tmp_file))
                rgb_img = image.read(render_tmp_file)

        return rgb_img

    def render_blender_animation(self, do_render: bool = True, quality: int = 128,
                               start_t: Optional[int] = None, end_t: Optional[int] = None):
        """Render an animation sequence using Blender.

        This function renders a sequence of frames in Blender based on the animation timeline.
        It sets up the scene, allows for optional manual adjustments before rendering,
        and yields each rendered frame along with its timestamp.

        Args:
            do_render (bool): If False, pauses before rendering to allow manual scene adjustments.
                            Default is True.
            quality (int): Render quality/samples in Blender. Higher values give better quality but
                         slower rendering. Default is 128.
            start_t (Optional[int]): Starting frame number for the animation sequence. If None,
                                   uses the animator's default start frame. Default is None.
            end_t (Optional[int]): Ending frame number for the animation sequence. If None,
                                 uses the animator's default end frame. Default is None.

        Yields:
            Tuple[int, np.ndarray]: A tuple containing:
                - int: Current frame number
                - np.ndarray: RGB image of the rendered frame as a numpy array in uint8 format.
        """
        # Set up the static scene elements in Blender
        self._setup_blender_static()
        
        # Send animation data to Blender
        self.animator.send_blender()
        
        # Optionally pause before rendering if requested or env var is set
        if not do_render or 'PAUSE_BEFORE_RENDER' in os.environ.keys():
            blender.poll_notified()

        # Get animation frame range
        t_start, t_end = self.animator.get_range()
        if start_t is not None:
            t_start = start_t
        if end_t is not None:
            t_end = end_t
            
        # Render each frame in the sequence
        for t_cur in range(t_start, t_end + 1):
            # Set current frame in Blender
            blender.send_eval(f"bpy.context.scene.frame_set({t_cur})")
            
            # Render frame to temporary file and read it back
            with tempfile.TemporaryDirectory() as render_tmp_dir_p:
                # By default, blender will add '.png' to the input path if the suffix didn't exist
                render_tmp_file = Path(render_tmp_dir_p) / "rgb.png"
                blender.send_render(quality=quality, save_path=str(render_tmp_file))
                rgb_img = image.read(render_tmp_file)
                
            yield t_cur, rgb_img

    def render_opengl_depth(self) -> np.ndarray:
        """Render a single depth image using OpenGL. GUI is required.
        
        This function creates an OpenGL rendering engine, renders the current scene state,
        and captures the depth buffer. The depth values are returned as a floating point array.
        The OpenGL window is destroyed after rendering.

        Returns:
            np.ndarray: Depth image as a floating point numpy array, where each pixel value
                       represents the depth from the camera to the surface at that point.
        """
        # Create OpenGL engine instance
        gl_engine = self._build_gl_engine("_", False)
        engine = gl_engine.engine
        
        # Update and render the scene
        engine.poll_events()
        engine.update_renderer()
        
        # Capture depth buffer
        captured_depth = engine.capture_depth_float_buffer(do_render=True)
        
        # Clean up OpenGL context
        engine.destroy_window()
        
        return np.asarray(captured_depth)
    
    def render_opengl_depth_animation(self):
        """Render an animation sequence of depth images using OpenGL. GUI is required.
        
        This function creates an OpenGL rendering engine, renders the scene state at each frame,
        and captures the depth buffer. The depth values are returned as a floating point array.
        The OpenGL window is destroyed after rendering.

        Returns:
            Generator[Tuple[int, np.ndarray]]: A generator that yields each frame's depth image.
        """
        gl_engine = self._build_gl_engine("_", False)
        engine = gl_engine.engine

        t_start, t_end = self.animator.get_range()
        for t_cur in range(t_start, t_end + 1):
            self.animator.set_frame(t_cur)
            self._update_gl_engine(gl_engine)
            engine.poll_events()
            engine.update_renderer()
            captured_depth = np.asarray(engine.capture_depth_float_buffer(do_render=True))
            yield t_cur, captured_depth

        engine.destroy_window()

    def render_raycasting_depth(self):
        raise NotImplementedError

    def render_opengl(self, multisample: int = 1, need_alpha=False, save_path: str = None):
        """Render a single RGB image using OpenGL. GUI is required.

        This function creates an OpenGL rendering engine, renders the current scene state,
        and captures the screen buffer. The RGB values are returned as a floating point array.
        The OpenGL window is destroyed after rendering.

        Args:
            multisample (int): Render quality/samples in OpenGL. Higher values give better quality but
                             slower rendering. Default is 1.
            need_alpha (bool): If True, include an alpha channel in the output image. Default is False.
            save_path (Optional[str]): Path to save the rendered image. If None, the image is returned
                                     as a numpy array instead. Default is None.

        Returns:
            np.ndarray: Rendered RGB image as a numpy array in uint8 format.
        """
        saved_intrinsic = copy.copy(self.camera_intrinsic)
        self.camera_intrinsic = self.camera_intrinsic.scale(multisample)
        gl_engine = self._build_gl_engine("_", False)
        engine = gl_engine.engine
        engine.poll_events()
        engine.update_renderer()
        captured_rgb = engine.capture_screen_float_buffer(do_render=True)
        captured_rgb = np.asarray(captured_rgb)
        captured_alpha = 1.0 - (np.sum(captured_rgb, axis=2, keepdims=True) > 2.99).astype(float)
        engine.destroy_window()
        self.camera_intrinsic = saved_intrinsic
        if multisample != 1:
            h = captured_rgb.shape[0] // multisample
            w = captured_rgb.shape[1] // multisample
            if need_alpha:
                captured_rgb = captured_rgb * captured_alpha
            captured_alpha = captured_alpha.reshape(h, multisample, w, multisample, 1).mean(axis=1).mean(axis=2)
            captured_rgb = captured_rgb.reshape(h, multisample, w, multisample, 3).mean(axis=1).mean(axis=2)
            if need_alpha:
                captured_rgb /= (captured_alpha + 1e-8)
        if need_alpha:
            captured_rgb = np.concatenate([captured_rgb, captured_alpha], axis=2)
        captured_rgb = (captured_rgb * 255).astype(np.uint8)
        if save_path is not None:
            image.write(captured_rgb, save_path)

        return captured_rgb

    def render_opengl_animation(self, multisample: int = 2):
        """Render an animation sequence of RGB images using OpenGL. GUI is required.

        This function creates an OpenGL rendering engine, renders the scene state at each frame,
        and captures the screen buffer. The RGB values are returned as a floating point array.
        The OpenGL window is destroyed after rendering.

        Args:
            multisample (int): Render quality/samples in OpenGL. Higher values give better quality but
                             slower rendering. Default is 2.

        Returns:
            Generator[Tuple[int, np.ndarray]]: A generator that yields each frame's RGB image.
        """
        saved_intrinsic = copy.copy(self.camera_intrinsic)
        self.camera_intrinsic = self.camera_intrinsic.scale(multisample)
        gl_engine = self._build_gl_engine("_", False)
        engine = gl_engine.engine

        t_start, t_end = self.animator.get_range()
        for t_cur in range(t_start, t_end + 1):
            self.animator.set_frame(t_cur)
            self._update_gl_engine(gl_engine)
            engine.poll_events()
            engine.update_renderer()
            captured_rgb = np.asarray(engine.capture_screen_float_buffer(do_render=True))
            if multisample != 1:
                h = captured_rgb.shape[0] // multisample
                w = captured_rgb.shape[1] // multisample
                captured_rgb = captured_rgb.reshape(h, multisample, w, multisample, 3).mean(axis=1).mean(axis=2)
            captured_rgb = (captured_rgb * 255).astype(np.uint8)
            yield t_cur, captured_rgb

        engine.destroy_window()
        self.camera_intrinsic = saved_intrinsic

    def render_filament(self):
        """Render a single RGB image using Filament. GUI is NOT required.
        
        This function creates a Filament rendering engine, renders the current scene state,
        and captures the screen buffer. The RGB values are returned as a floating point array.

        Returns:
            np.ndarray: Rendered RGB image as a numpy array in uint8 format.
        """
        # Cache DISPLAY environment (don't use)
        #   My guess: whenever you create filament (off-screen or on-screen), EGL's context will be cached using
        # the current DISPLAY variable. Hence if you first do off-screen, then do on-screen, error will happen.
        #   Solution: Just don't use this...
        # x11_environ = None
        # if 'DISPLAY' in os.environ:
        #     x11_environ = os.environ['DISPLAY']
        #     del os.environ['DISPLAY']
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.camera_intrinsic.w, self.camera_intrinsic.h, "")
        self._build_filament_engine(renderer)
        img = renderer.render_to_image()
        # if x11_environ is not None:
        #     os.environ['DISPLAY'] = x11_environ
        return np.array(img)

    def render_filament_animation(self):
        """Render an animation sequence using Filament renderer. GUI is NOT required.

        This function creates a Filament offscreen renderer, renders each frame in the animation
        sequence based on the current animator settings, and yields each rendered frame along
        with its timestamp.

        The renderer uses the current camera intrinsics and scene settings to render high-quality
        frames without requiring a GUI window.

        Returns:
            Generator[Tuple[int, np.ndarray]]: A generator that yields tuples containing:
                - int: Current frame number
                - np.ndarray: RGB image of the rendered frame as a numpy array in uint8 format
        """
        # Create offscreen renderer with current camera resolution
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.camera_intrinsic.w, self.camera_intrinsic.h, "")
        
        # Configure renderer with scene settings
        self._build_filament_engine(renderer)

        # Get animation frame range
        t_start, t_end = self.animator.get_range()
        
        # Render each frame in sequence
        for t_cur in range(t_start, t_end + 1):
            # Update scene to current animation frame
            self.animator.set_frame(t_cur)
            self._update_filament_engine(renderer)
            
            # Render frame and convert to numpy array
            img = renderer.render_to_image()
            yield t_cur, np.array(img)

    def render_pyrender(self):
        """Render a single RGB image of mesh objects using Pyrender. GUI is NOT required.
        
        This function creates a Pyrender rendering engine, renders the current scene state,
        and captures the screen buffer. pyrender seems to have a better support for EGL.

        Returns:
            np.ndarray: Rendered RGB image as a numpy array in uint8 format.
        """
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        import pyrender
        import trimesh

        pr_scene = pyrender.Scene(
            ambient_light=[self.ambient_color[-1]] * 3,
            bg_color=self.ambient_color[:3]
        )
        for obj_uuid, obj_data in self.objects.items():
            if isinstance(obj_data.geom, o3d.geometry.TriangleMesh):
                if self.viewport_shading == 'NORMAL':
                    obj_data.geom.compute_triangle_normals(normalized=True)
                    pr_scene.add(pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(
                            vertices=np.asarray(obj_data.geom.vertices),
                            faces=np.asarray(obj_data.geom.triangles),
                            face_colors=np.asarray(obj_data.geom.triangle_normals) * 0.5 + 0.5), smooth=False),
                        name=obj_uuid, pose=obj_data.pose.matrix)
                else:
                    pr_scene.add(pyrender.Mesh.from_trimesh(
                        trimesh.Trimesh(
                            vertices=np.asarray(obj_data.geom.vertices),
                            faces=np.asarray(obj_data.geom.triangles),
                            vertex_colors=np.asarray(obj_data.geom.vertex_colors))),
                        name=obj_uuid, pose=obj_data.pose.matrix)

        cam_pose = self.camera_pose @ Isometry.from_axis_angle('+X', 180.0)
        pr_scene.add(
            pyrender.IntrinsicsCamera(
                self.camera_intrinsic.fx, self.camera_intrinsic.fy,
                self.camera_intrinsic.cx, self.camera_intrinsic.cy,
            ), pose=cam_pose.matrix
        )

        # Cache the renderer as context creation is slow.
        if self.cached_pyrenderer is None or self.cached_pyrenderer.viewport_height != self.camera_intrinsic.h or \
                self.cached_pyrenderer.viewport_width != self.camera_intrinsic.w:
            self.cached_pyrenderer = pyrender.OffscreenRenderer(
                self.camera_intrinsic.w, self.camera_intrinsic.h)

        render_flag = pyrender.RenderFlags.FLAT
        if not self.backface_culling:
            render_flag |= pyrender.RenderFlags.SKIP_CULL_FACES

        return self.cached_pyrenderer.render(pr_scene, flags=render_flag)[0]


class BaseTheme:
    """Base class for scene rendering themes.
    
    This class serves as a base for different rendering themes that can be applied to scenes.
    It provides common functionality for theme initialization and application.
    """

    def __init__(self, info: str):
        """Initialize the theme with descriptive info text.

        Args:
            info (str): Description of the theme and its intended usage
        """
        print(textwrap.dedent(info))

    def apply_to(self, scene: Scene):
        """Apply this theme's settings to the given scene.

        Args:
            scene (Scene): The scene to apply theme settings to

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError

    @classmethod
    def determine_sun_iso(cls, scene: Scene, back_deg: float, right_deg: float) -> Isometry:
        """Calculate sun light direction based on camera orientation and tilt angles.

        This method determines the sun light direction by:
        1. Creating initial light orientation aligned with scene's up axis
        2. Projecting camera direction onto the plane perpendicular to up axis
        3. Applying back and right tilts relative to camera orientation

        Args:
            scene (Scene): Scene containing camera and up axis information
            back_deg (float): Backward tilt angle in degrees relative to camera direction
            right_deg (float): Rightward tilt angle in degrees relative to camera direction

        Returns:
            Isometry: Isometry transform representing the sun light direction
        """
        # Create initial light orientation aligned with up axis
        light_iso = Isometry.look_at(np.zeros(3), Isometry._str_to_axis(scene.up_axis))
        
        # Get projection directions and camera orientation
        proj_dir = light_iso.matrix[:3, :2]  # First two columns define projection plane
        cam_dir = scene.camera_pose.matrix[:3, [0, 2]]  # Camera x and z axes

        cam_uv = proj_dir @ np.linalg.lstsq(proj_dir, cam_dir, rcond=None)[0] 
        light_dir = Isometry.from_axis_angle(cam_uv[:, 0], back_deg) @ \
                    Isometry.from_axis_angle(cam_uv[:, 1], -right_deg) @ light_iso
        return light_dir

class ThemeAngela(BaseTheme):
    """A theme class optimized for rendering indoor room scenes in Blender.
    
    This theme provides a clean, indoor lighting setup with customizable base colors
    and sun light parameters. It disables ambient occlusion and sets up transparent 
    backgrounds for better composition flexibility.

    Attributes:
        Color: Inner class defining preset color schemes
            MILD_AQUA: Light blue color (R=0.555, G=0.769, B=0.926)
            PAPER_WOOD: Reserved for paper/wood color
            GRAY: Reserved for gray color
    """
    class Color:
        MILD_AQUA = (0.555, 0.769, 0.926)
        PAPER_WOOD = ()  # Reserved for future use
        GRAY = ()        # Reserved for future use

    def __init__(self, base_color: tuple = Color.MILD_AQUA, smooth_shading: bool = False,
                 sun_tilt_right: float = 0.0, sun_tilt_back: float = 20.0,
                 sun_energy: float = 1.5, sun_angle: float = 45.7):
        """Initialize the Angela theme with customizable parameters.

        Args:
            base_color (tuple): RGB color tuple for base material color. Defaults to MILD_AQUA.
            smooth_shading (bool): Whether to enable smooth shading. Defaults to False.
            sun_tilt_right (float): Sun light tilt angle rightward in degrees. Defaults to 0.0.
            sun_tilt_back (float): Sun light tilt angle backward in degrees. Defaults to 20.0.
            sun_energy (float): Sun light intensity/energy value. Defaults to 1.5.
            sun_angle (float): Sun light angle in degrees. Defaults to 45.7.
        """
        super().__init__('''
        Indoor Room Theme v1.0 (target: blender)
            This is ideal for rendering indoor rooms. Remember to crop the ceiling!
        ''')
        self.base_color = copy.deepcopy(base_color)
        self.smooth_shading = smooth_shading
        self.sun_tilt_right = sun_tilt_right
        self.sun_tilt_back = sun_tilt_back
        self.sun_energy = sun_energy
        self.sun_angle = sun_angle

    def apply_to(self, scene: Scene):
        """Apply the Angela theme settings to a scene.

        Configures scene lighting, materials, and render settings according to the theme:
        - Clears existing lights and sets up directional sun lighting
        - Enables transparent background
        - Applies base color and smooth shading to objects
        - Disables ambient occlusion
        - Sets standard view transform in Blender

        Args:
            scene (Scene): The scene to apply the theme settings to

        Returns:
            None
        """
        # Clear existing scene lighting setup
        scene.lights.clear()
        scene.film_transparent = True
        scene.viewport_shading = 'LIT'
        scene.remove_object('auto_plane', non_exist_ok=True)
        scene.env_map = None

        # Configure material settings for all objects
        for o in scene.objects.values():
            if self.base_color is not None:
                o.attributes['uniform_color'] = [*self.base_color, 1.0]  # Add alpha=1.0
            o.attributes['smooth_shading'] = self.smooth_shading
            o.attributes['material.ao'] = {"on": False}  # Disable ambient occlusion
        
        # Set ambient lighting color (white with 0.6 intensity)
        scene.ambient_color = (1.0, 1.0, 1.0, 0.6)

        # Configure and add sun light
        light_iso = self.determine_sun_iso(scene, self.sun_tilt_back, self.sun_tilt_right)
        scene.add_light_sun(
            light_dir=light_iso.q.q, light_energy=self.sun_energy, angle=self.sun_angle / 180.0 * np.pi)
            
        # Set Blender-specific render settings
        scene.additional_blender_commands = "bpy.data.scenes[0].view_settings.view_transform = 'Standard'"


class ThemeNormalShadow(BaseTheme):
    """A theme class that renders objects with it's normals and shadow casting in Blender.

    Args:
        smooth_shading (bool): Whether to use smooth shading for objects. Defaults to False.
        sun_tilt_right (float): Sun tilt angle to the right in degrees. Defaults to 0.0.
        sun_tilt_back (float): Sun tilt angle backwards in degrees. Defaults to 0.0. 
        sun_energy (float): Intensity of the sun light. Defaults to 1.5.
        sun_angle (float): Angular size of the sun in degrees. Defaults to 20.0.
        normal_saturation (float): Saturation of normal map colors. Defaults to 1.5.
    """
    def __init__(self, smooth_shading: bool = False,
                 sun_tilt_right: float = 0.0, sun_tilt_back: float = 0.0,
                 sun_energy: float = 1.5, sun_angle: float = 20.0,
                 normal_saturation: float = 1.5):
        super().__init__('''
        Normal rendering with shadow casted (for blender).
        ''')
        self.smooth_shading = smooth_shading
        self.sun_tilt_right = sun_tilt_right
        self.sun_tilt_back = sun_tilt_back
        self.sun_energy = sun_energy
        self.sun_angle = sun_angle
        self.normal_saturation = normal_saturation  # blender default is 1.0

    def apply_to(self, scene: Scene) -> None:
        """Apply the normal shadow theme settings to a scene.

        Configures scene lighting and materials to show object normals with shadows:
        - Clears existing lights and sets up directional sun lighting
        - Enables transparent background
        - Configures normal shading for objects
        - Adds a shadow-catching ground plane
        - Sets standard view transform in Blender

        Args:
            scene (Scene): The scene to apply the theme settings to

        Returns:
            None
        """
        # Clear existing scene lighting setup
        scene.lights.clear()
        scene.film_transparent = True
        scene.viewport_shading = 'LIT'
        scene.remove_object('auto_plane', non_exist_ok=True)
        scene.env_map = None

        # Configure material settings for all objects
        for o in scene.objects.values():
            o.attributes['smooth_shading'] = self.smooth_shading
            o.attributes['material.ao'] = {"on": False}  # Disable ambient occlusion
            o.attributes['material.normal'] = {"on": True, "saturation": self.normal_saturation} # normal shading

        # Add shadow-catching ground plane
        scene.auto_plane(dist_ratio=0.0)
        scene.objects['auto_plane'].attributes['cycles.is_shadow_catcher'] = True
        
        # Set ambient lighting color (white with 0.6 intensity)
        scene.ambient_color = (1.0, 1.0, 1.0, 0.6)

        # Configure and add sun light
        light_iso = self.determine_sun_iso(scene, self.sun_tilt_back, self.sun_tilt_right)
        scene.add_light_sun(
            light_dir=light_iso.q.q, light_energy=self.sun_energy, angle=self.sun_angle / 180.0 * np.pi)
            
        # Set Blender-specific render settings
        scene.additional_blender_commands = "bpy.data.scenes[0].view_settings.view_transform = 'Standard'"


class ThemeDiffuseShadow(BaseTheme):
    """A theme that creates diffuse-looking objects with shadows.
    
    Args:
        base_color (Optional[List[float]]): RGB base color for objects. If None, keeps original colors.
            Default is None.
        smooth_shading (bool): Whether to enable smooth shading on objects. Default is False.
        sun_tilt_right (float): Sun tilt angle to the right in degrees. Default is 0.0.
        sun_tilt_back (float): Sun tilt angle backwards in degrees. Default is 0.0.
        sun_energy (float): Sun light intensity/energy. Default is 4.5.
        sun_angle (float): Sun angle in degrees. Default is 40.0.
    """
    def __init__(self, base_color: Optional[List[float]] = None, smooth_shading: bool = False,
                 sun_tilt_right: float = 0.0, sun_tilt_back: float = 0.0,
                 sun_energy: float = 4.5, sun_angle: float = 40.0):
        super().__init__('''
        Diffuse-looking with shadow.
        ''')
        self.base_color = copy.deepcopy(base_color)
        self.smooth_shading = smooth_shading
        self.sun_tilt_right = sun_tilt_right
        self.sun_tilt_back = sun_tilt_back
        self.sun_energy = sun_energy
        self.sun_angle = sun_angle

    def apply_to(self, scene: Scene) -> None:
        """Apply the diffuse shadow theme settings to a scene.

        Configures scene lighting and materials to create a diffuse appearance with shadows:
        - Clears existing lights and sets up directional sun lighting
        - Enables transparent background
        - Disables ambient occlusion
        - Adds a shadow-catching ground plane
        - Allows customization of object colors and sun light parameters

        Args:
            scene (Scene): The scene to apply the theme settings to.

        Returns:
            None
        """
        # Clear existing scene lighting setup
        scene.lights.clear()
        scene.film_transparent = True
        scene.viewport_shading = 'LIT'
        scene.remove_object('auto_plane', non_exist_ok=True)
        scene.env_map = None

        # Configure material settings for all objects
        for o in scene.objects.values():
            if self.base_color is not None:
                o.attributes['uniform_color'] = [*self.base_color, 1.0]
            o.attributes['smooth_shading'] = self.smooth_shading
            o.attributes['material.ao'] = {"on": False}  # Disable ambient occlusion

        # Add shadow-catching ground plane
        scene.auto_plane(dist_ratio=0.0)
        
        # Set ambient lighting color (white with 0.0 intensity)
        scene.ambient_color = (1.0, 1.0, 1.0, 0.0)

        # Configure and add sun light
        light_iso = self.determine_sun_iso(scene, self.sun_tilt_back, self.sun_tilt_right)
        scene.add_light_sun(
            light_dir=light_iso.q.q, light_energy=self.sun_energy, angle=self.sun_angle / 180.0 * np.pi)
            
        # Set Blender-specific render settings
        scene.additional_blender_commands = "bpy.data.scenes[0].view_settings.view_transform = 'Standard'"


class ThemeNKSR(BaseTheme):
    """A rendering theme class used for NKSR v1.0 that configures scenes for both Blender and Filament renderers.
    
    This theme applies a consistent look with ambient occlusion, optional ground plane shadows,
    and environment lighting using a rainforest HDR map.

    Args:
        base_color (list[float], optional): RGB base color values between 0-1. Defaults to [0.8, 0.8, 0.8].
        smooth_shading (bool, optional): Whether to enable smooth shading on objects. Defaults to False.
        need_plane (bool, optional): Whether to add a shadow-catching ground plane. Defaults to False.
    """
    def __init__(self, base_color: list[float] = [0.8, 0.8, 0.8], smooth_shading: bool = False, need_plane: bool = False):
        super().__init__('''
        Rendering theme used for NKSR v1.0 (target: blender & filament).
        ''')
        self.base_color = copy.deepcopy(base_color)
        self.smooth_shading = smooth_shading
        self.need_plane = need_plane

    def apply_to(self, scene: Scene) -> None:
        """Apply the NKSR theme settings to a scene.

        Configures scene lighting and materials with:
        - Transparent background
        - LIT viewport shading
        - Ambient occlusion on all objects
        - Optional shadow-catching ground plane
        - Rainforest HDR environment map
        - Filmic view transform in Blender

        Args:
            scene (Scene): The scene to apply the theme settings to.

        Returns:
            None
        """
        # Clear existing scene lighting setup
        scene.lights.clear()
        scene.film_transparent = True
        scene.viewport_shading = 'LIT'
        scene.remove_object('auto_plane', non_exist_ok=True)

        # Configure material settings for all objects
        for o in scene.objects.values():
            if self.base_color is not None:
                o.attributes['uniform_color'] = [*self.base_color, 1.0]
            o.attributes['material.ao'] = {"on": True, "gamma": 2.0, "strength": 0.5}
            o.attributes['smooth_shading'] = self.smooth_shading

        # Add optional shadow-catching ground plane
        if self.need_plane:
            scene.auto_plane(dist_ratio=0.0)
            scene.objects['auto_plane'].attributes['cycles.is_shadow_catcher'] = True

        # Configure scene lighting
        scene.ambient_color = (1.0, 1.0, 1.0, 1.0)
        scene.env_map = image.read(get_assets_path() / "envmaps" / "rainforest.hdr")
        scene.set_filament_env_map("pycg-rainforest")
        
        # Set Blender-specific render settings
        scene.additional_blender_commands = "bpy.data.scenes[0].view_settings.view_transform = 'Filmic'"


def multiview_image(geoms: list, width: int = 256, height: int = 256, up_axis=None, pitch_angle=20.0,
                    viewport_shading='NORMAL', backend='filament'):
    """
    Headless render a multiview image of geometry list, mainly used for training visualization.
    :param geoms: list of geometry, could be annotated.
    :param width: width of each image
    :param height: height of each image
    :param up_axis: up axis of scene
    :param viewport_shading: shading used.
    :param backend: opengl, filament, or pyrender
        --privileged allows us to use VSCode inside docker, but will let pyrender fail :(
    :return: an image.
    """
    scene = Scene()
    for geom in geoms:
        scene.add_object(geom=geom)
    scene.viewport_shading = viewport_shading
    multiview_pics = []
    for view_angle in [0.0, 90.0, 180.0, 270.0]:
        scene.quick_camera(w=width, h=height, fov=45.0, up_axis=up_axis,
                           fill_percent=0.7, plane_angle=view_angle + 45.0,
                           pitch_angle=pitch_angle)
        if backend == 'opengl':
            my_pic = scene.render_opengl()
        elif backend == 'filament':
            my_pic = scene.render_filament()
        elif backend == 'pyrender':
            my_pic = scene.render_pyrender()
        else:
            raise NotImplementedError
        multiview_pics.append(my_pic)
    return image.hlayout_images(multiview_pics)
