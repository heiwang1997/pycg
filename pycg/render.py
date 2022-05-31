import shutil
import cv2
import json
import os
import tempfile
import open3d as o3d
import numpy as np
from .animation import SceneAnimator
from .isometry import Isometry
import pycg.vis as vis
import pycg.image as image
import pycg.blender_client as blender
from pathlib import Path
from pyquaternion import Quaternion
import uuid
import copy
import pickle
import functools
from collections import defaultdict
import matplotlib.cm


class GLEngineWrapper:
    def __init__(self, engine):
        self.engine = engine
        self.displayed_geometries = {}


class VisualizerManager:
    TITLE_HEIGHT_FIX = 24

    def __init__(self):
        self.scenes = []
        self.scene_titles = []
        self.pose_change_callbacks = []
        self.reset()

    def reset(self):
        self.scenes = []
        self.scene_titles = []
        self.pose_change_callbacks = []

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
        r_min = data.min()
        r_max = data.max()
        pos = np.linspace(r_min, r_max, 200)

        kernel = stats.gaussian_kde(data)
        return r_min, r_max, kernel(pos)

    def run(self, use_new_api=False, key_bindings=None, max_cols=-1, scale=1.0):
        assert len(self.scenes) > 0, "No scene to show."
        if max_cols == -1:
            max_cols = len(self.scenes)

        if scale != 1.0:
            # If scale is applied, modify the cameras of the scene to fit scale.
            for scene in self.scenes:
                scene.camera_intrinsic = scene.camera_intrinsic.scale(scale)

        if use_new_api:
            # Please refer to Open3D/python/open3d/visualization/draw.py
            from open3d.visualization import gui

            gui.Application.instance.initialize()

            all_windows = []
            for scene_idx, (cur_scene, cur_title) in enumerate(zip(self.scenes, self.scene_titles)):
                if cur_title is None:
                    cur_title = f"Scene-{scene_idx:03d}"

                c_param = o3d.visualization.O3DVisualizer.ConstructParams()
                c_param.set_scene_widget(gui.SceneWidget() if len(self.scenes) > 1 else None)

                # Analyse and add histogram window if necessary...
                all_hists = []
                for mesh_name, mesh_obj in cur_scene.objects.items():
                    if mesh_obj.annotations is None:
                        continue
                    if isinstance(mesh_obj.geom, o3d.geometry.PointCloud):
                        hist_data = np.asarray(mesh_obj.geom.points)
                        h_min, h_max, h_values = self.convert_histogram(mesh_obj.annotations[0])
                        hist_widget = gui.Histogram(20, 50 + len(all_hists) * 100, 400, 100, f"PC-{mesh_name[:6]}")
                        hist_widget.set_value(h_min, h_max, h_values, {
                            'viridis': gui.Color.Colormap.VIRIDIS, 'plasma': gui.Color.Colormap.PLASMA,
                            'jet': gui.Color.Colormap.JET, 'spectral': gui.Color.Colormap.SPECTRAL
                        }[mesh_obj.annotations[1]])
                        all_hists.append(hist_widget)
                c_param.other_widgets = all_hists

                w = o3d.visualization.O3DVisualizer(title=cur_title, width=cur_scene.camera_intrinsic.w,
                                                    height=cur_scene.camera_intrinsic.h + self.TITLE_HEIGHT_FIX,
                                                    param=c_param)
                all_windows.append(w)
                cur_scene._build_filament_engine(w)
                w.show_settings = False
                w.reset_camera_to_default()     # important, because it correctly set up scene bounds
                w.setup_camera(cur_scene.camera_intrinsic.to_open3d_intrinsic(), cur_scene.camera_pose.inv().matrix)

                def on_key(e, idx):
                    if e.type == gui.KeyEvent.DOWN:
                        if e.key == gui.KeyName.X:
                            raise KeyboardInterrupt
                        elif e.key == gui.KeyName.Q:
                            for w_cur in all_windows:
                                w_cur.close()
                        elif e.key == gui.KeyName.S:
                            all_windows[idx].show_settings = not all_windows[idx].show_settings
                        elif e.key == gui.KeyName.A:
                            all_windows[idx].show_axes = not all_windows[idx].show_axes
                        elif e.key == gui.KeyName.H:
                            for hst in all_hists:
                                hst.visible = not hst.visible
                        elif e.key == gui.KeyName.R:
                            if self.pose_change_callbacks[idx] is not None:
                                # When output current pose, need to change back this.
                                if scale != 1.0:
                                    self.scenes[idx].camera_intrinsic = self.scenes[idx].camera_intrinsic.scale(1.0 / scale)
                                self.pose_change_callbacks[idx](all_windows[idx])
                                if scale != 1.0:
                                    self.scenes[idx].camera_intrinsic = self.scenes[idx].camera_intrinsic.scale(scale)
                        elif e.key == gui.KeyName.L:
                            print("Light information written to scene.")
                            sun_dir = w.scene.scene.get_sun_light_direction()
                            cur_scene.lights["default"].pose = Isometry.look_at(source=np.zeros((3, )),
                                                                                target=-sun_dir)
                        elif e.key == gui.KeyName.EQUALS:       # + point size
                            target_size = all_windows[idx].point_size + 1
                            for w_cur in all_windows:
                                w_cur.point_size = target_size
                                w_cur.scene_widget.force_redraw()
                                w_cur.post_redraw()
                        elif e.key == gui.KeyName.MINUS:       # - point size
                            target_size = all_windows[idx].point_size - 1
                            for w_cur in all_windows:
                                w_cur.point_size = target_size
                                w_cur.scene_widget.force_redraw()
                                w_cur.post_redraw()
                        elif e.key == gui.KeyName.LEFT_BRACKET:
                            pose, intr = self.get_window_camera_info(all_windows[idx])
                            intr.fov_x += np.deg2rad(5.0)
                            all_windows[idx].setup_camera(intr.to_open3d_intrinsic(), pose.inv().matrix)
                        elif e.key == gui.KeyName.RIGHT_BRACKET:
                            pose, intr = self.get_window_camera_info(all_windows[idx])
                            intr.fov_x -= np.deg2rad(5.0)
                            all_windows[idx].setup_camera(intr.to_open3d_intrinsic(), pose.inv().matrix)
                    return 0

                # You can also do w.scene_widget.set_on_key, but on panels it won't respond.
                w.set_on_key(functools.partial(on_key, idx=scene_idx))

                if len(self.scenes) > 1:
                    # Sync camera.
                    def on_mouse(e, idx):
                        if e.type == gui.MouseEvent.DRAG or e.type == gui.MouseEvent.WHEEL:
                            cur_w = all_windows[idx]
                            cam_pose, cam_intrinsic = self.get_window_camera_info(cur_w)

                            for w_others in all_windows:
                                if id(w_others) != id(cur_w):
                                    w_others.setup_camera(cam_intrinsic.to_open3d_intrinsic(), cam_pose.inv().matrix)
                                    w_others.scene_widget.force_redraw()
                                    w_others.post_redraw()
                        return 0
                    w.scene_widget.set_on_mouse(functools.partial(on_mouse, idx=scene_idx))

                gui.Application.instance.add_window(w)

                if len(self.scenes) > 1:
                    w.os_frame = gui.Rect(w.os_frame.width * scene_idx, 0, w.os_frame.width, w.os_frame.height)

            gui.Application.instance.run()

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
                    if scale == 1.0:
                        new_callback = self.pose_change_callbacks[scene_idx]
                    else:
                        def new_callback(vis):
                            assert False, "Fuck this is buggy. Don't use."
                            if scale != 1.0:
                                cur_scene.camera_intrinsic = cur_scene.camera_intrinsic.scale(1.0 / scale)
                            self.pose_change_callbacks[scene_idx](vis)
                            if scale != 1.0:
                                cur_scene.camera_intrinsic = cur_scene.camera_intrinsic.scale(scale)

                    engine.register_key_callback(key=ord("R"), callback_func=new_callback)

                # if add_ruler:
                #     ruler_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                #     engine.add_geometry(ruler_frame_mesh, reset_bounding_box=False)

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
                                camera_intrinsic.get_pinhole_camera_param(camera_pose, fix_bug=True)
                            )
                            eng.get_render_option().point_size = vis.get_render_option().point_size
                            # self.gl_render_options.save_to_json("/tmp/ro.json")
                            # engine.get_render_option().load_from_json("/tmp/ro.json")

                    engine.register_view_refresh_callback(on_view_refresh)

                all_engines.append(engine)

            if len(all_engines) < 2:
                # For one window, use the faster solution.
                all_engines[0].run()
            else:
                while True:
                    can_stop = False
                    for eng in all_engines:
                        if not eng.poll_events():
                            can_stop = True
                        eng.update_renderer()
                    if can_stop:
                        break

            for eng in all_engines:
                eng.destroy_window()

        if scale != 1.0:
            # If scale is applied, modify the cameras of the scene to fit scale.
            for scene in self.scenes:
                scene.camera_intrinsic = scene.camera_intrinsic.scale(1.0 / scale)

        self.reset()


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


class CameraIntrinsic:
    def __init__(self, w, h, fx, fy, cx, cy):
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @staticmethod
    def from_fov(w, h, fov_x):
        new_intr = CameraIntrinsic(w, h, None, None, None, None)
        new_intr.fov_x = fov_x
        return new_intr

    @property
    def fov_x(self):
        return 2 * np.arctan2(self.w, 2 * self.fx)

    @fov_x.setter
    def fov_x(self, value):
        if not (np.deg2rad(5.) < value < np.deg2rad(170.)):
            print(f"Error setting fov_x, target degree is {np.rad2deg(value)}")
            return
        self.fx = self.fy = self.w / 2.0 / np.tan(value / 2.0)
        self.cx = self.w / 2.
        self.cy = self.h / 2.

    @property
    def blender_attributes(self):
        # resX, resY, shiftX, shiftY, angleX
        return self.w, self.h, self.cx / self.w - 0.5, self.cy / self.h - 0.5, np.arctan(self.w / 2.0 / self.fx) * 2.0

    USD_FIX_AW = 36

    def set_usd_attributes(self, cam_prim):
        from pxr import Sdf

        usd_fix_ah = self.USD_FIX_AW / self.w * self.h
        cam_prim.GetHorizontalApertureAttr().Set(self.USD_FIX_AW)
        cam_prim.GetVerticalApertureAttr().Set(usd_fix_ah)
        cam_prim.GetFocalLengthAttr().Set(self.USD_FIX_AW / self.w * self.fx)
        cam_prim.GetHorizontalApertureOffsetAttr().Set(self.USD_FIX_AW / self.w * self.cx - self.USD_FIX_AW / 2.)
        cam_prim.GetVerticalApertureOffsetAttr().Set(usd_fix_ah / self.h * self.cy - usd_fix_ah / 2.)

        # These could not be properly saved.
        cam_prim.GetPrim().CreateAttribute('render_width', Sdf.ValueTypeNames.Int).Set(self.w)
        cam_prim.GetPrim().CreateAttribute('render_height', Sdf.ValueTypeNames.Int).Set(self.h)

    @staticmethod
    def from_open3d_intrinsic(intr: o3d.camera.PinholeCameraIntrinsic):
        return CameraIntrinsic(intr.width, intr.height, intr.get_focal_length()[0], intr.get_focal_length()[1],
                               intr.get_principal_point()[0] + 0.5, intr.get_principal_point()[1] + 0.5)

    def to_open3d_intrinsic(self, fix_bug=False):
        intr = o3d.camera.PinholeCameraIntrinsic()
        delta = 0.5 if fix_bug else 0.0
        intr.set_intrinsics(width=self.w, height=self.h, fx=self.fx, fy=self.fy, cx=self.cx - delta, cy=self.cy - delta)
        return intr

    @property
    def K(self):
        return np.asarray([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])

    @K.setter
    def K(self, val):
        self.fx = val[0, 0]
        self.fy = val[1, 1]
        self.cx = val[0, 2]
        self.cy = val[1, 2]

    @property
    def inv_K(self):
        return np.asarray([
            [1 / self.fx, 0.0, -self.cx / self.fx],
            [0.0, 1 / self.fy, -self.cy / self.fy],
            [0.0, 0.0, 1.0]
        ])

    def get_pinhole_camera_param(self, pose: Isometry, fix_bug=False):
        param = o3d.camera.PinholeCameraParameters()
        pose_mat = np.linalg.inv(pose.matrix)
        param.extrinsic = pose_mat
        param.intrinsic = self.to_open3d_intrinsic(fix_bug=fix_bug)
        return param

    def scale(self, factor, quantize=True):
        return CameraIntrinsic(self.w * factor if not quantize else int(self.w * factor),
                               self.h * factor if not quantize else int(self.h * factor),
                               self.fx * factor,
                               self.fy * factor,
                               self.cx * factor,
                               self.cy * factor)


class SceneObject:
    def __init__(self, geom, pose: Isometry = Isometry(), attributes: dict = None):
        if attributes is None:
            attributes = {}
        if isinstance(geom, vis.AnnotatedGeometry):
            self.geom = geom.geom
            self.annotations = geom.annotations
            attributes.update(geom.attributes)
        else:
            self.geom, self.annotations = geom, None
        self.pose = pose
        self.visible = True
        """
        List of Attributes:
            - alpha: 0.0-1.0
            - material.specular: 0.0-1.0
            - material.metallic: 0.0-1.0
            - material.roughness: 0.0-1.0
            - material.checker: {"on": True/False, "color_a": (rgba), "color_b": (rgbd), "scale": 1.0}
            - material.normal: {"on": True/False}
            - cycles_visibility.camera: True/False
            - cycles_visibility.shadow: True/False
            - cycles_visibility.diffuse: True/False
            - cycles.is_shadow_catcher: True/False
            - smooth_shading: True/False
            - uniform_color: (rgba) if set then ignore mesh vertex color.
        """
        self.attributes = attributes

    def get_extent(self):
        # Note this is only a rough extent!
        bound_points = self.geom.get_axis_aligned_bounding_box().get_box_points()
        bound_points = self.pose @ np.asarray(bound_points)
        return np.min(bound_points, axis=0), np.max(bound_points, axis=0)

    def get_transformed(self):
        transformed_geom = copy.deepcopy(self.geom)
        transformed_geom.transform(self.pose.matrix)
        return transformed_geom

    def add_usd_prim(self, stage, prim_path):
        # https://raw.githubusercontent.com/NVIDIAGameWorks/kaolin/master/kaolin/io/usd.py
        from pxr import UsdGeom, Vt, Gf

        # xform = UsdGeom.Xform.Define(stage, prim_path)
        # xform.AddTransformOp().Set(Gf.Matrix4d(self.pose.matrix.T))

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
        elif isinstance(self.geom, o3d.geometry.PointCloud):
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
    def __init__(self, mtype, pose: Isometry = Isometry(), attributes: dict = None):
        self.type = mtype
        self.pose = pose
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def add_usd_prim(self, stage, prim_path):
        pass


class Scene:
    def __init__(self, cam_path: str = None):
        self.objects = {}
        self.lights = {}
        self.output = ["rgb"]

        # camera_pose = camera_base @ rel_camera_pose
        self.camera_base = Isometry()
        self.relative_camera_pose = Isometry()      # This only allows for quantized animation.

        self.camera_intrinsic = CameraIntrinsic(300, 300, 150, 150, 150, 150)
        self.cam_path = Path(cam_path) if cam_path else cam_path
        # These are old-api specific settings
        self.gl_render_options = o3d.visualization.RenderOption()
        self.gl_render_options.mesh_show_back_face = True
        self.gl_render_options.point_show_normal = False
        try:
            self.load_camera()
        except FileNotFoundError:
            pass
        except AttributeError:
            pass
        # These are old/new shared settings
        self.ambient_color = (1.0, 1.0, 1.0, 1.0)
        self.film_transparent = False
        self.point_size = 5.0
        self.viewport_shading = 'LIT'       # Supported: LIT, UNLIT, NORMAL
        self.up_axis = '+Y'

        # Add a default light.
        self.add_light_sun(name="default")
        self.animator = SceneAnimator(self)

    @property
    def camera_pose(self):
        return self.camera_base @ self.relative_camera_pose

    @camera_pose.setter
    def camera_pose(self, value):
        self.relative_camera_pose = self.camera_base.inv() @ value

    def export(self, path):
        """
        Export using USD file.
            To render in omniverse
        :return:
        """
        from pxr import Usd, UsdGeom, Vt, Gf

        if self.camera_intrinsic.w < self.camera_intrinsic.h:
            print("Your camera is vertical. Rendering in omniverse may not re-create the same look."
                  "Consider rotate it by 90 degrees.")
            # Otherwise, when importing to OV, select '/camera/main' in the viewport, uncheck 'fit viewport',
            #   and manually change the resolution to desired ratio in `render movie' window.

        stage = Usd.Stage.CreateNew(path)
        if self.up_axis == '+Y':
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        else:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.SetMetadata('comment', 'Exported from pycg.Scene')
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(0.0)

        # Add Scene objects.
        for obj_name, obj in self.objects.items():
            obj.add_usd_prim(stage, f"/scene/{obj_name}")

        # Setup camera
        usd_relative_camera = Isometry.from_axis_angle('+X', degrees=180.0)
        cam_base_xform = UsdGeom.Xform.Define(stage, "/camera")
        cam_base_xform.AddTransformOp().Set(Gf.Matrix4d(self.camera_base.matrix.T))
        cam_prim = UsdGeom.Camera.Define(stage, "/camera/main")
        cam_prim.AddTransformOp().Set(Gf.Matrix4d(
            (self.relative_camera_pose @ usd_relative_camera).matrix.T))
        self.camera_intrinsic.set_usd_attributes(cam_prim)

        # Setup lights
        for light_name, light in self.lights.items():
            light.add_usd_prim(stage, f"/scene/{light_name}")

        stage.GetRootLayer().Save()

    def load(self, path):
        from pxr import Usd, Vt
        stage = Usd.Stage.Open(path)
        xform = stage.GetPrimAtPath('/hello')
        sphere = stage.GetPrimAtPath('/hello/world')

        print(xform.GetPropertyNames())

    def load_camera(self):
        with self.cam_path.open('rb') as cam_f:
            camera_data = pickle.load(cam_f)
        self.camera_pose = camera_data['extrinsic']
        self.camera_intrinsic = camera_data['intrinsic']

    def save_camera(self):
        self.cam_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cam_path.open('wb') as scene_f:
            pickle.dump({
                'extrinsic': self.camera_pose,
                'intrinsic': self.camera_intrinsic
            }, scene_f)

    def quick_camera(self, pos=None, look_at=None, w=1024, h=768, fov=60.0, up_axis=None,
                     fill_percent=0.5, plane_angle=0.0, force_override=False):
        if self.cam_path is not None and self.cam_path.exists() and not force_override:
            return self

        if up_axis is not None:
            if isinstance(up_axis[0], str):
                # This is just a guess.
                self.up_axis = up_axis
                up_axis = Isometry._str_to_axis(up_axis)
            else:
                up_axis = np.array(up_axis)
        else:
            up_axis = Isometry._str_to_axis(self.up_axis)

        if pos is None or look_at is None:
            cplane_x = np.array([up_axis[1], -up_axis[0], 0.0])
            cplane_y = np.cross(up_axis, cplane_x)
            cplane_y = cplane_y / np.linalg.norm(cplane_y)

            if len(self.objects) == 0:
                print("Warning. Please add objects before using quick-camera!")
                pos = np.array([1.0, 0.0, 0.0])
                look_at = np.array([0.0, 0.0, 0.0])
            else:
                # Auto camera according to Open3D.
                min_extent, max_extent = self.get_scene_extent()
                # self.add_object(vis.wireframe_bbox(min_extent, max_extent))
                view_ratio = np.max(max_extent - min_extent) / fill_percent / 2
                distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
                look_at = (min_extent + max_extent) / 2.
                pos = look_at + distance * (np.cos(plane_angle / 180.0 * np.pi) * cplane_x +
                                            np.sin(plane_angle / 180.0 * np.pi) * cplane_y)

        self.camera_pose = Isometry.look_at(np.asarray(pos), np.asarray(look_at), up_axis)
        self.camera_intrinsic = CameraIntrinsic.from_fov(w, h, np.deg2rad(fov))
        return self

    def add_object(self, geom, name=None, pose=None, attributes=None, return_name=False):
        if name is None:
            new_name = "obj" + str(uuid.uuid1())[:8]
        else:
            assert name not in self.objects.keys()
            new_name = name
        if pose is None:
            pose = Isometry()
        if isinstance(geom, SceneObject):
            new_obj = copy.deepcopy(geom)
        else:
            new_obj = SceneObject(geom, pose, attributes)
        self.objects[new_name] = new_obj
        return self if not return_name else new_name

    def remove_object(self, name):
        del self.objects[name]
        return self

    def set_object_attribute(self, name, kwargs: dict):
        assert name in self.objects.keys()
        self.objects[name].attributes.update(kwargs)
        return self

    def add_light(self, light, name=None, pose=Isometry(), attributes=None):
        if name is None:
            new_name = str(uuid.uuid1())[:8]
        else:
            assert name not in self.lights.keys()
            new_name = name
        if isinstance(light, SceneLight):
            new_light = copy.deepcopy(light)
        else:
            new_light = SceneLight(light, pose, attributes)
        self.lights[new_name] = new_light
        return self

    def remove_light(self, name):
        del self.lights[name]
        return self

    def add_light_sun(self, name=None, light_dir=None, light_energy=5.0, angle=0.1745):
        if light_dir is None:
            if self.up_axis == '+Z':
                light_dir = (1.0, 0.0, 0.0, 0.0)
            elif self.up_axis == '-Z':
                light_dir = (0.0, 1.0, 0.0, 0.0)
            elif self.up_axis == '+Y':
                light_dir = (0.707, -0.707, 0.0, 0.0)
            elif self.up_axis == '-Y':
                light_dir = (0.707, 0.707, 0.0, 0.0)
            elif self.up_axis == '+X':
                light_dir = (0.707, 0.0, 0.707, 0.0)
            elif self.up_axis == '-X':
                light_dir = (0.707, 0.0, -0.707, 0.0)
            else:
                raise NotImplementedError
        return self.add_light('SUN', name, Isometry(t=(0.0, 0.0, 0.0), q=Quaternion(light_dir)),
                              {'energy': light_energy, 'angle': angle})

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

    def auto_plane(self, config=None, dist_ratio=0.1, scale=10.0):
        min_extent, max_extent = self.get_scene_extent()
        scene_center = (min_extent + max_extent) / 2.
        if config is None:
            config = self.up_axis
        else:
            self.up_axis = config       # Guess
        axis_idx = ['X', 'Y', 'Z'].index(config[1])
        symbol_mult = 1 if config[0] == '-' else -1
        plane_normal = np.zeros((3,))
        plane_normal[axis_idx] = -symbol_mult
        plane_center = np.copy(scene_center)
        scene_extent = max_extent - min_extent
        plane_center[axis_idx] += symbol_mult * scene_extent[axis_idx] * (0.5 + dist_ratio)
        scene_extent[axis_idx] = 0.0
        my_plane = vis.plane(plane_center, plane_normal, scale=np.linalg.norm(scene_extent) * scale)
        my_plane.compute_vertex_normals()
        return self.add_object(my_plane, name='auto_plane')

    def get_scene_extent(self):
        all_bbox = [obj.get_extent() for obj in self.objects.values()]
        all_bbox = list(zip(*all_bbox))
        min_extent = np.asarray(all_bbox[0]).min(axis=0)
        max_extent = np.asarray(all_bbox[1]).max(axis=0)
        return min_extent, max_extent

    def center_geometries(self):
        # Compute the center of geometries.
        min_extent, max_extent = self.get_scene_extent()
        center = (min_extent + max_extent) / 2.
        for obj in self.objects.values():
            obj.pose.t -= center
        return self

    def _build_gl_engine(self, window_name, visible, pos=None):
        if pos is None:
            pos = (50, 50)
        engine = o3d.visualization.VisualizerWithKeyCallback()
        warpped_engine = GLEngineWrapper(engine)
        engine.create_window(window_name=window_name,
                             width=self.camera_intrinsic.w,
                             height=self.camera_intrinsic.h,
                             left=pos[0], top=pos[1],
                             visible=visible)
        for mesh_name, mesh_obj in self.objects.items():
            geom = mesh_obj.get_transformed()
            engine.add_geometry(geom)
            warpped_engine.displayed_geometries[mesh_name] = geom
        engine.get_view_control().convert_from_pinhole_camera_parameters(
            self.camera_intrinsic.get_pinhole_camera_param(self.camera_pose, fix_bug=True)
        )
        self.gl_render_options.point_size = self.point_size
        self.gl_render_options.save_to_json("/tmp/ro.json")
        engine.get_render_option().load_from_json("/tmp/ro.json")
        if self.viewport_shading == 'NORMAL':
            engine.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
            engine.get_render_option().point_color_option = o3d.visualization.PointColorOption.Normal
        elif self.viewport_shading == 'UNLIT':
            engine.get_render_option().light_on = False
        return warpped_engine

    def _update_gl_engine(self, gl_engine: GLEngineWrapper):
        engine = gl_engine.engine
        if "relative_camera" in self.animator.events.keys() or "camera_base" in self.animator.events.keys():
            engine.get_view_control().convert_from_pinhole_camera_parameters(
                self.camera_intrinsic.get_pinhole_camera_param(self.camera_pose, fix_bug=True)
            )
        for obj_uuid in self.animator.events.keys():
            if obj_uuid in gl_engine.displayed_geometries.keys():
                old_geom = gl_engine.displayed_geometries[obj_uuid]
                engine.remove_geometry(old_geom, reset_bounding_box=False)
                geom = self.objects[obj_uuid].get_transformed()
                engine.add_geometry(geom, reset_bounding_box=False)
                gl_engine.displayed_geometries[obj_uuid] = geom

    def _build_filament_engine(self, scene):
        # Note: sv is the object with shared API.
        if isinstance(scene, o3d.visualization.rendering.Open3DScene):
            visualizer = None
            sv = scene
        elif isinstance(scene, o3d.visualization.O3DVisualizer):
            visualizer = scene
            scene = visualizer.scene
            sv = visualizer
        else:
            raise NotImplementedError

        sv.set_background((1.0, 1.0, 1.0, 1.0), None)
        sv.show_skybox(False)
        for mesh_name, mesh_obj in self.objects.items():
            # TODO: Adapt to other types of geometries!
            #   cpp/open3d/visualization/visualizer/O3DVisualizer.cpp:883
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = {
                'LIT': "defaultLit", 'UNLIT': "defaultUnlit", 'NORMAL': 'normals'
            }[self.viewport_shading]
            mat.point_size = int(self.point_size)
            mat.line_width = 1
            if "alpha" in mesh_obj.attributes:
                mat.base_color = (1.0, 1.0, 1.0, mesh_obj.attributes["alpha"])
                mat.shader = "defaultLitTransparency"
            sv.add_geometry(mesh_name, mesh_obj.get_transformed(), mat)

        if visualizer is not None:
            # Although we already set material above, the following will make the UI correct.
            visualizer.point_size = int(self.point_size)
            visualizer.line_width = 1
            visualizer.scene_shader = {
                'LIT': visualizer.STANDARD, 'UNLIT': visualizer.UNLIT, 'NORMAL': visualizer.NORMALS
            }[self.viewport_shading]

        import open3d.visualization.rendering as o3dr
        scene.view.set_color_grading(o3dr.ColorGrading(o3dr.ColorGrading.Quality.ULTRA,
                                                       o3dr.ColorGrading.ToneMapping.LINEAR))
        # w.scene.scene.add_directional_light('light', [1, 1, 1], np.array([0,-1,-1]), 1e5, True)
        # w.scene.scene.add_directional_light('light2', [1, 1, 1], np.array([0,1,1]), 1e5, True)
        # w.scene.scene.add_point_light('plight', [1, 1, 1], [0, 1, 1], 1e6, 1000, True)
        # w.scene.view.set_shadowing(True, o3dr.ShadowType.PCF)

        cur_scene_light = self.lights.get("default", SceneLight('NONE'))
        if cur_scene_light.type == 'SUN':
            scene.scene.set_sun_light(
                -cur_scene_light.pose.q.rotation_matrix[:, 2],
                [255, 255, 255],
                400.0
            )
        elif cur_scene_light.type == 'NONE':
            scene.scene.set_sun_light([0., 1., 0.], [0, 0, 0], 0.0)

    def record_camera_pose(self, vis):
        if not hasattr(vis, "get_view_control"):
            self.camera_pose, self.camera_intrinsic = VisualizerManager.get_window_camera_info(vis)
        else:
            cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            cam_pose = cam_param.extrinsic
            self.camera_pose = Isometry.from_matrix(cam_pose).inv()
            self.camera_intrinsic = CameraIntrinsic.from_open3d_intrinsic(cam_param.intrinsic)
        if self.cam_path is not None:
            self.save_camera()
        print(f"Camera parameter saved! w={self.camera_intrinsic.w}, h={self.camera_intrinsic.h}")
        return False

    def preview(self, allow_change_pose=True, add_ruler=True, title="Render Preview", use_new_api=False,
                key_bindings=None, with_animation=False):

        if len(vis_manager.scenes) != 0:
            print("Warning: there are still buffered scenes in the manager which are not fully shown.")
            vis_manager.reset()
        vis_manager.add_scene(self, title, self.record_camera_pose if allow_change_pose else None)
        vis_manager.run(use_new_api=use_new_api, key_bindings=key_bindings)
        return self

    def _setup_blender_static(self):
        # Set-up the static contents of the scene
        blender.send_clear()
        for obj_uuid, obj_data in self.objects.items():
            blender.send_entity(obj_data.geom, obj_uuid, obj_data.pose, obj_data.attributes)
        for light_uuid, light_data in self.lights.items():
            blender.send_light(light_data, light_uuid)
        blender.send_camera(self.camera_pose, self.camera_intrinsic)
        blender.send_eval(f"bg_node=bpy.data.worlds['World'].node_tree.nodes['Background'];"
                          f"bg_node.inputs[0].default_value=({self.ambient_color[0]},{self.ambient_color[1]},{self.ambient_color[2]},1);"
                          f"bg_node.inputs[1].default_value={self.ambient_color[3]}")
        blender.send_eval(f"bpy.context.scene.render.film_transparent={self.film_transparent}")

    def render_blender(self, do_render: bool = True, save_path: str = None, quality: int = 128):
        self._setup_blender_static()
        if not do_render:
            # Wait for user to respond
            blender.poll_notified()

        if save_path is not None:
            blender.send_render(quality=quality, save_path=save_path)
            return None
        else:
            with tempfile.TemporaryDirectory() as render_tmp_dir_p:
                # By default, blender will add '.png' to the input path if the suffix didn't exist.
                render_tmp_file = Path(render_tmp_dir_p) / "rgb.png"
                blender.send_render(quality=quality, save_path=str(render_tmp_file))
                rgb_img = cv2.imread(str(render_tmp_file), cv2.IMREAD_UNCHANGED)

        if rgb_img.shape[2] == 3 or rgb_img.shape[2] == 4:
            rgb_img[:, :, :3] = rgb_img[:, :, :3][:, :, ::-1]
        return rgb_img

    def render_blender_animation(self, do_render: bool = True, quality: int = 128):
        self._setup_blender_static()
        self.animator.send_blender()
        if not do_render:
            blender.poll_notified()

        t_start, t_end = self.animator.get_range()
        for t_cur in range(t_start, t_end + 1):
            blender.send_eval(f"bpy.context.scene.frame_set({t_cur})")
            with tempfile.TemporaryDirectory() as render_tmp_dir_p:
                # By default, blender will add '.png' to the input path if the suffix didn't exist.
                render_tmp_file = Path(render_tmp_dir_p) / "rgb.png"
                blender.send_render(quality=quality, save_path=str(render_tmp_file))
                rgb_img = cv2.imread(str(render_tmp_file), cv2.IMREAD_UNCHANGED)
                if rgb_img.shape[2] == 3 or rgb_img.shape[2] == 4:
                    rgb_img[:, :, :3] = rgb_img[:, :, :3][:, :, ::-1]
            yield t_cur, rgb_img

    def render_opengl(self, multisample: int = 1, need_alpha=False, save_path: str = None):
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

    def render_filament(self, headless: bool = True):
        # Cache DISPLAY environment
        x11_environ = None
        if 'DISPLAY' in os.environ:
            x11_environ = os.environ['DISPLAY']
            del os.environ['DISPLAY']
        renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.camera_intrinsic.w, self.camera_intrinsic.h, "", headless)
        self._build_filament_engine(renderer.scene)
        renderer.setup_camera(self.camera_intrinsic.to_open3d_intrinsic(), self.camera_pose.inv().matrix)
        img = renderer.render_to_image()
        if x11_environ is not None:
            os.environ['DISPLAY'] = x11_environ
        return np.array(img)


class AnimatedVisualizer:
    def __init__(self, scene):      # scene is an Animator.
        self.scene = scene
        self.engine = o3d.visualization.VisualizerWithKeyCallback()
        self.engine.create_window(window_name="Animated Visualizer",
                                  width=scene.scene.camera_intrinsic.w,
                                  height=scene.scene.camera_intrinsic.h,
                                  visible=True)
        self.engine.get_render_option().mesh_show_back_face = True
        self.current_displaying = {}
        if True:    # Open3D cannot correctly initialize from given camera pose without this mesh due to bad znear and zfar
            probe_mesh = o3d.geometry.TriangleMesh.create_box(width=5.0, height=5.0, depth=5.0)
            self.engine.add_geometry(probe_mesh)
            self.engine.remove_geometry(probe_mesh, reset_bounding_box=False)

    def _update_geometry(self, name, transformed_geom, geom_pose):
        print("Updated", name, transformed_geom, geom_pose)
        if name in self.current_displaying.keys():
            self.engine.remove_geometry(self.current_displaying[name].geom, reset_bounding_box=False)
            del self.current_displaying[name]
        if transformed_geom is not None:
            self.engine.add_geometry(transformed_geom, reset_bounding_box=False)
            self.current_displaying[name] = SceneObject(transformed_geom, geom_pose)

    def run(self):
        self.engine.run()

    def refresh(self):
        # Camera pose will always be updated.
        self.engine.get_view_control().convert_from_pinhole_camera_parameters(
            self.scene.scene.camera_intrinsic.get_pinhole_camera_param(
                self.scene.scene.camera_pose, fix_bug=True)
        )
        # Checkout object updates.
        for mesh_name, mesh_obj in self.scene.scene.objects.items():
            # Object visibility.
            if mesh_name not in self.current_displaying.keys() and mesh_obj.visible:
                self._update_geometry(mesh_name, mesh_obj.get_transformed(), mesh_obj.pose)
            elif mesh_name in self.current_displaying.keys() and not mesh_obj.visible:
                self._update_geometry(mesh_name, None, None)
            # Object pose. (do not worry the object being added twice because we only capture difference)
            # TODO: Finalize the debug this...
            # if mesh_name in self.current_displaying.keys() and mesh_obj.pose != self.current_displaying[mesh_obj].pose:
            #     self._update_geometry(mesh_name, mesh_obj.get_transformed(), mesh_obj.pose)


# class Animator:
#     def __init__(self, scene):
#         self.scene = scene
#         # uuid -> attributes -> interpolator
#         self.events = defaultdict(lambda: defaultdict(BaseValueInterpolator))
#
#     def get_range(self):
#         frame_max = -1e10
#         frame_min = 1e10
#         for obj_attrib in self.events.values():
#             for obj_interp in obj_attrib.values():
#                 frame_max = max(frame_max, obj_interp.get_last_t())
#                 frame_min = min(frame_min, obj_interp.get_first_t())
#         return frame_min, frame_max
#
#     def set_frame(self, t):
#         for obj_uuid, obj_attribs in self.events.items():
#             for attrib_name, attrib_interp in obj_attribs.items():
#                 attrib_val = attrib_interp.get_value(t)
#                 if obj_uuid == "camera" and attrib_name == "pose":
#                     self.scene.camera_pose = attrib_val
#                 elif attrib_name == "pose":
#                     self.scene.objects[obj_uuid].pose = attrib_val
#                 elif attrib_name == "visible":
#                     self.scene.objects[obj_uuid].visible = attrib_val
#
#     def move_camera_event(self, frame_id: int, new_pose: Isometry):
#         self.events["camera"]["pose"].set_keyframe(frame_id, new_pose)
#
#     def add_object_event(self, frame_id: int, obj_uuid: str = None, obj_geom=None):
#         """
#         Add an object at @frame_id. If @obj_uuid is not None, will re-use previously-added object,
#         otherwise will add new object as @obj_geom
#         """
#         if obj_geom is not None:
#             obj_uuid = self.scene.add_object(obj_geom, return_name=True)
#
#         assert obj_uuid in self.scene.objects.keys()
#         self.events[obj_uuid]["visible"].set_keyframe(frame_id, True)
#
#
#     def remove_object_event(self, frame_id: int, obj_uuid: str):
#         assert obj_uuid in self.scene.objects.keys()
#         self.events[obj_uuid]["visible"].set_keyframe(frame_id, False)
#
#     def move_object_event(self, frame_id: int, obj_uuid: str, new_pose: Isometry):
#         assert obj_uuid in self.scene.objects.keys()
#         self.events["camera"]["pose"].set_keyframe(frame_id, new_pose)
#
#     def render_blender(self, increment=1):
#         self.scene._setup_blender_static()
#
#         for obj_uuid, obj_attribs in self.events.items():
#             for attrib_name, attrib_interp in obj_attribs.items():
#                 attrib_keyframes = attrib_interp.get_keyframes()
#                 for kt, kval in attrib_keyframes:
#                     blender.send_add_keyframe(obj_uuid, int(kt // increment), attrib_name, kval)
#
#         blender.poll_notified()
#         # if not do_render:
#         #     # Wait for user to respond
#         #     blender.poll_notified()
#         #
#         # if save_path is not None:
#         #     blender.send_render(quality=quality, save_path=save_path)
#         #     return None
#         # else:
#         #     with tempfile.TemporaryDirectory() as render_tmp_dir_p:
#         #         # By default, blender will add '.png' to the input path if the suffix didn't exist.
#         #         render_tmp_file = Path(render_tmp_dir_p) / "rgb.png"
#         #         blender.send_render(quality=quality, save_path=str(render_tmp_file))
#         #         rgb_img = cv2.imread(str(render_tmp_file), cv2.IMREAD_UNCHANGED)
#         #
#         # if rgb_img.shape[2] == 3 or rgb_img.shape[2] == 4:
#         #     rgb_img[:, :, :3] = rgb_img[:, :, :3][:, :, ::-1]
#         # return rgb_img
#
#     def preview(self, start_frame=0, increment=1, autoplay=False):
#         visualizer = AnimatedVisualizer(self)
#
#         cur_frame = start_frame
#         self.set_frame(cur_frame)
#         visualizer.refresh()
#
#         def next_frame(vis):
#             nonlocal cur_frame
#             self.set_frame(cur_frame + increment)
#             cur_frame += increment
#             visualizer.refresh()
#
#         def prev_frame(vis):
#             nonlocal cur_frame
#             self.set_frame(cur_frame - increment)
#             cur_frame -= increment
#             visualizer.refresh()
#
#         visualizer.engine.register_key_callback(key=ord("A"), callback_func=next_frame)
#         visualizer.engine.register_key_callback(key=ord("B"), callback_func=prev_frame)
#         if autoplay:
#             visualizer.engine.register_animation_callback(callback_func=next_frame)
#
#         visualizer.engine.run()


def multiview_image(geoms: list, width: int = 256, height: int = 256, up_axis=None, viewport_shading='NORMAL'):
    """
    Headless render a multiview image of geometry list, mainly used for training visualization.
    :param geoms: list of geometry, could be annotated.
    :param width: width of each image
    :param height: height of each image
    :param up_axis: up axis of scene
    :param viewport_shading: shading used.
    :return: an image.
    """
    scene = Scene()
    for geom in geoms:
        scene.add_object(geom=geom)
    scene.viewport_shading = viewport_shading
    multiview_pics = []
    for view_angle in [0.0, 90.0, 180.0, 270.0]:
        scene.quick_camera(w=width, h=height, fov=45.0, up_axis=up_axis,
                           fill_percent=0.7, plane_angle=view_angle + 45.0)
        my_pic = scene.render_filament(headless=True)
        multiview_pics.append(my_pic)
    return image.hlayout_images(multiview_pics)


def preset_shadowed_render(geom: o3d.geometry.TriangleMesh, set_camera: bool = False):
    """
    A preset that quickly convert 3D into a picture.
        This preset features in a white background and a nice color-bleed shadow. Perfect for a paper.
    :param geom:
    :param set_camera:
    :return:
    """
    pass


def render_depth(render_group_lists: list, camera_list: list, camera_intrinsic: CameraIntrinsic,
                 backend='blender', exec_id: int = 0):
    """
    Render a list of depth images.
    :param render_group_lists: List of [mesh, mesh, ...]
    :param camera_list: List of Isometry
    :param camera_intrinsic: Intrinsic of the camera.
    :param exec_id: Run identifier.
    :return: list of numpy mat
    """
    assert len(render_group_lists) == len(camera_list), "Camera and groups must be same"
    if backend == 'blender':
        return render_depth_blender(render_group_lists, camera_list, camera_intrinsic, exec_id)
    elif backend == 'open3d':
        return render_depth_open3d(render_group_lists, camera_list, camera_intrinsic)
    else:
        raise NotImplementedError


def render_full_model(render_group_lists: list, camera_list: list, n_points: int, poisson_disk_sampling=True):
    """
    Render with an X-ray camera. This is equivalent to simply sample on a mesh.
    We render this to test the importance of missing areas.
    :return: list of point clouds.
    """
    from utils.point_util import furthest_sampling_cloud
    n_group = len(render_group_lists)

    pc_points = []
    pc_segms = []
    for group_id in range(n_group):
        parts_point = []
        parts_segm = []
        for part_id, part_mesh in enumerate(render_group_lists[group_id]):
            if len(part_mesh.triangles) == 0:
                continue
            if poisson_disk_sampling:
                part_pc = part_mesh.sample_points_poisson_disk(n_points)
            else:
                part_pc = part_mesh.sample_points_uniformly(n_points)
            # print(np.asarray(part_pc.points))
            parts_point.append(np.asarray(part_pc.points))
            parts_segm.append(np.ones((n_points,), dtype=int) * (part_id + 1))
        group_points = np.vstack(parts_point)
        group_segm = np.concatenate(parts_segm)
        # o3d.visualization.draw_geometries([vis_util.pointcloud(group_points, group_segm)])
        sampled_indices = furthest_sampling_cloud(group_points, n_points, output_indices=True)
        pc_points.append((camera_list[group_id].inv() @ group_points[sampled_indices, :]))
        pc_segms.append(group_segm[sampled_indices])
    return pc_points, pc_segms


def render_depth_open3d(render_group_lists: list, camera_list: list, camera_intrinsic: CameraIntrinsic):
    n_group = len(render_group_lists)

    depth_maps = []
    obid_maps = []
    engine = o3d.visualization.Visualizer()
    engine.create_window(width=camera_intrinsic.w, height=camera_intrinsic.h, visible=False)
    for group_id in range(n_group):
        cur_cam = camera_list[group_id]
        for part_id, part_mesh in enumerate(render_group_lists[group_id]):
            # Used for indexing.
            part_mesh.paint_uniform_color([(part_id + 1) * 8 / 255.0, 0, 0])
            engine.add_geometry(part_mesh)
        engine.get_view_control().convert_from_pinhole_camera_parameters(
            camera_intrinsic.get_pinhole_camera_param(cur_cam, fix_bug=True)
        )
        captured_depth = engine.capture_depth_float_buffer(do_render=True)
        captured_depth = np.asarray(captured_depth)
        # Note: We should create a separate frame buffer object for this to work.
        #   In that sense
        captured_rgb = engine.capture_screen_float_buffer(do_render=True)
        captured_rgb = (np.asarray(captured_rgb) * 255.0).astype(np.uint8)
        captured_indexes = captured_rgb[:, :, 0] // 8

        depth_maps.append(captured_depth)
        obid_maps.append(captured_indexes)
        engine.clear_geometries()

    return depth_maps, obid_maps


def render_depth_blender(render_group_lists: list, camera_list: list, camera_intrinsic: CameraIntrinsic,
                         exec_id: int = 0):
    print("Warning: Blender backend does not support parallel computing. To do so, change assets/.py")

    asset_base = Path(__file__).parent / "assets"
    blender_file_path = asset_base / "render.blend"
    blender_script_path = asset_base / "blender_depth.py"

    # Save objects and camera as file.
    n_group = len(render_group_lists)

    with tempfile.TemporaryDirectory() as render_tmp_dir_p:
        render_tmp_dir = Path(render_tmp_dir_p)
        with (render_tmp_dir / f"{exec_id}.camera").open('w') as cam_f:
            for group_id in range(n_group):
                cur_cam = camera_list[group_id]
                cur_cam_q = cur_cam.q * Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)
                intr = camera_intrinsic.blender_attributes
                cam_f.write(f"{cur_cam.t[0]} {cur_cam.t[1]} {cur_cam.t[2]} {cur_cam_q[0]} "
                            f"{cur_cam_q[1]} {cur_cam_q[2]} {cur_cam_q[3]} "
                            f"{intr[0]} {intr[1]} {intr[2]} {intr[3]} {intr[4]}\n")
                for part_id, part_mesh in enumerate(render_group_lists[group_id]):
                    o3d.io.write_triangle_mesh(str(render_tmp_dir / f"{exec_id}-{group_id}-{part_id}.obj"),
                                               part_mesh)

                # o3d.visualization.draw_geometries(render_group_lists[group_id] +
                #                                   [o3d.geometry.TriangleMesh.create_coordinate_frame().transform(Isometry(cur_cam_q, cur_cam.t).matrix),
                #                                    o3d.geometry.TriangleMesh.create_coordinate_frame()])

        # Perform rendering.
        os.system(f"{BLENDER_EXEC_PATH} --background {blender_file_path} --python {blender_script_path} -- "
                  f"--asset {render_tmp_dir} --output {render_tmp_dir} --exec {exec_id} --num {n_group}")

        # Load in depth map
        depth_maps = []
        obid_maps = []
        for group_id in range(n_group):
            exr_depth = cv2.imread(str(render_tmp_dir / f"{exec_id}-{group_id}.exr"),
                                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            exr_depth = exr_depth[:, :, 0]
            depth_maps.append(exr_depth)
            exr_obid = cv2.imread(str(render_tmp_dir / f"{exec_id}-{group_id}-ob.exr"), cv2.IMREAD_UNCHANGED)
            exr_obid = exr_obid[:, :, 0].astype(int)
            obid_maps.append(exr_obid)

    return depth_maps, obid_maps


def depth_to_point_cloud(depth_mat: np.ndarray, camera_intrinsic: CameraIntrinsic, norm_ray=False):
    """
    Convert depth map to point cloud, considering the depth map is a z buffer
    :param depth_mat: W x H np.ndarray
    :param camera_intrinsic: Camera Intrinsic
    :param norm_ray: Whether the depth map is generated by normalizing rays.
    :return: (point cloud, W x H binary mask)
    """
    # Build mesh grid
    xx, yy = np.meshgrid(np.arange(0, camera_intrinsic.w), np.arange(0, camera_intrinsic.h))
    mg = np.concatenate((xx.reshape(1, -1), yy.reshape(1, -1)), axis=0)
    mg_homo = np.vstack((mg, np.ones((1, mg.shape[1]))))
    pc = np.matmul(camera_intrinsic.inv_K, mg_homo)
    if norm_ray:
        pc /= np.linalg.norm(pc, axis=0)
    pc = depth_mat.ravel()[np.newaxis, :] * pc
    # Crop invalid observations.
    # print(pc)
    if norm_ray:
        pc_mask = pc[2] < 1e2
    else:
        pc_mask = pc[2] > 0.0
    pc = pc[:, pc_mask].T
    return pc, pc_mask
