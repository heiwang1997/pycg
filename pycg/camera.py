import numpy as np
from pycg import o3d
from pycg.isometry import Isometry



class CameraIntrinsic:
    """
    A notice on Perspective/Orthogonal Projection.
        - In theory, orthogonal projection can also be represented by fx,fy,cx,cy, but multiple combinations of
    camera pose and fx/fy can give the same projection.
        - In old API, once Fov <= 5.0, it will be treated as Orthogonal projection.
        - In new API, in GUI you cannot change to Orthogonal projection (with minimum fov = 5.0).
        - Hence, in our implementation, we use the fov=5.0 perspective to mimic behaviour of ortho-projection.
    No code in Python end should be changed, just remove the warning from C++ end suffices.
    """

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

    @property
    def fov_y(self):
        return 2 * np.arctan2(self.h, 2 * self.fy)

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
    
    @staticmethod
    def from_angle_coverage(height: int, theta_min: float = -45.0, theta_max: float = 45.0, edge: bool = True):
        angle_radian = np.deg2rad(max(abs(theta_min), abs(theta_max)))
        cy = height / 2
        fy = cy / np.tan(angle_radian)
        if edge:
            fy *= np.sqrt(2) / 2.
        cx = fx = fy
        return CameraIntrinsic(int(cx * 2), height, fx, fy, cx, cy)

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

    @classmethod
    def interpolate(cls, intr1, intr2, t):
        return CameraIntrinsic(
            int(intr1.w * (1 - t) + intr2.w * t),
            int(intr1.h * (1 - t) + intr2.h * t),
            intr1.fx * (1 - t) + intr2.fx * t,
            intr1.fy * (1 - t) + intr2.fy * t,
            intr1.cx * (1 - t) + intr2.cx * t,
            intr1.cy * (1 - t) + intr2.cy * t
        )

    def world_to_ndc(self, near: float = 0.01, far: float = 100.0):
        perspective_mat = np.asarray([
            [self.fx, 0,       -self.w / 2. + self.cx, 0.0],
            [0.0,     self.fy, -self.h / 2. + self.cy, 0.0],
            [0.0,     0.0,     0.0,                    1.0],       # Per Kaolin
            [0.0,     0.0,     -1.0,                   0.0]
        ])
        U = -2.0 * near * far / (far - near)
        V = -(far + near) / (far - near)
        ndc_mat = np.asarray([
            [2.0 / self.w, 0.0,          0.0, 0.0],
            [0.0,          2.0 / self.h, 0.0, 0.0],
            [0.0,          0.0,          U,   V],
            [0.0,          0.0,          0.0, -1.0]
        ])
        return ndc_mat @ perspective_mat

    def get_rays(self, device=None, normalized: bool = True):

        if device is None:
            # Numpy backend
            uu, vv = np.meshgrid(
                (np.arange(0, self.w) + 0.5 - self.cx) * (1 / self.fx),
                (np.arange(0, self.h) + 0.5 - self.cy) * (1 / self.fy),
                indexing='xy'
            )
            xyz = np.stack([uu, vv, np.ones_like(uu)], axis=-1)

            if normalized:
                xyz /= np.linalg.norm(xyz, axis=0)

            return xyz.reshape((self.h, self.w, 3))
        
        else:
            # Pytorch backend
            import torch

            uu, vv = torch.meshgrid(
                (torch.arange(self.w, device=device) + 0.5 - self.cx) * (1 / self.fx),
                (torch.arange(self.h, device=device) + 0.5 - self.cy) * (1 / self.fy),
                indexing='xy'
            )
            xyz = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)

            if normalized:
                xyz = xyz / torch.linalg.norm(xyz, dim=-1, keepdim=True)

            return xyz.view(self.h, self.w, 3)

    def nvdiffrast_matrices(self, pose: Isometry, near: float = 0.01, far: float = 100.0):
        """
        v_ndc = ctx.vertex_transform(v_world, out["mvp"])
        """
        return {
            "mvp": self.world_to_ndc(near, far) @ pose.inv().matrix
        }
    
    def gsplat_matrices(self, pose: Isometry, device=None, near: float = 0.01, far: float = 100.0):
        """
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.h),
            image_width=int(viewpoint_camera.w),
            tanfovx=viewpoint_camera.w / (2 * viewpoint_camera.fx),
            tanfovy=viewpoint_camera.h / (2 * viewpoint_camera.fy),
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            campos=viewpoint_camera.camera_center,
            ...
        )
        """
        P = np.zeros((4, 4))
        P[0, 0] = 2 * self.fx / self.w
        P[1, 1] = 2 * self.fy / self.h
        P[0, 2] = 2 * self.cx / self.w - 1
        P[1, 2] = 2 * self.cy / self.h - 1
        P[3, 2] = 1.0
        P[2, 2] = far / (far - near)
        P[2, 3] = -(far * near) / (far - near)

        wvt = pose.inv().matrix.T
        proj = wvt @ P.T
        campos = pose.t

        if device is not None:
            import torch
            wvt = torch.tensor(wvt, device=device).float()
            proj = torch.tensor(proj, device=device).float()
            campos = torch.tensor(campos, device=device).float()

        return {
            "image_height": int(self.h),
            "image_width": int(self.w),
            "tanfovx": self.w / (2 * self.fx),
            "tanfovy": self.h / (2 * self.fy),
            "viewmatrix": wvt,
            "projmatrix": proj,
            "campos": campos
        }
