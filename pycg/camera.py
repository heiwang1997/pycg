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

    def get_rays(self, normalized: bool = True):
        xx, yy = np.meshgrid(np.arange(0, self.w) + 0.5, np.arange(0, self.h) + 0.5)
        mg = np.concatenate((xx.reshape(1, -1), yy.reshape(1, -1)), axis=0)
        mg_homo = np.vstack((mg, np.ones((1, mg.shape[1]))))
        pc = np.matmul(self.inv_K, mg_homo)
        if normalized:
            pc /= np.linalg.norm(pc, axis=0)
        return pc.T.reshape((self.w, self.h, 3))
