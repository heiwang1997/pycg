"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import copy
from typing import Union, Optional

import numpy as np
from pycg.exp import logger
from pyquaternion import Quaternion


def so3_vee(Phi):
    """Extract the vector representation from a skew-symmetric matrix.
    
    Args:
        Phi: A (3,3) or (N,3,3) skew-symmetric matrix
        
    Returns:
        phi: A (3,) or (N,3) vector containing the unique elements
    """
    if Phi.ndim < 3:
        Phi = np.expand_dims(Phi, axis=0)

    if Phi.shape[1:3] != (3, 3):
        raise ValueError("Phi must have shape ({},{}) or (N,{},{})".format(3, 3, 3, 3))

    phi = np.empty([Phi.shape[0], 3])
    phi[:, 0] = Phi[:, 2, 1]  # Extract unique elements from skew-symmetric matrix
    phi[:, 1] = Phi[:, 0, 2]
    phi[:, 2] = Phi[:, 1, 0]
    return np.squeeze(phi)


def so3_wedge(phi):
    """Convert a vector to its skew-symmetric matrix representation.
    
    Args:
        phi: A (3,) or (N,3) vector
        
    Returns:
        Phi: A (3,3) or (N,3,3) skew-symmetric matrix
    """
    phi = np.atleast_2d(phi)
    if phi.shape[1] != 3:
        raise ValueError(
            "phi must have shape ({},) or (N,{})".format(3, 3))

    Phi = np.zeros([phi.shape[0], 3, 3])
    # Fill skew-symmetric matrix elements
    Phi[:, 0, 1] = -phi[:, 2]
    Phi[:, 1, 0] = phi[:, 2]
    Phi[:, 0, 2] = phi[:, 1]
    Phi[:, 2, 0] = -phi[:, 1]
    Phi[:, 1, 2] = -phi[:, 0]
    Phi[:, 2, 1] = phi[:, 0]
    return np.squeeze(Phi)


def so3_log(matrix):
    """Maps a rotation matrix from SO(3) into its corresponding vector representation in so(3)
    
    Args:
        matrix: A (3,3) rotation matrix
        
    Returns:
        phi: A (3,) vector representing the rotation
    """
    cos_angle = 0.5 * np.trace(matrix) - 0.5
    cos_angle = np.clip(cos_angle, -1., 1.)
    angle = np.arccos(cos_angle)
    if np.isclose(angle, 0.):
        return so3_vee(matrix - np.identity(3))
    else:
        return so3_vee((0.5 * angle / np.sin(angle)) * (matrix - matrix.T))


# Please note that the right jacobian is just J_r(xi) = -J_l(xi) for both SO3 and SE3

def so3_left_jacobian(phi):
    """Compute the left Jacobian for SO(3).
    
    Args:
        phi: A (3,) vector representing rotation in so(3)
        
    Returns:
        J: A (3,3) left Jacobian matrix
    """
    angle = np.linalg.norm(phi)

    if np.isclose(angle, 0.):
        return np.identity(3) + 0.5 * so3_wedge(phi)

    axis = phi / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return (s / angle) * np.identity(3) + \
           (1 - s / angle) * np.outer(axis, axis) + \
           ((1 - c) / angle) * so3_wedge(axis)


def se3_curlywedge(xi):
    xi = np.atleast_2d(xi)

    Psi = np.zeros([xi.shape[0], 6, 6])
    # Fill block matrix structure
    Psi[:, 0:3, 0:3] = so3_wedge(xi[:, 3:6])  # Rotation part
    Psi[:, 0:3, 3:6] = so3_wedge(xi[:, 0:3])  # Translation part
    Psi[:, 3:6, 3:6] = Psi[:, 0:3, 0:3]       # Copy rotation part

    return np.squeeze(Psi)


def se3_left_jacobian_Q_matrix(xi):
    rho = xi[0:3]  # translation part
    phi = xi[3:6]  # rotation part

    rx = so3_wedge(rho)
    px = so3_wedge(phi)

    ph = np.linalg.norm(phi)
    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = np.cos(ph)
    sph = np.sin(ph)

    m1 = 0.5
    m2 = (ph - sph) / ph3
    m3 = (0.5 * ph2 + cph - 1.) / ph4
    m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

    t1 = rx
    t2 = px.dot(rx) + rx.dot(px) + px.dot(rx).dot(px)
    t3 = px.dot(px).dot(rx) + rx.dot(px).dot(px) - 3. * px.dot(rx).dot(px)
    t4 = px.dot(rx).dot(px).dot(px) + px.dot(px).dot(rx).dot(px)

    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4


def se3_left_jacobian(xi):
    rho = xi[0:3]  # translation part
    phi = xi[3:6]  # rotation part

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.):
        return np.identity(6) + 0.5 * se3_curlywedge(xi)

    so3_jac = so3_left_jacobian(phi)
    Q_mat = se3_left_jacobian_Q_matrix(xi)

    jac = np.zeros([6, 6])
    jac[0:3, 0:3] = so3_jac
    jac[0:3, 3:6] = Q_mat
    jac[3:6, 3:6] = so3_jac

    return jac


def se3_inv_left_jacobian(xi):
    rho = xi[0:3]  # translation part
    phi = xi[3:6]  # rotation part

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.):
        return np.identity(6) - 0.5 * se3_curlywedge(xi)

    so3_inv_jac = so3_inv_left_jacobian(phi)
    Q_mat = se3_left_jacobian_Q_matrix(xi)

    jac = np.zeros([6, 6])
    jac[0:3, 0:3] = so3_inv_jac
    jac[0:3, 3:6] = -so3_inv_jac.dot(Q_mat).dot(so3_inv_jac)
    jac[3:6, 3:6] = so3_inv_jac

    return jac


def so3_inv_left_jacobian(phi):
    angle = np.linalg.norm(phi)

    if np.isclose(angle, 0.):
        return np.identity(3) - 0.5 * so3_wedge(phi)

    axis = phi / angle
    half_angle = 0.5 * angle
    cot_half_angle = 1. / np.tan(half_angle)

    return half_angle * cot_half_angle * np.identity(3) + \
           (1 - half_angle * cot_half_angle) * np.outer(axis, axis) - \
           half_angle * so3_wedge(axis)


def project_orthogonal(rot):
    """Project a matrix to the closest orthogonal matrix.
    
    Args:
        rot: A (3,3) matrix
        
    Returns:
        rot: A (3,3) orthogonal matrix in SO(3)
    """
    u, s, vh = np.linalg.svd(rot, full_matrices=True, compute_uv=True)
    rot = u @ vh
    if np.linalg.det(rot) < 0:  # Ensure proper rotation (det=1)
        u[:, 2] = -u[:, 2]
        rot = u @ vh
    return rot


class Isometry:
    """
    Class representing rigid body transformations (rotation + translation).
    
    When representing camera, the camera convention is right-hand and opencv-style (RDF)
    """
    
    GL_POST_MULT = Quaternion(degrees=180.0, axis=[1.0, 0.0, 0.0])

    def __init__(self, q=None, t=None):
        """Initialize an isometry.
        
        Args:
            q: A Quaternion representing rotation
            t: A (3,) translation vector
        """
        if q is None:
            q = Quaternion()
        if t is None:
            t = np.zeros(3)
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)
        t = t.ravel()
        assert t.shape[0] == 3
        self.q = q
        self.t = t

    def __repr__(self):
        return f"Isometry: t = {self.t}, q = {self.q}"

    @property
    def rotation(self):
        """Get the rotation component only."""
        return Isometry(q=self.q)

    @property
    def matrix(self):
        """Get the 4x4 transformation matrix."""
        mat = self.q.transformation_matrix
        mat[0:3, 3] = self.t
        return mat

    @staticmethod
    def from_matrix(mat, t_component=None, ortho=False):
        """Create an isometry from a transformation matrix.
        
        Args:
            mat: A (3,3) or (3,4) or (4,4) matrix
            t_component: Optional (3,) translation vector. need to be provided if mat is (3,3)
            ortho: Whether to enforce orthogonality
            
        Returns:
            iso: An Isometry object
        """
        assert isinstance(mat, np.ndarray)
        if t_component is None:
            assert mat.shape == (4, 4) or mat.shape == (3, 4)
            rot_mat = mat[:3, :3]
            if ortho:
                rot_mat = rot_mat.astype(float)
                rot_mat = project_orthogonal(rot_mat)
            else:
                rot_scale = np.diag(rot_mat.T @ rot_mat)
                if not np.allclose(rot_scale, np.ones_like(rot_scale)):
                    logger.warning(f"Rotation matrix has scale {rot_scale}, perhaps use ScaledIsometry?")
            return Isometry(q=Quaternion(matrix=rot_mat), t=mat[:3, 3])
        else:
            assert mat.shape == (3, 3)
            assert t_component.shape == (3,)
            mat = mat.astype(float)
            if ortho:
                mat = project_orthogonal(mat)
            return Isometry(q=Quaternion(matrix=mat), t=t_component)

    @staticmethod
    def _str_to_axis(name):
        """Convert string axis name to vector.
        
        Args:
            name: String like 'X','Y','Z','+X','-X' etc.
            
        Returns:
            axis: A (3,) unit vector
        """
        name = name.upper()
        if len(name) == 2:
            symbol, name = ['-', '+'].index(name[0]) * 2 - 1, name[1]
        else:
            symbol = 1
        axis_idx = ['X', 'Y', 'Z'].index(name)
        axis = np.zeros((3,))
        axis[axis_idx] = symbol
        return axis

    @staticmethod
    def from_axis_angle(axis, degrees=None, radians=None, t=None):
        """Create an isometry from axis-angle representation.
        
        Args:
            axis: A (3,) vector or string like 'X','Y','Z'
            degrees: Rotation angle in degrees
            radians: Rotation angle in radians
            t: Optional (3,) translation vector
            
        Returns:
            iso: An Isometry object
        """
        if degrees is None and radians is None:
            degrees = 0.0
        if isinstance(axis, str):
            axis = Isometry._str_to_axis(axis)
        return Isometry(q=Quaternion(axis=axis, degrees=degrees, radians=radians), t=t)

    @staticmethod
    def from_euler_angles(a1, a2, a3, format='XYZ', t=None):
        """Create an isometry from Euler angles.
        
        Args:
            a1,a2,a3: Rotation angles in degrees
            format: Rotation order, either 'XYZ' or 'YZX'
            t: Optional (3,) translation vector
            
        Returns:
            iso: An Isometry object
        """
        assert format in ['XYZ', 'YZX']
        rot1 = Quaternion(axis=Isometry._str_to_axis(format[0]), degrees=a1)
        rot2 = Quaternion(axis=Isometry._str_to_axis(format[1]), degrees=a2)
        rot3 = Quaternion(axis=Isometry._str_to_axis(format[2]), degrees=a3)
        return Isometry(q=(rot3 * rot2 * rot1), t=t)

    @staticmethod
    def from_twist(xi: np.ndarray):
        """Create an isometry from a twist vector.
        
        Args:
            xi: A (6,) twist vector [rho, phi] in se(3)
            
        Returns:
            iso: An Isometry object
        """
        rho = xi[:3]
        phi = xi[3:6]
        iso = Isometry.from_so3_exp(phi)
        iso.t = so3_left_jacobian(phi) @ rho
        return iso

    @staticmethod
    def from_so3_exp(phi: np.ndarray):
        """Create an isometry from so(3) vector
        
        Args:
            phi: A (3,) rotation vector in so(3)
            
        Returns:
            iso: An Isometry object
        """
        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return Isometry(q=Quaternion(matrix=np.identity(3) + so3_wedge(phi)))

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        rot_mat = (c * np.identity(3) +
                   (1 - c) * np.outer(axis, axis) +
                   s * so3_wedge(axis))

        return Isometry(q=Quaternion(matrix=rot_mat))

    @property
    def continuous_repr(self):
        """Get a continuous 9D representation [R1,R2,t]. 
        R1, R2 are the first two columns of the rotation matrix

        Returns:
            rep: A (9,) vector containing first two columns of rotation and translation
        """
        rot = self.q.rotation_matrix[:, 0:2].T.flatten()  # (6,)
        return np.concatenate([rot, self.t])  # (9,)

    @staticmethod
    def from_continuous_repr(rep, gs=True):
        """Create an isometry from continuous representation.
        
        Args:
            rep: A (9,) vector [R1,R2,t], R1, R2 are the first two columns of the rotation matrix
            gs: Whether to use Gram-Schmidt orthogonalization
            
        Returns:
            iso: An Isometry object
        """
        if isinstance(rep, list):
            rep = np.asarray(rep)
        assert isinstance(rep, np.ndarray)
        assert rep.shape == (9,)
        # For rotation, use Gram-Schmidt orthogonalization
        col1 = rep[0:3]
        col2 = rep[3:6]
        if gs:
            col1 /= np.linalg.norm(col1)
            col2 = col2 - np.dot(col1, col2) * col1
            col2 /= np.linalg.norm(col2)
        col3 = np.cross(col1, col2)
        return Isometry(q=Quaternion(matrix=np.column_stack([col1, col2, col3])), t=rep[6:9])

    @property
    def full_repr(self):
        """Get full 12D representation [R1,R2,R3,t].
        R1, R2, R3 are the first three columns of the rotation matrix
        
        Returns:
            rep: A (12,) vector containing full rotation matrix and translation
        """
        rot = self.q.rotation_matrix.T.flatten()
        return np.concatenate([rot, self.t])

    @staticmethod
    def from_full_repr(rep, ortho=False):
        """Create an isometry from full representation.
        
        Args:
            rep: A (12,) vector [R1,R2,R3,t], R1, R2, R3 are the first three columns of the rotation matrix
            ortho: Whether to enforce orthogonality
            
        Returns:
            iso: An Isometry object
        """
        assert isinstance(rep, np.ndarray)
        assert rep.shape == (12,)
        rot = rep[0:9].reshape(3, 3).T
        if ortho:
            rot = project_orthogonal(rot)
        return Isometry(q=Quaternion(matrix=rot), t=rep[9:12])

    def torch_matrices(self, device):
        """Convert to PyTorch tensors.
        
        Args:
            device: PyTorch device
            
        Returns:
            R: (3,3) rotation tensor
            t: (3,) translation tensor
        """
        import torch
        return torch.from_numpy(self.q.rotation_matrix).to(device).float(), \
               torch.from_numpy(self.t).to(device).float()

    @staticmethod
    def random():
        """Create a random isometry.
        
        Returns:
            iso: A random Isometry object
        """
        return Isometry(q=Quaternion.random(), t=np.random.random((3,)))
    
    @staticmethod
    def randn(sigma_degree: float = 0.0, sigma_t: float = 0.0):
        """Create a random isometry with normal distribution.
        
        Args:
            sigma_degree: Standard deviation for rotation (degrees)
            sigma_t: Standard deviation for translation
            
        Returns:
            iso: A random Isometry object
        """
        rand_axis = np.random.randn(3)
        rand_axis /= np.linalg.norm(rand_axis)
        rand_angle = np.random.randn() * sigma_degree
        rand_t = np.random.randn(3) * sigma_t
        return Isometry.from_axis_angle(axis=rand_axis, degrees=rand_angle, t=rand_t)

    def inv(self):
        """Get inverse transformation.
        
        Returns:
            iso: Inverse Isometry object
        """
        qinv = self.q.inverse
        return Isometry(q=qinv, t=-(qinv.rotate(self.t)))

    def to_gl_camera(self):
        """Convert to OpenGL camera convention. 
        Current camera convention is right-hand and opencv-style (RDF)
        Target camera convention is right-hand and opengl-style (RUB)
        
        Returns:
            iso: Converted Isometry object in OpenGL camera convention
        """
        return Isometry(q=(self.q * self.GL_POST_MULT), t=self.t)

    @staticmethod
    def look_at(source: Union[np.ndarray, list], target: Union[np.ndarray, list], up: Optional[np.ndarray] = None):
        """Create coordinate with camera at source, looking at target, and up direction is up. 
        The camera convention is right-hand and opencv-style

        Args:
            source: Camera position
            target: Look-at target point
            up: Up direction (default [0,1,0])
            
        Returns:
            iso: An Isometry object
        """
        if not isinstance(source, np.ndarray):
            source = np.asarray(source)
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)
        if not isinstance(up, np.ndarray) and up is not None:
            up = np.asarray(up)

        z_dir = target - source
        if np.linalg.norm(z_dir) < 1e-6:
            logger.warning(f"source {source} and target {target} are too close, use default z_dir.")
            z_dir = np.asarray([0.0, 0.0, 1.0])

        z_dir /= np.linalg.norm(z_dir)
        if up is None:
            up = np.asarray([0.0, 1.0, 0.0])
            if np.linalg.norm(np.cross(z_dir, up)) < 1e-6:
                up = np.asarray([1.0, 0.0, 0.0])
        else:
            up /= np.linalg.norm(up)
        x_dir = np.cross(z_dir, up)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)
        R = np.column_stack([x_dir, y_dir, z_dir])

        return Isometry(q=Quaternion(matrix=R, rtol=1.0, atol=1.0), t=source)

    @staticmethod
    def chordal_l2_mean(*isometries):
        """Compute L2 mean of multiple isometries.
        
        Args:
            *isometries: Variable number of Isometry objects
            
        Returns:
            iso: Mean Isometry object
        """
        # Ref: Rotation Averaging, Hartley et al. 2013
        # The Isometry distance is defined as squared of chordal distance (Frobenius form)
        # For other distance like geodesic distance or other norms like L1-norm, no closed-form is provided.
        assert len(isometries) >= 1
        t_mean = sum([iso.t for iso in isometries]) / len(isometries)
        q_mean_mat = sum([np.einsum('i,j->ij', iso.q.q, iso.q.q) for iso in isometries]) / len(isometries)
        w, v = np.linalg.eigh(q_mean_mat)
        q_mean = Quaternion(v[:, -1])
        return Isometry(q=q_mean, t=t_mean)

    @staticmethod
    def from_point_flow(xyz: np.ndarray, flow: np.ndarray):
        """Estimate rigid transformation from point flow.
        
        Args:
            xyz: (N,3) point cloud
            flow: (N,3) flow vectors
            
        Returns:
            iso: Estimated Isometry object
        """
        # So that this @ xyz = xyz + flow
        pc1 = np.copy(xyz)
        pc2 = np.copy(xyz + flow)
        pc1_mean = np.mean(pc1, axis=0)
        pc2_mean = np.mean(pc2, axis=0)
        pc1 -= pc1_mean;
        pc2 -= pc2_mean
        cov = pc1.T @ pc2
        u, _, vh = np.linalg.svd(cov, compute_uv=True)
        R = vh.T @ u.T
        if np.linalg.det(R) < 0.0:
            u[:, -1] = -u[:, -1]
            R = vh.T @ u.T
        t = pc2_mean - R @ pc1_mean
        return Isometry.from_matrix(R, t)

    @staticmethod
    def copy(iso):
        """Create a deep copy.
        
        Args:
            iso: Isometry object to copy
            
        Returns:
            iso: New Isometry object
        """
        return copy.deepcopy(iso)

    def adjoint_matrix(self):
        # Please refer to VisualSLAM-14 Equ. (4.49), result will be 6x6 matrix
        R = self.q.rotation_matrix
        twR = so3_wedge(self.t) @ R
        adjoint = np.zeros((6, 6))
        adjoint[0:3, 0:3] = R
        adjoint[3:6, 3:6] = R
        adjoint[0:3, 3:6] = twR
        return adjoint

    def log(self):
        """Get se(3) vector from isometry
        
        Returns:
            xi: (6,) twist vector [rho, phi] in se(3)
        """
        phi = so3_log(self.q.rotation_matrix)
        rho = so3_inv_left_jacobian(phi) @ self.t
        return np.hstack([rho, phi])

    def tangent(self, prev_iso, next_iso):
        t = 0.5 * (next_iso.t - prev_iso.t)
        l1 = Quaternion.log((self.q.inverse * prev_iso.q).normalised)
        l2 = Quaternion.log((self.q.inverse * next_iso.q).normalised)
        e = Quaternion()
        e.q = -0.25 * (l1.q + l2.q)
        e = self.q * Quaternion.exp(e)
        return Isometry(t=t, q=e)

    def __matmul__(self, other):
        """Matrix multiplication operator.
        
        Args:
            other: can be Open3D object,Isometry, ScaledIsometry, numpy array or PyTorch tensor
            
        Returns:
            Transformed object
        """
        # "@" operator: other can be (N,3) or (3,).
        if hasattr(other, "transform"):
            # Open3d stuff...
            if hasattr(other, "to"):
                other = other.clone()
            else:
                other = copy.deepcopy(other)
            if "OrientedBoundingBox" in str(type(other)):
                other = other.rotate(self.q.rotation_matrix, center=np.zeros((3, 1)))
                other.translate(self.t)
                return other
            return other.transform(self.matrix)
        
        if hasattr(other, "device"):  # Torch tensor
            th_R, th_t = self.torch_matrices(other.device)
            assert other.size(-1) == 3
            res = other.view(-1, 3) @ th_R.t() + th_t.unsqueeze(0)
            return res.view(other.shape)

        if isinstance(other, Isometry) or isinstance(other, ScaledIsometry):
            return self.dot(other)
        
        if type(other) != np.ndarray or other.ndim == 1:
            return self.q.rotate(other) + self.t
        
        else:
            # Numpy arrays
            return (other @ self.q.rotation_matrix.T + self.t[np.newaxis, :]).astype(other.dtype)

    @staticmethod
    def interpolate(source, target, alpha, cartesian: bool = True):
        """Interpolate between two isometries.
        
        Args:
            source: Source Isometry
            target: Target Isometry
            alpha: Interpolation parameter [0,1]
            cartesian: Whether to use cartesian interpolation, if True, use slerp to interpolate rotation, otherwise interpolate in se(3) space
            
        Returns:
            iso: Interpolated Isometry
        """
        if cartesian:
            iquat = Quaternion.slerp(source.q, target.q, alpha)
            it = source.t * (1 - alpha) + target.t * alpha
            return Isometry(q=iquat, t=it)
        else:
            v = (source.inv() @ target).log()
            return source @ Isometry.from_twist(alpha * v)

    def validified(self):
        """Fix invalid values in isometry.
        
        Returns:
            iso: Valid Isometry object
        """
        q = self.q.normalised
        if np.any(np.isnan(q.q)) or np.any(np.isinf(q.q)):
            logger.warning("Isometry.validified get invalid q.")
            q = Quaternion()

        t = np.copy(self.t)
        if np.any(np.isnan(t)) or np.any(np.isinf(t)):
            logger.warning("Isometry.validified get invalid t.")
            t = np.zeros((3,))

        return Isometry(q=q, t=t)


class ScaledIsometry:
    """
    s (Rx + t), applied outside
    """

    def __init__(self, s=1.0, iso: Isometry = None):
        if iso is None:
            iso = Isometry()
        self.iso = iso
        self.s = s

    @property
    def t(self):
        return self.iso.t

    @property
    def q(self):
        return self.iso.q

    def __repr__(self):
        return f"ScaledIsometry: t = {self.t}, q = {self.q}, s = {self.s}"

    @property
    def rotation(self):
        return self.iso.rotation

    @staticmethod
    def from_inner_form(iso: Isometry, scale: float):
        # inner form is sRx + t
        new_t = iso.t / scale if scale != 0.0 else iso.t
        return ScaledIsometry(s=scale, iso=Isometry(q=iso.q, t=new_t))

    @staticmethod
    def from_matrix(mat, ortho=False):
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (4, 4) or mat.shape == (3, 4)
        mat = mat.astype(float)
        rot_mat = mat[:3, :3]
        scale_vec = np.sqrt(np.diag(rot_mat.T @ rot_mat))
        scale_value = np.mean(scale_vec)
        assert np.allclose(scale_vec, np.full_like(scale_vec, scale_value))
        rot_mat /= scale_value
        t_component = mat[:3, 3] / scale_value
        if ortho:
            rot_mat = project_orthogonal(rot_mat)
        return ScaledIsometry(scale_value.item(), Isometry(q=Quaternion(matrix=rot_mat), t=t_component))

    @property
    def matrix(self):
        mat = self.iso.matrix
        mat[:3] *= self.s
        return mat

    def inv(self):
        qinv = self.q.inverse
        return ScaledIsometry(s=1.0 / self.s, iso=Isometry(q=qinv, t=-(qinv.rotate(self.t) * self.s)))

    def __matmul__(self, other):
        if isinstance(other, Isometry) or isinstance(other, ScaledIsometry):
            return self.dot(other)
        elif hasattr(other, "transform"):
            other = copy.deepcopy(other)
            other.transform(self.iso.matrix)
            other.scale(self.s, center=np.zeros(3))
            return other
        else:
            res = self.iso @ other
            return self.s * res


class BoundingBox:
    """A class representing an oriented bounding box in 3D space.
    
    This class defines a bounding box with a position, orientation, and extent (dimensions).
    The box is defined by an isometry (position and orientation) and extent (size in each dimension).
    
    Attributes:
        VERT_SEQ (np.ndarray): Pre-computed unit cube vertex coordinates (8 vertices x 3 coordinates)
    """

    VERT_SEQ = np.array([
        [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]
    ])

    def __init__(self, iso: Isometry = None, extent: np.ndarray = None) -> None:
        """Initialize a bounding box.
        
        Args:
            iso (Isometry, optional): Isometry defining position and orientation. Defaults to identity.
            extent (np.ndarray, optional): 3D array defining box dimensions. Defaults to unit cube.
        """
        if iso is None:
            iso = Isometry()
        if extent is None:
            extent = np.ones((3,))
        self.iso = iso
        self.extent = np.asarray(extent)

    @property
    def vertices(self) -> np.ndarray:
        """Get the coordinates of the box's vertices.
        
        Returns:
            np.ndarray: 8x3 array of vertex coordinates in world space
        """
        # Scale unit vertices by extent and transform by isometry
        return self.iso @ (self.VERT_SEQ * self.extent[np.newaxis, :])
    
    def __repr__(self) -> str:
        """String representation of the bounding box.
        
        Returns:
            str: String describing the box's isometry and extent
        """
        return f"BoundingBox: iso = {self.iso}, extent = {self.extent}"
    
    def inside(self, pos: np.ndarray) -> np.ndarray:
        """Test if point(s) lie inside the bounding box.
        
        Args:
            pos (np.ndarray): Nx3 array of points to test
            
        Returns:
            np.ndarray: Boolean array indicating which points are inside
        """
        pos = np.asarray(pos)
        # Transform points to box's local coordinates and compare against extent
        return np.all(np.abs(self.iso.inv() @ pos) <= self.extent / 2.0, axis=-1)
    
    def scaled(self, factor: float) -> 'BoundingBox':
        """Create a new scaled version of this bounding box.
        
        Args:
            factor (float): Scale factor to apply to box dimensions
            
        Returns:
            BoundingBox: New bounding box with scaled extent
        """
        return BoundingBox(iso=self.iso, extent=self.extent * factor)


def _isometry_dot(left, right):
    """Compute the dot product between two isometries or scaled isometries.
    
    This function handles multiplication between Isometry and ScaledIsometry objects,
    combining both their rotations/translations and scales appropriately.
    
    Args:
        left: An Isometry or ScaledIsometry object representing the left operand
        right: An Isometry or ScaledIsometry object representing the right operand
        
    Returns:
        A new Isometry or ScaledIsometry object representing the product.
        Returns Isometry if both inputs are unscaled, ScaledIsometry otherwise.
        
    Raises:
        NotImplementedError: If either input is not an Isometry or ScaledIsometry
    """
    # Extract isometry and scale components from left operand
    if isinstance(left, Isometry):
        left_iso, left_scale = left, 1.0
    elif isinstance(left, ScaledIsometry):
        left_iso, left_scale = left.iso, left.s
    else:
        raise NotImplementedError

    # Extract isometry and scale components from right operand 
    if isinstance(right, Isometry):
        right_iso, right_scale = right, 1.0
    elif isinstance(right, ScaledIsometry):
        right_iso, right_scale = right.iso, right.s
    else:
        raise NotImplementedError

    # Special case: both inputs are unscaled isometries
    if left_scale == right_scale == 1.0:
        return Isometry(q=(left_iso.q * right_iso.q), t=(left_iso.q.rotate(right_iso.t) + left_iso.t))
    # General case: handle scaled isometries
    else:
        return ScaledIsometry(
            s=left_scale * right_scale,  # Multiply scales
            iso=Isometry(q=left_iso.q * right_iso.q,  # Combine rotations
                         t=(right_scale * left_iso.q.rotate(right_iso.t) + left_iso.t) / right_scale)  # Combine translations with scale
        )


Isometry.dot = _isometry_dot
ScaledIsometry.dot = _isometry_dot
