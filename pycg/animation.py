from enum import Enum

import weakref
import numpy as np
from copy import deepcopy
from pycg.isometry import Isometry, Quaternion
from pycg import blender_client as blender
from collections import defaultdict


def quaternion_uniform_flip(q_list):
    """
    Flip the directions of quaternions to prevent from not taking the nearest path.
    """
    new_q_list = deepcopy(q_list)
    if len(q_list) < 2:
        return new_q_list

    last_q = new_q_list[0]
    for qi in range(1, len(new_q_list)):
        if np.dot(new_q_list[qi].q, last_q.q) < 0:
            new_q_list[qi].q = -new_q_list[qi].q
        last_q = new_q_list[qi]

    return new_q_list


class BaseInterpolator:
    def __init__(self):
        self._keyframes = {}
        self._precomputed = False       # For some interpolators this is needed.

    def set_keyframe(self, t, value):
        self._keyframes[t] = value
        self._precomputed = False

    def remove_keyframe(self, t):
        del self._keyframes[t]
        self._precomputed = False

    def ordered_keyframes(self):
        kf = list(self._keyframes.items())
        return sorted(kf, key=lambda t: t[0])

    def ordered_times(self):
        return sorted(list(self._keyframes.keys()))

    def get_value(self, t):
        raise NotImplementedError

    def get_first_t(self):
        if len(self._keyframes) == 0:
            return None
        return min(list(self._keyframes.keys()))

    def get_last_t(self):
        if len(self._keyframes) == 0:
            return None
        return max(list(self._keyframes.keys()))

    def send_blender(self, uuidx: str, data_path: str, index: int):
        raise NotImplementedError


class ConstantInterpolator(BaseInterpolator):
    """
    Values are taken as the first predecessor. If not exist, then use the first successor.
        - USD: Held interpolation type
        - Blender: Constant keyframe
    """
    def get_value(self, t):
        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')
        return self._keyframes[all_times[nki - 1]]

    def send_blender(self, uuidx: str, data_path: str, index: int):
        blender.send_add_animation_fcurve(uuidx, data_path, index, 'constant', self.ordered_keyframes())


class LinearInterpolator(BaseInterpolator):
    """
    Values are linearly interpolated from nearby keyframes.
        - USD: Linear interpolation
        - Blender: Linear keyframe
    """
    def get_value(self, t):
        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]

        nki = np.searchsorted(all_times, t, side='right')
        dt = all_times[nki] - all_times[nki - 1]

        alpha = (t - all_times[nki - 1]) / dt
        prev_value = self._keyframes[all_times[nki - 1]]
        next_value = self._keyframes[all_times[nki]]
        return (1 - alpha) * prev_value + alpha * next_value


class LinearQuaternionInterpolator(BaseInterpolator):
    """
    Specially designed for quaternions.
        - USD: Slerp interpolation as shown:
            https://graphics.pixar.com/usd/release/api/interpolation_8h.html#USD_LINEAR_INTERPOLATION_TYPES
        - Blender: does not support quaternion slerp yet.
    """
    def set_keyframe(self, t, value):
        assert isinstance(value, Quaternion)
        super().set_keyframe(t, value)

    def precompute(self):
        kfs = self.ordered_keyframes()
        new_qs = quaternion_uniform_flip([t[1] for t in kfs])
        for new_q, (time, _) in zip(new_qs, kfs):
            self._keyframes[time] = new_q
        self._precomputed = True

    def get_value(self, t):
        if not self._precomputed:
            self.precompute()

        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')
        dt = all_times[nki] - all_times[nki - 1]

        alpha = (t - all_times[nki - 1]) / dt
        prev_q = self._keyframes[all_times[nki - 1]]
        next_q = self._keyframes[all_times[nki]]
        return Quaternion.slerp(prev_q, next_q, alpha)


class BezierInterpolator(BaseInterpolator):
    """
    Cubic bezier curve (https://en.wikipedia.org/wiki/B%C3%A9zier_curve), used nearly everywhere in DCC apps!
    Given P0 P1 P2 P3, where P1 and P2 are (half-)handles of P0 and P3, then the curve goes:
        B(t) = lerp{ lerp[ lerp(P0,P1), lerp(P1,P2) ], lerp[ lerp(P1,P2), lerp(P2,P3) ] }  -- 6 lerps needed!
    where
        lerp(Px, Py) = (1-t) Px + t Py.
    this is equivalent to:
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3      0 <= t <= 1

    - Blender: Auto-clamped keyframes, with 'Handle Smoothing' set to 'None'.
        See: blender source code: source/blender/blenkernel/intern/curve.cc - calchandleNurb_intern.
        (The continuous acceleration model is hard to implement here because it involves solving a linear equation)
    - USD: not supported.
    """
    class BezierTriplet:
        def __init__(self):
            # vec[0], vec[1], vec[2] are 2D coords of left-handle, middle, right-handle.
            self.vec = np.zeros((3, 2))

    def __init__(self):
        super().__init__()
        self._bezier_triplets = []

    def precompute(self):
        self._bezier_triplets = []

        kfs = self.ordered_keyframes()
        for (prev_t, prev_value), (cur_t, cur_value), (next_t, next_value) in zip(
                [(None, None)] + kfs[:-1], kfs, kfs[1:] + [(None, None)]):
            cur_triplet = BezierInterpolator.BezierTriplet()
            self._bezier_triplets.append(cur_triplet)

            p2 = np.array([cur_t, cur_value])
            cur_triplet.vec[1] = p2

            if prev_t is not None and next_t is not None:
                p1 = np.array([prev_t, prev_value])
                p3 = np.array([next_t, next_value])
                len_a = cur_t - prev_t
                len_b = next_t - cur_t
                # Average slope with x length = 2.
                tvec = (p3 - p2) / len_b + (p2 - p1) / len_a
                # Never make the handle lengths too different
                len_a = min(len_a, 5.0 * len_b)
                len_b = min(len_b, 5.0 * len_a)
                cur_triplet.vec[0] = p2 - (tvec / (2 * 2.5614)) * len_a
                cur_triplet.vec[2] = p2 - (tvec / (2 * 2.5614)) * len_b
            else:
                if prev_t is None and next_t is None:
                    delta_t = 1.0
                elif prev_t is None:
                    delta_t = (next_t - cur_t) / 2.5614
                elif next_t is None:
                    delta_t = (cur_t - prev_t) / 2.5614
                cur_triplet.vec[0] = [cur_t - delta_t, cur_value]
                cur_triplet.vec[2] = [cur_t + delta_t, cur_value]
                continue

            ydiff1 = prev_value - cur_value
            ydiff2 = next_value - cur_value
            left_violate, right_violate = False, False
            if (ydiff1 <= 0.0 and ydiff2 <= 0.0) or (ydiff1 >= 0.0 and ydiff2 >= 0.0):
                # Clamp at extrema.
                cur_triplet.vec[0][1] = cur_value
                cur_triplet.vec[2][1] = cur_value
            else:
                # Left handle exceeds previous kf's value...
                if (ydiff1 <= 0.0 and prev_value > cur_triplet.vec[0][1]) or \
                        (ydiff1 > 0.0 and prev_value < cur_triplet.vec[0][1]):
                    cur_triplet.vec[0][1] = prev_value
                    left_violate = True
                # Right handle exceeds next kf's value...
                if (ydiff1 <= 0.0 and next_value < cur_triplet.vec[2][1]) or \
                        (ydiff1 > 0.0 and next_value > cur_triplet.vec[2][1]):
                    cur_triplet.vec[2][1] = next_value
                    right_violate = True
            h1_x = cur_triplet.vec[0][0] - p2[0]
            h2_x = p2[0] - cur_triplet.vec[2][0]
            if left_violate:        # <-- take left's slope if both are violated
                cur_triplet.vec[2][1] = p2[1] + ((p2[1] - cur_triplet.vec[0][1]) / h1_x) * h2_x
            elif right_violate:
                cur_triplet.vec[0][1] = p2[1] + ((p2[1] - cur_triplet.vec[2][1]) / h2_x) * h1_x

        self._precomputed = True

    def get_value(self, t):
        if not self._precomputed:
            self.precompute()

        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')

        p0 = self._bezier_triplets[nki - 1].vec[1]
        p1 = self._bezier_triplets[nki - 1].vec[2]
        p2 = self._bezier_triplets[nki].vec[0]
        p3 = self._bezier_triplets[nki].vec[1]

        o3t = -p0 + 3 * p1 - 3 * p2 + p3
        o2t = 3 * p0 - 6 * p1 + 3 * p2
        o1t = -3 * p0 + 3 * p1
        o0t = p0 - t

        from scipy.optimize import fsolve
        f = lambda x: o3t[0] * x ** 3 + o2t[0] * x ** 2 + o1t[0] * x + o0t[0]
        fy = lambda x: o3t[1] * x ** 3 + o2t[1] * x ** 2 + o1t[1] * x + o0t[1]
        fprime = lambda x: 3 * o3t[0] * x ** 2 + 2 * o2t[0] * x + o1t[0]
        t_target = fsolve(f, [0.5], fprime=fprime)[0]

        return fy(t_target)

    def send_blender(self, uuidx: str, data_path: str, index: int):
        if not self._precomputed:
            self.precompute()

        blender.send_add_animation_fcurve(uuidx, data_path, index, 'bezier',
                                          [t.vec for t in self._bezier_triplets])


class SquadQuaternionInterpolator(BaseInterpolator):
    """
    A spline [Shoemake, 1985] that evaluates faster than Bezier with fewer slerps:
    This is used in QGLViewer:
        B(t) = slerp[ slerp(P0, P3, t), slerp(P1, P2, t), 2t(1-t) ]  -- 3 slerps needed.
    Note that this is not equivalent to BezierQuaternionCurve, see '../examples/anime.py' for more details.
    Also: https://devtalk.blender.org/t/quaternion-interpolation/15883/15
    BezierQuaternionCurve is not implemented -- the control points are hard to choose!

    Neither supported in blender nor USD, but should be the best choice to interpolate arbitrary poses.
    """
    class KeyFrame:
        def __init__(self):
            self.iso = Quaternion()
            self.tangent = Quaternion()

    def __init__(self):
        super().__init__()
        self._q_keyframes = []

    def precompute(self):
        self._q_keyframes = []

        kfs = self.ordered_keyframes()
        new_qs = quaternion_uniform_flip([t[1] for t in kfs])
        for new_q, (time, _) in zip(new_qs, kfs):
            self._keyframes[time] = new_q
        all_times = self.ordered_times()
        for ki, ti in enumerate(all_times):
            q_prev = self._keyframes[all_times[max(ki - 1, 0)]]
            q_next = self._keyframes[all_times[min(ki + 1, len(all_times) - 1)]]
            cur_q_kf = SquadQuaternionInterpolator.KeyFrame()
            cur_q_kf.iso = self._keyframes[ti]
            cur_q_kf.tangent = Isometry(q=cur_q_kf.iso).tangent(
                Isometry(q=q_prev), Isometry(q=q_next)).q
            self._q_keyframes.append(cur_q_kf)

        self._precomputed = True

    def get_value(self, t):
        if not self._precomputed:
            self.precompute()

        all_times = self.ordered_times()
        if t < all_times[0]:
            return self._keyframes[all_times[0]]
        if t >= all_times[-1]:
            return self._keyframes[all_times[-1]]
        nki = np.searchsorted(all_times, t, side='right')

        dt = all_times[nki] - all_times[nki - 1]
        alpha = (t - all_times[nki - 1]) / dt

        prev_kf = self._q_keyframes[nki - 1]
        next_kf = self._q_keyframes[nki]

        qab = Quaternion.slerp(prev_kf.iso, next_kf.iso, alpha)
        qtgt = Quaternion.slerp(prev_kf.tangent, next_kf.tangent, alpha)
        q = Quaternion.slerp(qab, qtgt, 2.0 * alpha * (1.0 - alpha))

        return q


class InterpType(Enum):
    CONSTANT = 0
    LINEAR = 1
    BEZIER = 2


class AnimatorBase:
    def __init__(self):
        self.interpolators = {}
        # For better export-ability
        self.keyframes = {}

    def add_interpolator(self, name, inst):
        assert name not in self.interpolators.keys()
        self.interpolators[name] = inst
        return self.interpolators[name]

    def set_keyframe(self, t, value):
        raise NotImplementedError

    def remove_keyframe(self, t):
        for itp in self.interpolators.values():
            itp.remove_keyframe(t)

    def get_first_t(self):
        first_t_list = [itp.get_first_t() for itp in self.interpolators.values()]
        first_t_list = [t for t in first_t_list if t is not None]
        return min(first_t_list) if len(first_t_list) > 0 else None

    def get_last_t(self):
        last_t_list = [itp.get_last_t() for itp in self.interpolators.values()]
        last_t_list = [t for t in last_t_list if t is not None]
        return max(last_t_list) if len(last_t_list) > 0 else None

    def get_value(self, t):
        raise NotImplementedError


class FreePoseAnimator(AnimatorBase):

    class RotationType(Enum):
        BLENDER = 0     # separate 4 components of quaternion and re-normalize them afterwards.
        MANIFOLD = 1    # interpolate on manifold, will give better linearity / quadratic-ity

    def __init__(self, interp_type: InterpType, rotation_type: RotationType = RotationType.BLENDER):
        super().__init__()
        self.interp_type = interp_type
        self.rotation_type = rotation_type
        if self.interp_type == InterpType.LINEAR:
            self.interp_x = self.add_interpolator('x', LinearInterpolator())
            self.interp_y = self.add_interpolator('y', LinearInterpolator())
            self.interp_z = self.add_interpolator('z', LinearInterpolator())
            if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
                self.interp_qw = self.add_interpolator('qw', LinearInterpolator())
                self.interp_qx = self.add_interpolator('qx', LinearInterpolator())
                self.interp_qy = self.add_interpolator('qy', LinearInterpolator())
                self.interp_qz = self.add_interpolator('qz', LinearInterpolator())
            else:
                self.interp_q = self.add_interpolator('q', LinearQuaternionInterpolator())
        elif self.interp_type == InterpType.BEZIER:
            self.interp_x = self.add_interpolator('x', BezierInterpolator())
            self.interp_y = self.add_interpolator('y', BezierInterpolator())
            self.interp_z = self.add_interpolator('z', BezierInterpolator())
            if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
                self.interp_qw = self.add_interpolator('qw', BezierInterpolator())
                self.interp_qx = self.add_interpolator('qx', BezierInterpolator())
                self.interp_qy = self.add_interpolator('qy', BezierInterpolator())
                self.interp_qz = self.add_interpolator('qz', BezierInterpolator())
            else:
                self.interp_q = self.add_interpolator('q', SquadQuaternionInterpolator())

    def set_keyframe(self, t, value: Isometry):
        assert isinstance(value, Isometry)

        self.interp_x.set_keyframe(t, value.t[0])
        self.interp_y.set_keyframe(t, value.t[1])
        self.interp_z.set_keyframe(t, value.t[2])

        if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
            self.interp_qw.set_keyframe(t, value.q.q[0])
            self.interp_qx.set_keyframe(t, value.q.q[1])
            self.interp_qy.set_keyframe(t, value.q.q[2])
            self.interp_qz.set_keyframe(t, value.q.q[3])
        else:
            self.interp_q.set_keyframe(t, value.q)

    def get_value(self, t):
        new_x = self.interp_x.get_value(t)
        new_y = self.interp_y.get_value(t)
        new_z = self.interp_z.get_value(t)
        if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
            new_qw = self.interp_qw.get_value(t)
            new_qx = self.interp_qx.get_value(t)
            new_qy = self.interp_qy.get_value(t)
            new_qz = self.interp_qz.get_value(t)
            new_q = np.array([new_qw, new_qx, new_qy, new_qz])
            new_q = Quaternion(new_q / (np.linalg.norm(new_q) + 1.0e-6))
        else:
            new_q = self.interp_q.get_value(t)
        new_iso = Isometry(q=new_q, t=[new_x, new_y, new_z])
        return new_iso

    def send_blender(self, uuidx: str):
        self.interp_x.send_blender(uuidx, 'location', 0)
        self.interp_y.send_blender(uuidx, 'location', 1)
        self.interp_z.send_blender(uuidx, 'location', 2)
        if self.rotation_type == FreePoseAnimator.RotationType.BLENDER:
            self.interp_qw.send_blender(uuidx, 'rotation_quaternion', 0)
            self.interp_qx.send_blender(uuidx, 'rotation_quaternion', 1)
            self.interp_qy.send_blender(uuidx, 'rotation_quaternion', 2)
            self.interp_qz.send_blender(uuidx, 'rotation_quaternion', 3)
        else:
            self.interp_q.send_blender(uuidx, 'rotation_quaternion', None)


class SpinPoseAnimator:
    def __init__(self, interp_type: InterpType, spin_axis: str = '+X'):
        pass


class SceneAnimator:
    def __init__(self, scene):
        self.scene = scene
        # uuid -> attributes -> animator
        #   or 'free' -> 'command' -> animator
        self.events = defaultdict(dict)

    def get_range(self):
        frame_max = -1e10
        frame_min = 1e10
        for obj_attrib in self.events.values():
            for obj_interp in obj_attrib.values():
                frame_max = max(frame_max, obj_interp.get_last_t())
                frame_min = min(frame_min, obj_interp.get_first_t())
        return frame_min, frame_max

    def send_blender(self):
        for obj_uuid, obj_attribs in self.events.items():
            for attrib_name, attrib_interp in obj_attribs.items():
                attrib_interp.send_blender(obj_uuid)

    def set_frame(self, t):
        for obj_uuid, obj_attribs in self.events.items():
            for attrib_name, attrib_interp in obj_attribs.items():
                attrib_val = attrib_interp.get_value(t)
                if obj_uuid == "relative_camera":
                    if attrib_name == "pose":
                        self.scene.relative_camera_pose = attrib_val
                    else:
                        raise NotImplementedError
                elif obj_uuid == "free":
                    eval(f"self.scene.{attrib_name} = attrib_val")
                elif attrib_name == "pose":
                    self.scene.objects[obj_uuid].pose = attrib_val
                elif attrib_name == "visible":
                    self.scene.objects[obj_uuid].visible = attrib_val

    def set_relative_camera(self, animator):
        self.events["relative_camera"]["pose"] = animator

    def get_relative_camera(self):
        return self.events["relative_camera"]["pose"]

    def set_object_pose(self, obj_uuid: str, animator):
        assert obj_uuid in self.scene.objects.keys()
        self.events[obj_uuid]["pose"] = animator

    def set_object_visibility(self, obj_uuid: str, animator):
        assert obj_uuid in self.scene.objects.keys()
        self.events[obj_uuid]["visible"] = animator

    def set_free_command(self, command: str, animator):
        self.events['free'][command] = animator

    # def add_object_event(self, frame_id: int, obj_uuid: str = None, obj_geom=None):
    #     """
    #     Add an object at @frame_id. If @obj_uuid is not None, will re-use previously-added object,
    #     otherwise will add new object as @obj_geom
    #     """
    #     if obj_geom is not None:
    #         obj_uuid = self.scene.add_object(obj_geom, return_name=True)
    #
    #     assert obj_uuid in self.scene.objects.keys()
    #     self.events[obj_uuid]["visible"].set_keyframe(frame_id, True)
    #
    # def remove_object_event(self, frame_id: int, obj_uuid: str):
    #     assert obj_uuid in self.scene.objects.keys()
    #     self.events[obj_uuid]["visible"].set_keyframe(frame_id, False)
    #
    # def move_object_event(self, frame_id: int, obj_uuid: str, new_pose: Isometry):
    #     assert obj_uuid in self.scene.objects.keys()
    #     self.events["camera"]["pose"].set_keyframe(frame_id, new_pose)

    # def render_blender(self, increment=1):
    #     self.scene._setup_blender()
    #
    #     for obj_uuid, obj_attribs in self.events.items():
    #         for attrib_name, attrib_interp in obj_attribs.items():
    #             attrib_keyframes = attrib_interp.get_keyframes()
    #             for kt, kval in attrib_keyframes:
    #                 blender.send_add_keyframe(obj_uuid, int(kt // increment), attrib_name, kval)
    #
    #     blender.poll_notified()
