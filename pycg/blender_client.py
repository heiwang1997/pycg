"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import queue
import time
import socket
from pathlib import Path
import subprocess
import os, shutil
import open3d as o3d
import numpy as np
from pycg.isometry import Isometry
from pycg.exp import logger
from pyquaternion import Quaternion
from multiprocessing.managers import BaseManager


BLENDER_EXEC_PATH = None


def find_blender(force_refind: bool = False):
    global BLENDER_EXEC_PATH
    if BLENDER_EXEC_PATH is not None and not force_refind:
        return BLENDER_EXEC_PATH

    if 'BLENDER_HOME' in os.environ:
        BLENDER_EXEC_PATH = os.environ['BLENDER_HOME']
    else:
        BLENDER_EXEC_PATH = shutil.which('blender')

    if BLENDER_EXEC_PATH is None:
        raise FileNotFoundError("Could not find Blender. Please specify its path in BLENDER_HOME environment variable!")

    logger.info(f"Find blender in {BLENDER_EXEC_PATH}")
    return BLENDER_EXEC_PATH


class BlenderConnection:
    def __init__(self, port=10240):
        BaseManager.register('res_queue')
        BaseManager.register('cmd_queue')
        self.res_queue = None
        self.cmd_queue = None
        self.port = port
        self.conn_manager = BaseManager(address=('localhost', self.port), authkey=b'pycg.blender')

    def _run_blender(self):
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        location = ("localhost", self.port)
        result_of_check = a_socket.connect_ex(location)
        if result_of_check != 0:
            pg = subprocess.Popen([find_blender(), '--python',
                                   str(Path(__file__).parent / "blender_server" / "main.py"),
                                   '--', '--port', str(self.port)])

    def _connect(self):
        self._run_blender()
        for trial_count in range(5):
            try:
                self.conn_manager.connect()
            except ConnectionRefusedError:
                print(f"Waiting for blender startup, trial count = {trial_count + 1}")
                time.sleep(2)
                continue
            break
        self.res_queue = self.conn_manager.res_queue()
        self.cmd_queue = self.conn_manager.cmd_queue()

    def send(self, msg):
        if self.res_queue is None:
            self._connect()
        self.cmd_queue.put(msg)

    def receive(self, nowait=True):
        if self.res_queue is None:
            self._connect()
        if nowait:
            try:
                return self.res_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            return self.res_queue.get()

    def close(self):
        # Poll until EOF
        while True:
            time.sleep(0.5)
            try:
                self.res_queue.get()
            except EOFError:
                break


_blender_connection = BlenderConnection()


def send_entity(geom, uuidx: str = None, pose: Isometry = None, attributes=None):
    if attributes is None:
        attributes = {}
    if pose is None:
        pose = Isometry()

    info_dict = {
        'cmd': 'entity',
        'pose': [*pose.t.tolist(), *pose.q.q.tolist()]
    }
    if uuidx is not None:
        info_dict['uuid'] = uuidx
    info_dict['attributes'] = attributes
    if isinstance(geom, o3d.geometry.PointCloud):
        xyz = np.asarray(geom.points).astype(np.float32)
        if geom.has_colors():
            rgb = np.concatenate([np.asarray(geom.colors),
                                  np.ones((len(geom.colors), 1))], axis=1).astype(np.float32)
        else:
            rgb = None
        normal = np.asarray(geom.normals).astype(np.float32) if geom.has_normals() else None
        info_dict.update({
            'geometry_type': 'PC',
            'xyz': xyz, 'rgb': rgb, 'normal': normal
        })
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        vert = np.asarray(geom.vertices).astype(np.float32)
        faces = np.asarray(geom.triangles).astype(np.int32)
        if geom.has_vertex_colors():
            vert_colors = np.concatenate([np.asarray(geom.vertex_colors),
                                  np.ones((len(geom.vertex_colors), 1))], axis=1).astype(np.float32)
        else:
            vert_colors = None
        if geom.has_triangle_uvs():
            triangle_uvs = np.asarray(geom.triangle_uvs).astype(np.float32)
        else:
            triangle_uvs = None
        textures = []
        for t in geom.textures:
            if t.is_empty():
                continue
            t = np.asarray(t)
            if t.dtype == np.uint8:
                t = t.astype(np.float16) / 255.
            textures.append(t)
        info_dict.update({
            'geometry_type': 'MESH',
            'vert': vert, 'faces': faces, 'vert_colors': vert_colors,
            'triangle_uvs': triangle_uvs, 'textures': textures
        })
    else:
        raise NotImplementedError

    _blender_connection.send(info_dict)
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'created'
    return res_dict['uuid']


def send_entity_pose(uuidx: str, rotation_mode: str = None, rotation_value=None, location_value=None):
    assert rotation_mode in ['QUATERNION', 'XYZ', 'XZY', 'YZX', 'YXZ', 'ZXY', 'ZYX', 'AXIS_ANGLE']
    _blender_connection.send({
        'cmd': 'entity_pose',
        'uuid': uuidx,
        'rotation_mode': rotation_mode,
        'rotation_value': rotation_value,
        'location_value': location_value
    })
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_add_keyframe(uuidx: str, frame: int, attribute: str, value):
    if isinstance(value, Isometry):
        cam_q = value.q * Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)
        value = [*value.t.tolist(), *cam_q.q.tolist()]

    _blender_connection.send({
        'cmd': 'add_keyframe',
        'uuid': uuidx,
        'attribute': attribute,
        'frame': int(frame),
        'value': value
    })
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'added'


def send_add_animation_fcurve(uuidx: str, data_path: str, index: int, mode: str, values: list):
    assert mode in ['constant', 'linear', 'bezier']
    _blender_connection.send({
        'cmd': 'add_animation_fcurve',
        'uuid': uuidx,
        'data_path': data_path,
        'index': index,
        'mode': mode,
        'values': values
    })
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'added'


def send_render(quality, save_path):
    _blender_connection.send({
        'cmd': 'render',
        'quality': quality,
        'path': save_path
    })
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'rendered'


def send_camera(pose: Isometry = None, intrinsic=None):
    assert pose is not None or intrinsic is not None

    cmd_dict = {'cmd': 'camera'}
    if pose is not None:
        cam_q = pose.q * Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)
        cmd_dict['pose'] = [*pose.t.tolist(), *cam_q.q.tolist()],

    if intrinsic is not None:
        cmd_dict['intrinsic'] = intrinsic.blender_attributes

    _blender_connection.send(cmd_dict)
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_light(scene_light, uuidx: str = None):
    cmd_dict = {'cmd': 'light'}
    if uuidx is not None:
        cmd_dict['uuid'] = uuidx
    cmd_dict['type'] = scene_light.type
    cmd_dict['pos'] = scene_light.pose.t
    cmd_dict['rot'] = scene_light.pose.q.q
    cmd_dict.update(scene_light.attributes)
    _blender_connection.send(cmd_dict)

    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'created'
    return res_dict['uuid']


def send_envmap(data: np.ndarray, rotation: list = None):
    assert data.ndim == 3 and data.shape[2] == 3
    if data.dtype == np.uint8:
        data = data.astype(np.float64) / 255.

    cmd_dict = {'cmd': 'envmap'}
    cmd_dict['data'] = data
    if rotation is None:
        # XYZ euler angle (in radian)
        cmd_dict['rotation'] = [0.0, 0.0, 0.0]
    else:
        cmd_dict['rotation'] = rotation
    _blender_connection.send(cmd_dict)
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_eval(script: str):
    _blender_connection.send({'cmd': 'eval', 'script': script})
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'success'


def get_entity(uuidx):
    _blender_connection.send({
        "cmd": "get_entity",
        "uuid": uuidx
    })
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'got'
    del res_dict['result']
    return res_dict


def send_clear():
    _blender_connection.send({'cmd': 'clear'})
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_save(path: Path):
    _blender_connection.send({'cmd': 'save', 'path': path.resolve()})
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_detach():
    global _blender_connection
    _blender_connection.send({'cmd': 'detach'})
    _blender_connection.close()
    _blender_connection = BlenderConnection()


def poll_finished():
    res_dict = _blender_connection.receive(nowait=False)
    assert res_dict['result'] != 'failed'


def poll_notified():
    while True:
        try:
            res_dict = _blender_connection.receive(nowait=False)
        except EOFError:
            logger.error("Connection aborted while polling")
            raise
        if res_dict['result'] == 'notify':
            break
        print(f"Poll-notified: expected notify, but get {res_dict}")
