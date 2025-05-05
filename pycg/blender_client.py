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
import numpy as np
from pycg.isometry import Isometry
from pycg.exp import logger
from pyquaternion import Quaternion
from multiprocessing.managers import BaseManager

from dataclasses import dataclass

@dataclass
class BlenderConfig:
    exec_path: str = "auto"
    port: int = 10240
    debug: bool = False
    force_background: bool = False
    no_cuda: bool = False
    no_optix: bool = False


config = BlenderConfig()


BLENDER_EXEC_PATH = None


def find_blender(force_refind: bool = False):
    """Find and return the path to the Blender executable.
    
    This function attempts to locate the Blender executable by checking:
    1. The configured exec_path if not "auto"
    2. The BLENDER_HOME environment variable
    3. The system PATH using shutil.which()

    Args:
        force_refind (bool): If True, ignore cached path and search again. Defaults to False.

    Returns:
        str: Path to the Blender executable

    Raises:
        FileNotFoundError: If Blender executable cannot be found
    """
    global BLENDER_EXEC_PATH
    # Return cached path if available and not forcing refind
    if BLENDER_EXEC_PATH is not None and not force_refind:
        return BLENDER_EXEC_PATH

    # Try finding Blender in different locations
    if config.exec_path != "auto":
        BLENDER_EXEC_PATH = config.exec_path
    elif 'BLENDER_HOME' in os.environ:
        BLENDER_EXEC_PATH = os.environ['BLENDER_HOME'] 
    else:
        # Search system PATH
        BLENDER_EXEC_PATH = shutil.which('blender')

    if BLENDER_EXEC_PATH is None:
        raise FileNotFoundError("Could not find Blender. Please specify its path in BLENDER_HOME environment variable!")

    logger.info(f"Find blender in {BLENDER_EXEC_PATH}")
    return BLENDER_EXEC_PATH


class BlenderConnection:
    """Manages connection and communication with a Blender instance.

    This class handles starting up Blender, establishing a connection via sockets,
    and sending/receiving messages between Python and Blender.

    The communication is handled through Python's multiprocessing.managers.BaseManager,
    which provides a high-level interface for process communication. Two queues are used:
    - res_queue: For receiving responses from Blender
    - cmd_queue: For sending commands to Blender

    Args:
        port (int): Port number to use for the connection with Blender
    """
    def __init__(self, port):
        # Register queue types with BaseManager for inter-process communication (IPC)
        # These queues will be shared between the Python process and Blender process
        BaseManager.register('res_queue')  # Queue for responses from Blender
        BaseManager.register('cmd_queue')  # Queue for commands to Blender
        self.res_queue = None
        self.cmd_queue = None
        self.port = port
        # Create manager instance with authentication for secure IPC
        self.conn_manager = BaseManager(address=('localhost', self.port), authkey=b'pycg.blender')

    def _run_blender(self):
        """Start a new Blender instance if one is not already running on the specified port.
        
        Checks if port is available, then launches Blender with appropriate command line arguments.
        The Blender instance will run the server script that connects back to this client.
        """
        # Check if port is already in use
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        location = ("localhost", self.port)
        result_of_check = a_socket.connect_ex(location)
        if result_of_check != 0:
            # Check if we have a display available
            has_display = os.environ.get('DISPLAY', '') != ''
            # Build Blender command with required arguments
            run_commands = [
                find_blender(), '--python',
                str(Path(__file__).parent / "blender_server" / "main.py"),
                '--', '--port', str(self.port)
            ]
            # Add optional arguments based on config
            if not has_display or config.force_background:
                run_commands.insert(1, '--background')
            if config.debug:
                run_commands.insert(1, '--debug')
                run_commands.insert(1, '--debug-cycles')
            if config.no_cuda:
                run_commands.append('--no-cuda')
            if config.no_optix:
                run_commands.append('--no-optix')
            # Start Blender process
            pg = subprocess.Popen(run_commands)

    def _connect(self):
        """Establish connection with Blender instance.
        
        Attempts to connect multiple times with delays to allow Blender to start up.
        When Blender is launched, it will host a server that will be connected to by this client.
        Once connected, initializes the command and response queues for IPC.
        """
        self._run_blender()
        for trial_count in range(5):
            try:
                self.conn_manager.connect()
            except ConnectionRefusedError:
                print(f"Waiting for blender startup, trial count = {trial_count + 1}")
                time.sleep(2)
                continue
            break
        # Get queue proxies after connection is established
        self.res_queue = self.conn_manager.res_queue()
        self.cmd_queue = self.conn_manager.cmd_queue()

    def send(self, msg):
        """Send a message to Blender through the command queue.

        Args:
            msg: Message to send to Blender. Can be any picklable Python object.
        """
        if self.res_queue is None:
            self._connect()
        self.cmd_queue.put(msg)

    def receive(self, nowait=True):
        """Receive a message from Blender through the response queue.

        Args:
            nowait (bool): If True, return immediately if no message available.
                          If False, block until message is received.

        Returns:
            Message received from Blender, or None if nowait=True and no message available.
            Message can be any Python object that was sent from Blender.
        """
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
        """Close the connection with Blender.
        
        Polls the response queue until EOF is received, indicating Blender has closed.
        This ensures clean shutdown of the IPC connection.
        """
        # Poll until EOF
        while True:
            time.sleep(0.5)
            try:
                self.res_queue.get()
            except EOFError:
                break


_blender_connection = None


def get_blender_connection():
    global _blender_connection
    if _blender_connection is None:
        _blender_connection = BlenderConnection(config.port)
    return _blender_connection


def send_entity(geom, uuidx: str = None, pose: Isometry = None, attributes=None):
    """Send a basic geometry entity to be created in Blender.
    
    Supports sending PointCloud or TriangleMesh geometries with associated properties
    like colors, normals, textures etc.

    Args:
        geom: The geometry object (PointCloud or TriangleMesh)
        uuidx (str, optional): Unique ID for the entity
        pose (Isometry, optional): Pose transform for the entity
        attributes (dict, optional): Additional attributes for the entity

    Returns:
        str: UUID of the created entity in Blender

    Server interaction:
        Sends: {cmd: 'entity', geometry data and properties}
        Receives: {result: 'created', uuid: entity_uuid}
    """
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

    
    from pycg import o3d
    
    # point cloud
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
    # triangle mesh
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
            else:
                logger.warning(f"Texture data type {t.dtype} is not uint8, texture might be wrong")
            if t.shape[2] == 3:
                t = np.pad(t, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.0)
            assert t.shape[2] == 4
            textures.append(t)
        info_dict.update({
            'geometry_type': 'MESH',
            'vert': vert, 'faces': faces, 'vert_colors': vert_colors,
            'triangle_uvs': triangle_uvs, 'textures': textures
        })
    else:
        raise NotImplementedError(f"Geometry type {type(geom)} is not supported")

    get_blender_connection().send(info_dict)
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'created'
    return res_dict['uuid']


def send_entity_pose(uuidx: str, rotation_mode: str = None, rotation_value=None, location_value=None):
    assert rotation_mode in ['QUATERNION', 'XYZ', 'XZY', 'YZX', 'YXZ', 'ZXY', 'ZYX', 'AXIS_ANGLE']
    get_blender_connection().send({
        'cmd': 'entity_pose',
        'uuid': uuidx,
        'rotation_mode': rotation_mode,
        'rotation_value': rotation_value,
        'location_value': location_value
    })
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_add_keyframe(uuidx: str, frame: int, attribute: str, value):
    """Add a keyframe to an object's animation in Blender.

    Args:
        uuidx (str): UUID of the object to animate
        frame (int): Frame number to add the keyframe at
        attribute (str): when uuid is camera, attribute can be 'pose'
        value: Value to set at the keyframe. Can be a list/tuple of floats or an Isometry object.
            If Isometry, will be converted to translation + quaternion rotation.

    Returns:
        None

    Server interaction:
        Sends: {cmd: 'add_keyframe', uuid, attribute, frame, value}
        Receives: {result: 'added'}
    """
    if isinstance(value, Isometry):
        cam_q = value.q * Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)
        value = [*value.t.tolist(), *cam_q.q.tolist()]

    get_blender_connection().send({
        'cmd': 'add_keyframe',
        'uuid': uuidx,
        'attribute': attribute,
        'frame': int(frame),
        'value': value
    })
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'added'


def send_add_animation_fcurve(uuidx: str, data_path: str, index: int, mode: str, values: list):
    assert mode in ['constant', 'linear', 'bezier']
    get_blender_connection().send({
        'cmd': 'add_animation_fcurve',
        'uuid': uuidx,
        'data_path': data_path,
        'index': index,
        'mode': mode,
        'values': values
    })
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'added', f"Failed to add animation fcurve: {res_dict}"


def send_render(quality, save_path):
    """Render the current scene and save to file.

    Args:
        quality: Render quality/samples
        save_path: Path to save the rendered image

    Server interaction:
        Sends: {cmd: 'render', quality, path}
        Receives: {result: 'rendered'}
    """
    start_time = time.time()
    get_blender_connection().send({
        'cmd': 'render',
        'quality': quality,
        'path': save_path
    })
    res_dict = get_blender_connection().receive(nowait=False)
    end_time = time.time()
    if end_time - start_time > 30.0:
        logger.info(f"Render time: {end_time - start_time:.2f}s seems to be too long. "
                    "If you stuck at kernel compilation, "
                    "please consider delete precompiled kernels in BLENDER_HOME/scripts/addons_core/cycles/lib")
    assert res_dict['result'] == 'rendered'


def send_camera(pose: Isometry = None, intrinsic=None):
    """Update the Blender camera's pose and/or intrinsic parameters.

    Args:
        pose (Isometry, optional): New camera pose
        intrinsic (optional): Camera intrinsic parameters

    Server interaction:
        Sends: {cmd: 'camera', pose and/or intrinsic data} 
                note the pose is in the format of [x, y, z, qx, qy, qz, qw]
        Receives: {result: 'success'}
    """
    assert pose is not None or intrinsic is not None

    cmd_dict = {'cmd': 'camera'}
    if pose is not None:
        cam_q = pose.q * Quaternion(axis=[1.0, 0.0, 0.0], degrees=180.0)
        cmd_dict['pose'] = [*pose.t.tolist(), *cam_q.q.tolist()],

    if intrinsic is not None:
        cmd_dict['intrinsic'] = intrinsic.blender_attributes

    get_blender_connection().send(cmd_dict)
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_light(scene_light, uuidx: str = None):
    """Create or update a light in the Blender scene.

    Args:
        scene_light: SceneLight object, see render.py for more details
        uuidx (str, optional): Unique ID for the light

    Returns:
        str: UUID of the created light

    Server interaction:
        Sends: {cmd: 'light', light properties}
        Receives: {result: 'created', uuid: light_uuid}
    """
    cmd_dict = {'cmd': 'light'}
    if uuidx is not None:
        cmd_dict['uuid'] = uuidx
    cmd_dict['type'] = scene_light.type # SUN, POINT, AREA, SPOT
    cmd_dict['pos'] = scene_light.pose.t
    cmd_dict['rot'] = scene_light.pose.q.q
    cmd_dict.update(scene_light.attributes)
    get_blender_connection().send(cmd_dict)

    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'created'
    return res_dict['uuid']

def send_envmap(data: np.ndarray, rotation: list = None):
    """Set the environment map for the Blender scene.

    Args:
        data (np.ndarray): Environment map image data with shape (H, W, 3).
            Can be uint8 [0-255] or float [0-1].
        rotation (list, optional): XYZ euler angles in radians for rotating the envmap.
            Defaults to [0,0,0].

    Server interaction:
        Sends: {cmd: 'envmap', data: envmap_array, rotation: [x,y,z]}
        Receives: {result: 'success'}
    """
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
    get_blender_connection().send(cmd_dict)
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_eval(script: str):
    """Execute arbitrary Python code in the Blender environment.

    Args:
        script (str): Python code to execute in Blender

    Server interaction:
        Sends: {cmd: 'eval', script: code_string}
        Receives: {result: 'success'}
    """
    get_blender_connection().send({'cmd': 'eval', 'script': script})
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'success'


def get_entity(uuidx):
    """Retrieve information about an entity from Blender by its UUID.

    Args:
        uuidx (str): UUID of the entity to retrieve

    Returns:
        dict: Entity data containing geometry and property information

    Server interaction:
        Sends: {cmd: 'get_entity', uuid: entity_uuid}
        Receives: {result: 'got', ...entity_data}
    """
    get_blender_connection().send({
        "cmd": "get_entity",
        "uuid": uuidx
    })
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'got'
    del res_dict['result']
    return res_dict


def send_clear():
    """Clear all entities, lights and animations from the scene.

    Server interaction:
        Sends: {cmd: 'clear'}
        Receives: {result: 'success'}
    """
    get_blender_connection().send({'cmd': 'clear'})
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_save(path: Path):
    """Save the current Blender scene to a .blend file.

    Args:
        path (Path): Path where the .blend file should be saved

    Server interaction:
        Sends: {cmd: 'save', path: file_path}
        Receives: {result: 'success'}
    """
    get_blender_connection().send({'cmd': 'save', 'path': path.resolve()})
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] == 'success'


def send_detach():
    """Close the connection with the Blender server.

    Server interaction:
        Sends: {cmd: 'detach'}
        Server closes connection
    """
    global _blender_connection
    get_blender_connection().send({'cmd': 'detach'})
    get_blender_connection().close()
    _blender_connection = None


def poll_finished():
    """Poll the Blender server for a response and verify it succeeded.
    
    Waits for and receives a response from the Blender server, checking that
    the result was not 'failed'. Used to verify commands completed successfully.

    Raises:
        AssertionError: If the response result is 'failed'
    """
    res_dict = get_blender_connection().receive(nowait=False)
    assert res_dict['result'] != 'failed'


def poll_notified():
    """Poll the Blender server until a notification response is received.
    
    Continuously polls for responses from the server until one with result='notify' 
    is received. Other responses are logged but ignored.

    Raises:
        EOFError: If the connection is closed while polling
    """
    while True:
        try:
            res_dict = get_blender_connection().receive(nowait=False)
        except EOFError:
            logger.error("Connection aborted while polling")
            raise
        if res_dict['result'] == 'notify':
            break
        print(f"Poll-notified: expected notify, but get {res_dict}")
