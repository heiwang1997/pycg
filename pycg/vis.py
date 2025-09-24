"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""
import copy
import numpy as np
from pycg import o3d
import matplotlib.colors
import matplotlib.cm
import math

from pycg.isometry import Isometry, BoundingBox
from pycg.color import map_quantized_color
from pycg.exp import logger
from pyquaternion import Quaternion
from pathlib import Path
from typing import List, Dict, Tuple, Union, Iterable, Optional


try:
    from multiprocessing import Process, Queue

    # Get monitor size (for some reason, there's a segfault so I can only handle it this way...)
    def get_monitor_size(q: Queue):
        try:
            from screeninfo import get_monitors
            q.put(get_monitors()[0])
        except Exception:
            pass

    queue = Queue()
    p = Process(target=get_monitor_size, args=(queue, ))
    p.start()
    p.join(timeout=0.5)
    monitor_res = queue.get_nowait()

    MONITOR_HEIGHT = monitor_res.height
    MONITOR_WIDTH = monitor_res.width

except Exception:

    logger.warning("Failed to obtain monitor info!")
    MONITOR_WIDTH = MONITOR_HEIGHT = None


# Open3D Stuff
SERIALIZE_PROTO = {
    "PointCloud": {
        "class": o3d.geometry.PointCloud,
        "attributes": ["points", "colors", "normals"],
        "methods": [o3d.utility.Vector3dVector, o3d.utility.Vector3dVector, o3d.utility.Vector3dVector]
    },
    "LineSet": {
        "class": o3d.geometry.LineSet,
        "attributes": ["points", "lines", "colors"],
        "methods": [o3d.utility.Vector3dVector, o3d.utility.Vector2iVector, o3d.utility.Vector3dVector]
    },
    "TriangleMesh": {
        "class": o3d.geometry.TriangleMesh,
        "attributes": ["vertices", "vertex_colors", "vertex_normals", "triangles", "triangle_uvs", "triangle_normals"],
        "methods": [o3d.utility.Vector3dVector, o3d.utility.Vector3dVector, o3d.utility.Vector3dVector,
                    o3d.utility.Vector3iVector, o3d.utility.Vector2dVector, o3d.utility.Vector3dVector]
    }
}


class AnnotatedGeometry:
    def __init__(self, geom, annotations):
        self.geom = geom
        self.annotations = annotations
        self.attributes = {}


def convert_to_pickable(geom_obj):
    """Convert an Open3D geometry object to a pickable dictionary format.
    
    This function converts Open3D geometry objects (PointCloud, LineSet, TriangleMesh) into a 
    dictionary format that can be easily serialized/pickled. The conversion is done based on 
    the SERIALIZE_PROTO definitions.

    Args:
        geom_obj: An Open3D geometry object (PointCloud, LineSet, or TriangleMesh)

    Returns:
        dict: A dictionary containing the geometry type and its attributes as numpy arrays
        
    Raises:
        NotImplementedError: If the geometry type is not supported in SERIALIZE_PROTO
    """
    # Iterate through supported geometry types
    for type_name, type_def in SERIALIZE_PROTO.items():
        if isinstance(geom_obj, type_def["class"]):
            # Create dict with geometry type
            desc = {"type": type_name}
            # Convert each attribute to numpy array if it exists
            for attr in type_def["attributes"]:
                attr_val = getattr(geom_obj, attr)
                if attr_val is not None:
                    desc[attr] = np.asarray(attr_val)
            return desc
    raise NotImplementedError


def convert_from_pickable(geom_obj: dict):
    """Convert a pickable dictionary back to an Open3D geometry object.
    
    This function converts a dictionary containing geometry data back into the corresponding
    Open3D geometry object. The conversion is done based on the SERIALIZE_PROTO definitions.

    Args:
        geom_obj (dict): Dictionary containing geometry type and attributes as numpy arrays

    Returns:
        Open3D geometry object (PointCloud, LineSet, or TriangleMesh)
        
    Raises:
        AssertionError: If input is not a dict or geometry type is not supported
    """
    assert isinstance(geom_obj, dict)
    assert geom_obj["type"] in SERIALIZE_PROTO.keys()

    # Get the geometry type definition
    geom_type_def = SERIALIZE_PROTO[geom_obj['type']]
    # Create new geometry object
    geom = geom_type_def['class']()
    # Set each attribute using the corresponding creator method
    for attr_name, attr_creator in zip(geom_type_def["attributes"], geom_type_def["methods"]):
        if attr_name in geom_obj.keys():
            setattr(geom, attr_name, attr_creator(geom_obj[attr_name]))

    return geom


def ensure_from_torch(arr: Union["torch.Tensor", list, np.ndarray], dim: int = 2, remove_batch_dim: bool = True):
    """Convert input array to numpy array with specified dimensions.
    
    This function handles conversion of PyTorch tensors, lists, and numpy arrays to a numpy array
    with the specified number of dimensions. For PyTorch tensors, it can optionally handle batch dimensions.

    Args:
        arr (Union[torch.Tensor, list, np.ndarray]): Input array to convert
        dim (int, optional): Expected number of dimensions for output array. Defaults to 2.
        remove_batch_dim (bool, optional): Whether to handle batch dimension for PyTorch tensors. 
            If True, will extract first batch if batch dimension exists. Defaults to True.

    Returns:
        np.ndarray: Converted numpy array with specified dimensions

    Raises:
        AssertionError: If input tensor dimensions don't match expected dimensions
    """
    # Handle PyTorch tensor conversion
    if hasattr(arr, "detach"):
        if remove_batch_dim:
            # Check dimensions accounting for optional batch dim
            assert dim <= arr.ndim <= dim + 1, f"Torch tensor has dimension {arr.ndim} which is incorrect!"
            if arr.ndim == dim + 1:
                if arr.size(0) > 1:
                    print(f"Warning: input array is a torch tensor with batch dimension and batch size is {arr.size(0)}."
                          f"We only retrieve the first batch.")
                arr = arr[0]  # Extract first batch
        else:
            # Strict dimension check when not handling batch dim
            assert arr.ndim == dim, f"Torch tensor has dimension {arr.ndim}, should be {dim}!"
        arr = arr.detach().cpu().numpy()
    # Handle list conversion    
    elif isinstance(arr, list):
        arr = np.asarray(arr)
    
    assert isinstance(arr, np.ndarray)
    
    # Handle array dimensions
    if remove_batch_dim:
        if arr.ndim == dim - 1:
            arr = arr[None, ...]  # Add batch dimension if missing
    else:
        assert arr.ndim == dim
        
    return arr


def subset_pointcloud(pcd: o3d.geometry.PointCloud, inds: np.ndarray):
    """Create new point cloud containing subset of points specified by indices.
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud
        inds (np.ndarray): Array of indices specifying which points to keep

    Returns:
        o3d.geometry.PointCloud: New point cloud containing only specified points
    """
    new_pcd = o3d.geometry.PointCloud()
    if pcd.has_points():
        new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[inds])
    if pcd.has_normals():
        new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[inds])
    if pcd.has_colors():
        new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[inds])
    return new_pcd


def transparent(geom, alpha: float = 0.5):
    """Create transparent version of input geometry by setting alpha value.
    
    Args:
        geom (Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, o3d.geometry.LineSet]): Input geometry
        alpha (float, optional): Transparency value between 0 and 1. Defaults to 0.5.

    Returns:
        AnnotatedGeometry: Geometry with transparency attribute set
    """
    my_geom = AnnotatedGeometry(geom, None)
    my_geom.attributes["alpha"] = alpha
    return my_geom


def text(text: str, pos: Union[list, np.ndarray], is_mesh: bool = False, text_height: float = 0.2):
    """Create 3D text geometry either as mesh or point cloud.
    
    Args:
        text (str): Text content to display
        pos (Union[list, np.ndarray]): 3D position for text placement
        is_mesh (bool, optional): Whether to create text as mesh or point cloud. Defaults to False.
        text_height (float, optional): Height of text in world units. Defaults to 0.2.

    Returns:
        Union[o3d.geometry.TriangleMesh, AnnotatedGeometry of o3d.geometry.PointCloud]: Text geometry as either mesh or annotated point cloud
    """
    if is_mesh:
        # Default height is 16x16 pixels
        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=1).to_legacy()
        text_mesh.scale(1.0 / 16 * text_height, center=[0.0, 0.0, 0.0])
        text_mesh.translate(pos)
        text_mesh.compute_vertex_normals()
        return text_mesh
    else:
        text_geom = AnnotatedGeometry(text_pcd(text, pos), None)
        text_geom.attributes["text"] = text
        text_geom.attributes["text_pos"] = np.asarray(pos)
        return text_geom


def text_pcd(text, pos, direction=None, degree=0.0, font='DejaVuSansMono.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = (int(font_obj.getlength(text)), 24)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def layout_entities(*identities_groups, gaps_dx=None, gaps_dy=None, gaps_dz=None, layout=None, margin=1.0,
                    x_labels=None, y_labels=None, z_labels=None, group_names=None):
    if gaps_dx is None:
        gaps_dx = np.asarray([1.0, 0.0, 0.0]) * margin
    if gaps_dy is None:
        gaps_dy = np.asarray([0.0, 1.0, 0.0]) * margin
    if gaps_dz is None:
        gaps_dz = np.asarray([0.0, 0.0, 1.0]) * margin

    flattened_names = []
    if group_names is not None:
        assert len(group_names) == len(identities_groups)

    if x_labels is None:
        x_labels = []
    if y_labels is None:
        y_labels = []
    if z_labels is None:
        z_labels = []

    if layout is None:
        layout = (len(identities_groups),)

    ind_coords = np.indices(layout).reshape(len(layout), -1).T

    all_identities = []
    for gid, ind_coord in enumerate(ind_coords):
        layout = (ind_coord[0],
                  ind_coord[1] if len(ind_coord) > 1 else 0,
                  ind_coord[2] if len(ind_coord) > 2 else 0)
        for ident_id, ident in enumerate(identities_groups[gid]):
            all_identities.append((ident, Isometry(t=layout[0] * gaps_dx + layout[1] * gaps_dy + layout[2] * gaps_dz)))
            flattened_names.append(None if group_names is None else group_names[gid] + f"-{ident_id}")

    # Draw labels.
    for xi, x_label in enumerate(x_labels):
        all_identities.append((text(x_label, [0.0, 0.0, 0.0]), Isometry(t=xi * gaps_dx - gaps_dy)))
        flattened_names.append(None if group_names is None else f"x-label-{xi}")

    for yi, y_label in enumerate(y_labels):
        all_identities.append((text(y_label, [0.0, 0.0, 0.0]), Isometry(t=yi * gaps_dy - gaps_dx)))
        flattened_names.append(None if group_names is None else f"y-label-{yi}")

    for zi, z_label in enumerate(z_labels):
        all_identities.append((text(z_label, [0.0, 0.0, 0.0]), Isometry(t=zi * gaps_dz)))
        flattened_names.append(None if group_names is None else f"z-label-{zi}")

    return all_identities, flattened_names


def show_3d(*identities_groups, gaps_dx: np.ndarray = None, gaps_dy: np.ndarray = None, 
            gaps_dz: np.ndarray = None, layout: Tuple[int] = None, margin: float = 1.0,
            x_labels: List[str] = None, y_labels: List[str] = None, z_labels: List[str] = None, 
            show: bool = True, use_new_api: bool = False, group_names: List[str] = None, 
            cam_path: Union[str, Path] = None, key_bindings: Dict = None, separate_windows: bool = True, 
            point_size: float = 5.0, scale: float = 1.0, default_camera_kwargs: Dict = None,
            viewport_shading: str = 'LIT', auto_plane: bool = False, up_axis: str = "+Y", 
            polyscope: bool = False):
    """Display 3D geometries in one or multiple windows with customizable layout and camera settings.

    Args:
        *identities_groups: multiple geom objects to be displayed in separate windows, e.g. [geom_a, geom_b], [geom_c, geom_d], ...
        gaps_dx: Spacing between objects in x direction. Defaults to [1.0, 0.0, 0.0] * margin
        gaps_dy: Spacing between objects in y direction. Defaults to [0.0, 1.0, 0.0] * margin  
        gaps_dz: Spacing between objects in z direction. Defaults to [0.0, 0.0, 1.0] * margin
        layout: Tuple specifying grid layout dimensions. Defaults to (len(identities_groups),)
        margin: Margin between objects. Defaults to 1.0
        x_labels: Labels to display along x-axis. Defaults to None
        y_labels: Labels to display along y-axis. Defaults to None 
        z_labels: Labels to display along z-axis. Defaults to None
        show: Whether to display the visualization. Defaults to True
        use_new_api: Whether to use new Open3D visualization API. Defaults to False
        group_names: Names for each group of geometries. Defaults to None
        cam_path: Path to save/load camera parameters. Defaults to None
        key_bindings: Custom keyboard controls. Defaults to None
        separate_windows: Whether to show groups in separate windows. Defaults to True
        point_size: Size of points in point clouds. Defaults to 5.0
        scale: Scale factor for window size. Defaults to 1.0
        default_camera_kwargs: Default camera parameters. Defaults to None
        viewport_shading: Shading mode ('LIT', etc). Defaults to 'LIT'
        auto_plane: Whether to add ground plane automatically. Defaults to False
        up_axis: Up axis direction ("+Y", etc). Defaults to "+Y"
        polyscope: Whether to use polyscope for visualization. Defaults to False

    Returns:
        Union[List[Scene], Scene]: List of render.Scene objects if separate_windows=True and multiple groups,
            otherwise single render.Scene object
    """
    import pycg.render as render

    scenes = []
    scene_titles = []

    if default_camera_kwargs is None:
        default_camera_kwargs = {}

    # Handle separate windows case
    if separate_windows:
        # Split into individual groups
        identities_groups = [[t] for t in identities_groups]
        group_names = [[t] for t in group_names] if group_names is not None else [None for _ in identities_groups]

        # Calculate layout dimensions
        if layout is None or len(layout) == 1:
            layout_rows, layout_cols = 1, len(identities_groups)
        else:
            layout_rows = math.ceil(len(identities_groups) / layout[1]) if layout[0] == -1 else layout[0]
            layout_cols = math.ceil(len(identities_groups) / layout_rows)

        # Set default window size based on monitor dimensions if available
        if len(default_camera_kwargs) == 0:
            if MONITOR_HEIGHT is not None and MONITOR_WIDTH is not None:
                screen_width = MONITOR_WIDTH
                screen_height = MONITOR_HEIGHT - 128
                window_width = min(screen_width // layout_cols, 1024)
                window_height = min(screen_height // layout_rows, 768)
                default_camera_kwargs['w'] = int(window_width / scale)
                default_camera_kwargs['h'] = int(window_height / scale)
            else:
                default_camera_kwargs = {'w': 1024, 'h': 768}

    else:
        # Single window case
        identities_groups = [identities_groups]
        group_names = [group_names]

    # Create scenes for each group
    for ig, gn in zip(identities_groups, group_names):
        flattened_identities, flattened_names = layout_entities(*ig, gaps_dx=gaps_dx, gaps_dy=gaps_dy, gaps_dz=gaps_dz,
                                                                layout=layout if not separate_windows else None,
                                                                margin=margin,
                                                                x_labels=x_labels, y_labels=y_labels, z_labels=z_labels,
                                                                group_names=gn)
        scene = render.Scene(cam_path=cam_path, up_axis=up_axis)
        for (vis_geom, vis_pose), geom_name in zip(flattened_identities, flattened_names):
            scene.add_object(geom=vis_geom, pose=vis_pose, name=geom_name)

        if auto_plane:
            scene.auto_plane(dist_ratio=0.02, scale=1.0)

        if cam_path is None or not Path(cam_path).exists():
            scene.quick_camera(**default_camera_kwargs)

        scene.point_size = point_size
        scene.viewport_shading = viewport_shading
        scenes.append(scene)
        scene_titles.append(gn[0] if gn is not None else None)

    # Handle visualization
    if separate_windows and len(scenes) > 1:
        assert not polyscope, "Polyscope is not supported for multiple windows."
        if show:
            render.vis_manager.reset()
            for scene, scene_title in zip(scenes, scene_titles):
                render.vis_manager.add_scene(scene, title=scene_title,
                                             pose_change_callback=scene.record_camera_pose if cam_path is not None else None)
            render.vis_manager.run(use_new_api=use_new_api, key_bindings=key_bindings, max_cols=layout_cols, scale=scale)
            # Re-load camera, if there are some changes.
            for scene in scenes:
                try:
                    scene.load_camera()
                except Exception:
                    continue
        return scenes
    else:
        scene = scenes[0]
        if show:
            if polyscope:
                scene.preview_polyscope()
            else:
                scene.preview(title="View-Util 3D" if scene_titles[0] is None else scene_titles[0],
                            allow_change_pose=cam_path is not None, add_ruler=False, use_new_api=use_new_api,
                            key_bindings=key_bindings)
        return scene


def pointflow(base_pc: np.ndarray, base_flow: np.ndarray, dest_pc: np.ndarray = None,
              match_color: np.ndarray = None, return_transformed: bool = True):
    """Visualize point cloud flow/deformation with colored points and lines.
    
    This function creates a visualization of point flow/deformation by showing:
    - Source points in red
    - Transformed points (base_pc + flow) in green 
    - Flow lines connecting corresponding points
    - Optional target points in blue
    
    Args:
        base_pc (np.ndarray): Source point cloud of shape (N, 3)
        base_flow (np.ndarray): Flow vectors of shape (N, 3) representing point displacements
        dest_pc (np.ndarray, optional): Target/destination point cloud of shape (M, 3). Defaults to None.
        match_color (np.ndarray, optional): Color indices for flow lines of shape (N,). Defaults to None.
        return_transformed (bool, optional): Whether to include transformed points in output. Defaults to True.
        
    Returns:
        list: List of Open3D geometries:
            - Source point cloud (red)
            - Transformed point cloud (green) if return_transformed=True  
            - Flow lines connecting source and transformed points
            - Target point cloud (blue) if dest_pc is provided
    """
    # Convert inputs to numpy arrays with correct dimensions
    base_pc = ensure_from_torch(base_pc, 2)
    base_flow = ensure_from_torch(base_flow, 2)
    
    # Validate input shapes
    assert base_pc.shape[1] == 3 and len(base_pc.shape) == 2, f"Point cloud is of size {base_pc.shape}!"
    assert base_flow.shape[1] == 3 and len(base_flow.shape) == 2, f"Point flow is of size {base_flow.shape}!"
    assert base_flow.shape[0] == base_pc.shape[0], f"Cloud&flow mismatch: {base_pc.shape}, {base_flow.shape}"
    
    print("Start from red, go to green, target is blue.")
    
    # Create source point cloud (red)
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(base_pc)
    base_pcd.paint_uniform_color((1.0, 0., 0.))
    
    # Create transformed point cloud (green)
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(base_pc + base_flow)
    final_pcd.paint_uniform_color((0., 1.0, 0.))
    
    # Create flow lines connecting corresponding points
    corres_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([base_pc, base_pc + base_flow])),
        lines=o3d.utility.Vector2iVector(np.arange(2 * base_flow.shape[0]).reshape((2, -1)).T))
    
    # Build list of geometries to return
    if return_transformed:
        all_drawables = [base_pcd, final_pcd, corres_lineset]
    else:
        all_drawables = [base_pcd, corres_lineset]
        
    # Add target point cloud if provided (blue)
    if dest_pc is not None:
        dest_pc = ensure_from_torch(dest_pc, 2)
        assert dest_pc.shape[1] == 3 and len(dest_pc.shape) == 2, f"Point cloud is of size {dest_pc.shape}!"
        dest_pcd = o3d.geometry.PointCloud()
        dest_pcd.points = o3d.utility.Vector3dVector(dest_pc)
        dest_pcd.paint_uniform_color((0., 0., 1.0))
        all_drawables.append(dest_pcd)
        
    # Color flow lines if colors provided
    if match_color is not None:
        color_map = np.asarray(matplotlib.cm.get_cmap('tab10').colors)
        match_color = match_color % (color_map.shape[0])
        color_map = np.vstack([color_map, np.zeros((1, 3))])
        match_color[match_color < 0] = color_map.shape[0] - 1
        corres_lineset.colors = o3d.utility.Vector3dVector(color_map[match_color])

    return all_drawables


def correspondence(source_pc: Union[np.ndarray, o3d.geometry.PointCloud], 
                  target_pc: Union[np.ndarray, o3d.geometry.PointCloud], 
                  matches: Optional[Union[np.ndarray, "torch.Tensor"]] = None, 
                  match_color: Optional[np.ndarray] = None,
                  subsampled_ratio: float = 1.0, 
                  gap: Optional[np.ndarray] = None, 
                  match_color_normalize: bool = False, 
                  **color_kwargs) -> o3d.geometry.LineSet:
    """Creates a visualization of correspondences between two point clouds.

    This function visualizes point-to-point correspondences between source and target point clouds
    by creating a line set connecting corresponding points. The source points are colored red and 
    target points are colored green. The correspondence lines can be colored based on match_color.

    Args:
        source_pc: Source point cloud as numpy array (N,3) or Open3D PointCloud
        target_pc: Target point cloud as numpy array (N,3) or Open3D PointCloud  
        matches: Optional array of shape (M,2) containing indices of corresponding points.
                If None, assumes 1-to-1 correspondence between points.
        match_color: Optional array of shape (M,) for coloring correspondence lines.
                    Can be integers for discrete colors or floats for continuous colormap.
        subsampled_ratio: Float between 0 and 1 for randomly sampling a subset of matches to display.
        gap: Optional (1,3) array specifying translation to apply to target points.
        match_color_normalize: If True, normalizes float match_color values to [0,1] range.
        **color_kwargs: Additional arguments passed to lineset() for line coloring.

    Returns:
        o3d.geometry.LineSet: Line set visualization of point correspondences
    """
    # Convert point clouds to numpy arrays if needed
    if isinstance(source_pc, o3d.geometry.PointCloud):
        source_pc = np.asarray(source_pc.points)
    if isinstance(target_pc, o3d.geometry.PointCloud):
        target_pc = np.asarray(target_pc.points)

    # Convert inputs to numpy arrays with correct dimensions
    source_pc = ensure_from_torch(source_pc, dim=2)
    target_pc = ensure_from_torch(target_pc, dim=2)

    # Validate inputs
    assert 0.0 < subsampled_ratio <= 1.0
    assert source_pc.shape[1] == 3 and len(source_pc.shape) == 2, f"Source PC is of size {source_pc.shape}!"
    assert target_pc.shape[1] == 3 and len(target_pc.shape) == 2, f"Target PC is of size {target_pc.shape}!"
    if gap is None:
        gap = np.asarray([[0.0, 0.0, 0.0]])

    # Create matches if not provided
    if matches is None:
        assert source_pc.shape[0] == target_pc.shape[0]
        matches = np.expand_dims(np.arange(source_pc.shape[0]), 1).repeat(2, axis=1).astype(int)
    else:
        matches = ensure_from_torch(matches, 2)
        matches = np.asarray(matches).astype(int).copy()
        assert matches.shape[1] == 2, f"Matches is not valid! {matches.shape}"

    # Randomly subsample matches if ratio < 1
    match_sub_inds = np.random.choice(np.arange(matches.shape[0]),
                                     int(subsampled_ratio * matches.shape[0]),
                                     replace=False)
    matches = matches[match_sub_inds, :]
    matches[:, 1] += source_pc.shape[0]

    # Create source point cloud (red)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pc.astype(float).copy())
    source_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    
    # Create target point cloud (green)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pc.astype(float).copy() + gap)
    target_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    # Create line set connecting corresponding points
    corres_lineset = lineset(
        np.vstack([source_pc.astype(float), target_pc.astype(float).copy() + gap]), matches,
        **color_kwargs
    )

    # Color the correspondence lines if match_color provided
    if match_color is not None:
        match_color = match_color[match_sub_inds]
        if match_color.dtype == float or match_color.dtype == np.float32:
            # Handle continuous color values using jet colormap
            if match_color_normalize:
                min_match_color, max_match_color = match_color.min(), match_color.max()
                match_color = (match_color - min_match_color) / (max_match_color - min_match_color + 1e-6)
                print(f"Match Color, minimum = {min_match_color}, maximum = {max_match_color}.")
            corres_lineset.colors = o3d.utility.Vector3dVector(matplotlib.cm.jet(match_color)[:, :3])
        else:
            # Handle discrete color values using tab10 colormap
            color_map = np.asarray(matplotlib.cm.get_cmap('tab10').colors)
            match_color = match_color % (color_map.shape[0])
            color_map = np.vstack([color_map, np.zeros((1, 3))])
            match_color[match_color < 0] = color_map.shape[0] - 1
            corres_lineset.colors = o3d.utility.Vector3dVector(color_map[match_color])

    return corres_lineset


def multiview_correspondence(pcs: List[np.ndarray], matches: Dict[Tuple, np.ndarray], subsampled_ratio: float = 0.2,
                             gt_flow: Dict[Tuple, np.ndarray] = None, gap: float = 1.0, gt_thres=0.05, is_sphere=False,
                             segm=None, vp_good_dict: dict = None, vp_res_dict: dict = None):
    """
    Draw multi-view correspondence with gt annotation
    :param pcs: (M, N, 3), M is view count, N is point count
    :param matches: {view_i,view_j}->(N,2) matches, will only use (view_i < view_j part)
    :param subsampled_ratio: ratio for subsampling matches.
    :param gt_flow: used only to draw correctness of a correspondence.
    :param gap: radius of the Star of David.
    :param gt_thres: the threshold to judge that a matching is correct.
    """
    drawables = []
    pcds = []
    n_view = len(pcs)
    # First, draw point clouds.
    color_map = np.asarray(matplotlib.cm.get_cmap('tab10').colors)
    for view_i in range(n_view):
        assert pcs[view_i].shape[1] == 3 and len(pcs[view_i].shape) == 2
        pcd = pointcloud(pcs[view_i].astype(float).copy(), is_sphere=is_sphere, sphere_radius=0.06)
        # pcd.paint_uniform_color([0.6, 0.6, 0.6])
        pcd.paint_uniform_color(color_map[view_i])
        pcd.translate((gap * np.cos(view_i / n_view * 2 * np.pi), 0.0,
                       gap * np.sin(view_i / n_view * 2 * np.pi)))
        drawables.append(pcd)

        pcd_raw = pointcloud(pcs[view_i].astype(float).copy())
        pcd_raw.translate((gap * np.cos(view_i / n_view * 2 * np.pi), 0.0,
                           gap * np.sin(view_i / n_view * 2 * np.pi)))
        pcds.append(pcd_raw)

    # Then draw matches.
    num_matches = 0
    num_correct = 0
    for view_i in range(n_view):
        for view_j in range(view_i + 1, n_view):
            cur_match = matches[(view_i, view_j)].astype(int).copy()
            assert cur_match.shape[1] == 2, f"Matches is not valid! {cur_match.shape}"
            cur_match = cur_match[np.random.choice(np.arange(cur_match.shape[0]),
                                                   int(subsampled_ratio * cur_match.shape[0]),
                                                   replace=False), :]
            if gt_flow is not None:
                pd_pos = pcs[view_j][cur_match[:, 1]]
                gt_pos = pcs[view_i][cur_match[:, 0]] + gt_flow[(view_i, view_j)][cur_match[:, 0]]
                pn = (np.linalg.norm(pd_pos - gt_pos, axis=1) < gt_thres).astype(int)
                num_matches += pn.shape[0]
                num_correct += np.sum(pn)
                colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])[pn]
                if vp_good_dict is not None:
                    if np.sum(pn) > pn.shape[0] * 0.75:
                        vp_good_dict["vp"].append((view_i, view_j))
                    vp_good_dict["vps"][(view_i, view_j)] = pn.copy()
            else:
                colors = np.zeros((cur_match.shape[0], 3))

            cur_match[:, 1] += len(pcds[view_i].points)
            corres_lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(
                    np.vstack([np.asarray(pcds[view_i].points),
                               np.asarray(pcds[view_j].points)])),
                lines=o3d.utility.Vector2iVector(cur_match)
            )
            corres_lineset.colors = o3d.utility.Vector3dVector(colors)
            drawables.append(corres_lineset)
    if num_matches != 0:
        print("Percent of Correctness =", num_correct / num_matches)
        if vp_res_dict is not None:
            vp_res_dict["acc"] = num_correct / num_matches
    return drawables


def multiview_segmentation(pcs: List[np.ndarray], segm: Dict[int, np.ndarray], gap: float = 1.0):
    """
    Draw multi-view segmentation.
    :param pcs: (K, N, 3), K is view count, N is point count.
    :param segm: {view_i} -> (n, 2) index, segmentation.
    :param gap: radius of the Star of David.
    """
    drawables = []
    n_view = len(pcs)
    for view_i in range(n_view):
        cur_pc = pcs[view_i].astype(float).copy()
        n_point = cur_pc.shape[0]
        cur_segm = np.ones((n_point,)) * -1
        cur_segm[segm[view_i][:, 0]] = segm[view_i][:, 1]
        pcd = pointcloud(cur_pc, cur_segm)
        pcd.translate((gap * np.cos(view_i / n_view * 2 * np.pi),
                       0.0,
                       gap * np.sin(view_i / n_view * 2 * np.pi)))
        drawables.append(pcd)
    return drawables


def frame(transform: Isometry = Isometry(), size: float = 1.0):
    """Create a coordinate frame visualization object at the specified transform.
    
    Creates a coordinate frame mesh with RGB colored axes (Red=X, Green=Y, Blue=Z)
    positioned and oriented according to the input transform.

    Args:
        transform (Isometry): The pose transform to apply to the frame. Defaults to identity transform.
        size (float): The size/scale of the coordinate frame axes. Defaults to 1.0.

    Returns:
        o3d.geometry.TriangleMesh: Triangle mesh representing the coordinate frame
    """
    # Create coordinate frame mesh with specified size
    frame_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    # Apply transform to position and orient the frame
    frame_obj.transform(transform.matrix)
    return frame_obj

def camera(transform: Isometry = Isometry(), wh_ratio: float = 4.0 / 3.0, scale: float = 1.0, fovx: float = 90.0,
           color_id: int = -1, image: Optional[np.ndarray] = None):
    """Create a camera visualization object with optional image texture.
    
    Creates a wireframe camera frustum visualization with optional textured image plane.
    The camera is positioned and oriented according to the input transform.

    Args:
        transform (Isometry): The pose transform to apply to the camera. Defaults to identity transform.
        wh_ratio (float): Width/height ratio of the camera frustum. Defaults to 4/3.
        scale (float): Scale factor for camera size. Defaults to 1.0.
        fovx (float): Horizontal field of view in degrees. Defaults to 90.0.
        color_id (int): Color index for wireframe (-1 for black, otherwise uses tab10 colormap). Defaults to -1.
        image (Optional[np.ndarray]): Optional image to texture onto frustum far plane. Defaults to None.

    Returns:
        Union[o3d.geometry.LineSet, List[Union[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]]]: 
            If no image provided: LineSet representing camera wireframe
            If image provided: List containing wireframe LineSet and textured TriangleMesh
    """
    # Calculate frustum dimensions based on FOV and scale
    pw = np.tan(np.deg2rad(fovx / 2.)) * scale  # Half width of frustum far plane
    ph = pw / wh_ratio  # Half height of frustum far plane
    
    # Define frustum vertices and line connections
    all_points = np.asarray([
        [0.0, 0.0, 0.0],  # Camera center
        [pw, ph, scale],   # Top right
        [pw, -ph, scale],  # Bottom right 
        [-pw, ph, scale],  # Top left
        [-pw, -ph, scale], # Bottom left
    ])
    line_indices = np.asarray([
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
        [1, 2], [1, 3], [3, 4], [2, 4]   # Lines connecting corners
    ])
    
    # Create wireframe geometry
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(line_indices))

    # Set wireframe color
    if color_id == -1:
        my_color = np.zeros((3,))  # Black
    else:
        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[color_id, :3]
    geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), line_indices.shape[0], 0))
    geom.transform(transform.matrix)

    # Add optional textured image plane
    if image is not None:
        img_mesh = o3d.geometry.TriangleMesh()
        # Define vertices for image plane
        img_mesh.vertices = o3d.utility.Vector3dVector(np.array([
            [pw, ph, scale],    # Top right
            [pw, -ph, scale],   # Bottom right
            [-pw, ph, scale],   # Top left
            [-pw, -ph, scale],  # Bottom left
        ]))
        # Define triangles
        img_mesh.triangles = o3d.utility.Vector3iVector(np.asarray([
            [0, 1, 2], [2, 1, 3],  # Two triangles forming rectangle
        ]))
        # Define UV coordinates for texture mapping
        img_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array([
            [1.0, 1.0], [1.0, 0.0], [0.0, 1.0],
            [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]
        ]))
        img_mesh.textures = [o3d.geometry.Image(image)]
        img_mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros((len(img_mesh.triangles, )), dtype=np.int32))
        img_mesh.transform(transform.matrix)
        geom = [geom, img_mesh]

    return geom

def arrow(base: np.ndarray, target: np.ndarray,
          scale: float = 1.0, resolution: int = 5, 
          fix_cone_height: float = None, cone_enlarge_ratio: float = 4.0,
          color_id: int = 0, **cmap_kwargs):
    """Create 3D arrow(s) from base point(s) to target point(s).

    Args:
        base: Base point(s) of arrow(s), shape (N,3) or (3,)
        target: Target point(s) of arrow(s), shape (N,3) or (3,)
        scale: Overall scale factor for arrow size
        resolution: Resolution of arrow cylinder/cone, higher means smoother
        fix_cone_height: If set, fixes arrow head cone height to this value
        cone_enlarge_ratio: Ratio of cone radius to cylinder radius
        color_id: Color index for arrow(s), used with pycg colormap
        **cmap_kwargs: Additional arguments passed to colored_mesh()

    Returns:
        o3d.geometry.TriangleMesh: Colored arrow mesh(es)
    """
    # Convert torch tensors to numpy if needed
    base = ensure_from_torch(base, 2)
    target = ensure_from_torch(target, 2)

    n_verts = 0  # Track total vertices for face indexing
    arrow_verts = []
    arrow_faces = []
    
    # Create arrow for each base-target pair
    for b, t in zip(base, target):
        # Get transformation from origin to arrow direction
        arrow_iso = Isometry.look_at(b, t)
        arrow_len = np.linalg.norm(t - b)
        base_radius = scale * 0.05  # Base cylinder radius

        # Calculate arrow head (cone) height
        if fix_cone_height is not None:
            cone_height = min(fix_cone_height, arrow_len * 0.9)
        else:
            cone_height = 0.3 * arrow_len

        # Create arrow geometry
        arrow_obj = o3d.geometry.TriangleMesh.create_arrow(resolution=resolution,
                                                           cylinder_height=arrow_len - cone_height,
                                                           cone_height=cone_height,
                                                           cone_radius=cone_enlarge_ratio * base_radius,
                                                           cylinder_radius=base_radius,
                                                           cylinder_split=1)
        # Transform arrow to correct position/orientation
        arrow_obj.transform(arrow_iso.matrix)

        # Collect vertices and faces, updating face indices
        arrow_verts.append(np.asarray(arrow_obj.vertices))
        arrow_faces.append(np.asarray(arrow_obj.triangles) + n_verts)
        n_verts += len(arrow_obj.vertices)

    # Combine all arrows into single colored mesh
    return colored_mesh(np.concatenate(arrow_verts), np.concatenate(arrow_faces), ucid=color_id, **cmap_kwargs)


def sphere_from_pc(pcd: o3d.geometry.PointCloud, radius: float = 0.02, resolution: int = 5) -> o3d.geometry.TriangleMesh:
    """Convert a point cloud to a mesh of spheres by placing a sphere at each point location.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud to convert
        radius (float, optional): Radius of each sphere. Defaults to 0.02.
        resolution (int, optional): Resolution of each sphere mesh. Higher values create smoother spheres. Defaults to 5.

    Returns:
        o3d.geometry.TriangleMesh: Combined mesh of all spheres
    """
    # Initialize empty mesh to store combined spheres
    final_mesh = o3d.geometry.TriangleMesh()

    # Get point positions and colors if available
    pc_pos = np.asarray(pcd.points)
    if pcd.has_colors():
        pc_colors = np.asarray(pcd.colors)
    else:
        pc_colors = None

    # Create and position a sphere for each point
    for pc_id in range(len(pc_pos)):
        # Create sphere with specified radius and resolution
        cur_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        # Move sphere to point position
        cur_sphere.translate(pc_pos[pc_id])
        cur_sphere.compute_vertex_normals()
        # Color sphere if point cloud has colors
        if pc_colors is not None:
            cur_sphere.paint_uniform_color(pc_colors[pc_id])
        # Add to combined mesh
        final_mesh += cur_sphere

    return final_mesh

def surfel_from_pc(pcd: o3d.geometry.PointCloud, radius: float = 0.02, resolution: int = 5) -> o3d.geometry.TriangleMesh:
    """Convert a point cloud to a mesh of surfels (oriented disks) by placing a circular disk at each point location.
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud to convert. Must have normals.
        radius (float, optional): Radius of each surfel disk. Defaults to 0.02.
        resolution (int, optional): Number of vertices to use for approximating each circular disk. 
            Higher values create smoother circles. Defaults to 5.

    Returns:
        o3d.geometry.TriangleMesh: Combined mesh of all surfels
    """
    assert pcd.has_normals(), "Point Cloud must have normals!"

    # Convert points and normals to numpy arrays
    pc, normal = np.array(pcd.points), np.array(pcd.normals)
    
    # Compute local coordinate frame for each point based on its normal
    # normal_x and normal_y form an orthogonal basis in the tangent plane
    normal_x = np.stack([normal[:, 1] - normal[:, 2], -normal[:, 0], normal[:, 0]], axis=-1)
    normal_x /= np.linalg.norm(normal_x, axis=-1, keepdims=True)
    normal_y = np.cross(normal, normal_x)
    
    # Scale the basis vectors by radius
    normal_x *= radius
    normal_y *= radius

    # Generate vertices and faces for each surfel disk
    vertices = []
    faces = []
    for r in range(resolution):
        # Create vertices in a circular pattern
        angle = r / resolution * 2 * np.pi
        v = pc + normal_x * np.cos(angle) + normal_y * np.sin(angle)
        vertices.append(v)
        # Create triangular faces, connecting to center point
        if r > 1:
            faces.append([0, r - 1, r])
            
    # Reshape vertices and faces into final mesh structure
    vertices = np.stack(vertices, axis=1).reshape((-1, 3))
    faces = np.expand_dims(np.concatenate(faces), axis=0).repeat(pc.shape[0], 0)
    faces += np.expand_dims(np.arange(pc.shape[0]), axis=-1) * resolution
    faces = faces.reshape((-1, 3))

    # Create and populate the mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Transfer colors if the point cloud has them
    if pcd.has_colors():
        rgb = np.array(pcd.colors)
        color = np.tile(rgb, (1, resolution)).reshape((-1, 3))
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)

    return mesh


def oriented_pointcloud(pcd: o3d.geometry.PointCloud, knn: int = None, radius: float = None,
                        double_layer: bool = False, double_layer_delta: float = 0.01,
                        orient_k: int = 8, orient_th: float = 0.3, orient_outward: bool = True):

    # Need estimate normal.
    if knn is None and radius is None:
        assert pcd.has_normals(), "Point cloud does not have normal!"
    else:
        if knn is None:
            search_param = o3d.geometry.KDTreeSearchParamRadius(radius=radius)
        elif radius is None:
            search_param = o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        else:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn)

        pcd.estimate_normals(search_param=search_param)

    if double_layer:
        inverse_pcd = copy.deepcopy(pcd)
        inverse_normal = -np.array(inverse_pcd.normals)
        inverse_pcd.normals = o3d.utility.Vector3dVector(inverse_normal)
        inverse_pcd.points = o3d.utility.Vector3dVector(np.array(inverse_pcd.points) + double_layer_delta * inverse_normal)
        pcd += inverse_pcd
    else:
        pcd.orient_normals_flood_fill(orient_k, weight_threshold=orient_th)

        # Compute CH to deal with non-convex shapes.
        ch_mesh, ch_idx = pcd.compute_convex_hull()
        ch_center = (ch_mesh.get_max_bound() + ch_mesh.get_min_bound()) / 2.
        ch_points = np.array(pcd.points)[ch_idx]
        ch_dir = ch_points - ch_center[None, :]
        cur_normals = np.array(pcd.normals)[ch_idx]
        flip_score = np.sum(np.sum(cur_normals * ch_dir, axis=-1) < 0) / len(ch_idx)

        if flip_score > 0.5:
            pcd.normals = o3d.utility.Vector3dVector(-np.array(pcd.normals))

    return pcd


def sphere(center: np.ndarray = np.zeros((3, )), radius: float = 1.0, resolution: int = 10, ucid: int = None):
    """Create a sphere geometry using point cloud representation.

    Args:
        center (np.ndarray, optional): 3D coordinates of sphere center. Defaults to [0,0,0].
        radius (float, optional): Radius of the sphere. Defaults to 1.0.
        resolution (int, optional): Number of points used to represent the sphere surface. 
            Higher resolution means smoother sphere. Defaults to 10.
        ucid (int, optional): Unique color ID for the sphere. If provided, the sphere will be 
            colored according to this ID using the default colormap. Defaults to None.

    Returns:
        o3d.geometry.PointCloud: A point cloud geometry representing the sphere.
    """
    # Add singleton dimension to center coordinates to make it (1,3) shape
    # and pass to pointcloud() function with sphere parameters
    return pointcloud(
        center[None, :], is_sphere=True, ucid=ucid, sphere_radius=radius, sphere_resolution=resolution
    )


def pointcloud(pc: Union[np.ndarray, o3d.geometry.PointCloud], cid: np.ndarray = None, color: np.ndarray = None, 
               ucid: int = None, cmap: str = 'tab10', normal: np.ndarray = None, estimate_normals: bool = False, 
               estimate_normals_radius: float = None, estimate_normals_nn: int = 16, double_layer: bool = False, 
               double_layer_delta: float = 0.01, cfloat: np.ndarray = None, cfloat_cmap: str = 'jet', 
               cfloat_normalize: bool = False, cfloat_annotated: bool = True, is_sphere: bool = False, 
               sphere_radius: float = 0.02, sphere_resolution: int = 5, is_surfel: bool = False, 
               surfel_radius: float = 0.02, surfel_resolution: int = 5):
    """Create a point cloud visualization with various customization options.

    Args:
        pc (Union[np.ndarray, o3d.geometry.PointCloud]): Input point cloud data, either as numpy array of shape (N,3) or Open3D PointCloud
        cid (np.ndarray, optional): Color IDs for each point. Shape (N,). Defaults to None.
        color (np.ndarray, optional): RGB colors for points. Shape (N,3). Defaults to None.
        ucid (int, optional): Uniform color ID for all points. Defaults to None.
        cmap (str, optional): Colormap name for color mapping. Defaults to 'tab10'.
        normal (np.ndarray, optional): Normal vectors for points. Shape (N,3). Defaults to None.
        estimate_normals (bool, optional): Whether to estimate normals if not provided. Defaults to False.
        estimate_normals_radius (float, optional): Search radius for normal estimation. Defaults to None.
        estimate_normals_nn (int, optional): Number of neighbors for normal estimation. Defaults to 16.
        double_layer (bool, optional): Whether to create double-sided point cloud. Defaults to False.
        double_layer_delta (float, optional): Offset distance for double layer. Defaults to 0.01.
        cfloat (np.ndarray, optional): Continuous float values for coloring. Shape (N,). Defaults to None.
        cfloat_cmap (str, optional): Colormap for continuous float values. Defaults to 'jet'.
        cfloat_normalize (bool, optional): Whether to normalize cfloat values. Defaults to False.
        cfloat_annotated (bool, optional): Whether to store cfloat as annotation. Defaults to True.
        is_sphere (bool, optional): Whether to render points as spheres (o3d.geometry.TriangleMesh). Defaults to False.
        sphere_radius (float, optional): Radius of spheres if is_sphere=True. Defaults to 0.02.
        sphere_resolution (int, optional): Resolution of spheres if is_sphere=True. Defaults to 5.
        is_surfel (bool, optional): Whether to render points as surfels (o3d.geometry.TriangleMesh). Defaults to False.
        surfel_radius (float, optional): Radius of surfels if is_surfel=True. Defaults to 0.02.
        surfel_resolution (int, optional): Resolution of surfels if is_surfel=True. Defaults to 5.

    Returns:
        Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, AnnotatedGeometry]: Point cloud geometry, optionally with annotations
    """
    # Handle input if it's already an Open3D point cloud
    if isinstance(pc, o3d.geometry.PointCloud):
        if pc.has_normals() and normal is None:
            normal = np.asarray(pc.normals)
        if pc.has_colors() and color is None:
            color = np.asarray(pc.colors)
        pc = np.asarray(pc.points)
    pc = ensure_from_torch(pc)

    cloud_annotation = None
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)

    # Handle color assignment based on color IDs
    if cid is not None:
        cid = ensure_from_torch(cid, dim=1)
        assert cid.shape[0] == pc.shape[0], f"Point and color id must have same size {pc.shape[0]}, {cid.shape[0]}"
        assert cid.ndim == 1, f"color id must be of size (N,) currently ndim = {cid.ndim}"
        cid = cid.astype(int)
        color = map_quantized_color(cid, cmap=cmap)

    # Handle uniform color ID
    if ucid is not None:
        u_color = map_quantized_color(ucid, cmap=cmap)
        color = np.repeat(u_color[None, :], pc.shape[0], 0)

    # Handle continuous float values for coloring
    if cfloat is not None:
        cfloat = ensure_from_torch(cfloat, 1)
        if color is None:
            assert cfloat.shape[0] == pc.shape[0], f"Not match cfloat={cfloat.shape[0]}, pc={pc.shape[0]}"
            if cfloat_normalize:
                cfloat_min, cfloat_max = np.nanmin(cfloat), np.nanmax(cfloat)
                if cfloat_annotated:
                    cloud_annotation = [cfloat, cfloat_cmap]
                print(f"cfloat normalize: min = {cfloat_min}, max = {cfloat_max}")
                cfloat = (cfloat - cfloat_min) / (cfloat_max - cfloat_min + 1.0e-6)
            color = matplotlib.cm.get_cmap(cfloat_cmap)(cfloat)[:, :3]
        else:
            color = color * np.expand_dims(cfloat, 1)

    # Assign colors to point cloud
    if color is not None:
        color = ensure_from_torch(color)
        assert color.shape[0] == pc.shape[0], f"Point and color must have same size {color.shape[0]}, {pc.shape[0]}"
        if issubclass(color.dtype.type, np.integer):
            color = color.astype(float) / 255.
        point_cloud.colors = o3d.utility.Vector3dVector(color[:, :3].astype(float))

    # Handle normal vectors
    if normal is not None:
        normal = ensure_from_torch(normal)
        point_cloud.normals = o3d.utility.Vector3dVector(normal)

    # Estimate normals if requested
    if estimate_normals:
        assert normal is None, "Input already has normal!"
        point_cloud = oriented_pointcloud(point_cloud, knn=estimate_normals_nn, radius=estimate_normals_radius,
                                          double_layer=double_layer, double_layer_delta=double_layer_delta)

    # Convert points to spheres if requested
    if is_sphere:
        point_cloud = sphere_from_pc(point_cloud, sphere_radius, sphere_resolution)

    # Convert points to surfels if requested
    if is_surfel:
        point_cloud = surfel_from_pc(point_cloud, surfel_radius, surfel_resolution)

    # Add annotations if present
    if isinstance(point_cloud, o3d.geometry.PointCloud) and cloud_annotation is not None:
        point_cloud = AnnotatedGeometry(point_cloud, cloud_annotation)

    return point_cloud


def thin_box(plane_n: np.ndarray, plane_c: np.ndarray, is_mesh: bool = True):
    """Create a thin box (plane-like) geometry oriented according to a normal vector.

    Args:
        plane_n: A (3,) numpy array specifying the normal direction of the box
        plane_c: A (3,) numpy array specifying the center position of the box
        is_mesh: If True, returns a TriangleMesh, otherwise returns an OrientedBoundingBox

    Returns:
        o3d.geometry.TriangleMesh or o3d.geometry.OrientedBoundingBox: The created thin box geometry
    """
    # Set box dimensions - thin in z direction
    ex_x = ex_y = 2.0  # width and height
    ex_z = 0.002      # very small thickness

    # Calculate rotation to align z-axis with plane normal
    rot = Quaternion(axis=np.cross([0.0, 0.0, 1.0], plane_n), radians=np.arccos(plane_n[2])).rotation_matrix

    if is_mesh:
        # Create triangle mesh box
        obb = o3d.geometry.TriangleMesh.create_box(width=ex_x, height=ex_y, depth=ex_z)
        # Center the box at origin before rotation
        obb.translate((-ex_x / 2.0, -ex_y / 2.0, -ex_z / 2.0))
        obb.rotate(rot)
        obb.translate(plane_c)
    else:
        # Create oriented bounding box
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = plane_c
        obb.extent = (5.0, 5.0, 0.1)  # Different dimensions for bounding box
        obb.rotate(rot)
    return obb


def plane(center: np.ndarray, normal: np.ndarray, scale: float):
    """Create a planar mesh geometry oriented according to a normal vector.

    Args:
        center: A (3,) numpy array specifying the center position of the plane
        normal: A (3,) numpy array specifying the normal direction of the plane
        scale: A float value determining the size of the plane (half-width/height)

    Returns:
        o3d.geometry.TriangleMesh: A triangular mesh representing the plane, colored white
    """
    # Calculate rotation axis to align z-axis with plane normal
    raxis = np.cross([0.0, 0.0, 1.0], normal)
    # Handle case when normal is parallel to z-axis
    if abs(np.linalg.norm(raxis)) < 1e-6:
        raxis = np.asarray([1.0, 0.0, 0.0])
    # Compute rotation matrix
    rot = Quaternion(axis=raxis, radians=np.arccos(normal[2])).rotation_matrix
    
    # Create mesh and set vertices (square in xy-plane)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array([
        [-scale, -scale, 0.0],
        [-scale, scale, 0.0],
        [scale, scale, 0.0],
        [scale, -scale, 0.0],
    ]))
    # Define triangles (two triangles forming a square)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([
        [0, 2, 1], [0, 3, 2]
    ]))
    # Apply rotation and translation
    mesh.rotate(rot)
    mesh.translate(np.asarray(center))
    # Set color and compute normals
    mesh.paint_uniform_color([1.0, 1.0, 1.0])
    mesh.compute_vertex_normals()
    return mesh


def colored_mesh(*args, **kwargs) -> o3d.geometry.TriangleMesh:
    """Create a colored mesh by ensuring color parameters are set.
    
    This is a wrapper around the mesh() function that guarantees the resulting mesh
    will have color by setting a default unique color ID (ucid) if no color 
    parameters are provided.

    Args:
        *args: Variable length argument list to pass to mesh()
        **kwargs: Arbitrary keyword arguments to pass to mesh()
                 Supported color parameters:
                 - cid: Array of color indices
                 - ucid: Single unique color ID
                 If neither is provided, ucid=0 will be used

    Returns:
        o3d.geometry.TriangleMesh: A colored triangle mesh
    """
    # If no color parameters provided, set default unique color ID
    if 'cid' not in kwargs and 'ucid' not in kwargs:
        kwargs['ucid'] = 0
    # Forward all arguments to mesh() function
    return mesh(*args, **kwargs)


def textured_mesh(mesh: Union[o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh],
                  texture: Optional[np.ndarray] = None) -> Union[o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh]:
    """Apply a texture to a triangle mesh.

    This function takes a triangle mesh and applies a texture to it. If no texture is provided,
    it uses a default UV texture from the assets folder.

    Args:
        mesh: The input triangle mesh to be textured. Can be either legacy or tensor-based Open3D mesh.
        texture: Optional texture image as a numpy array. If None, uses default UV texture. 
               Expected shape is (H, W, C) where C is number of channels (3 for RGB, 4 for RGBA).
               Data type is expected to be uint8.

    Returns:
        The input mesh with texture applied. Returns same type as input mesh (legacy or tensor-based).
    """
    # Load default texture if none provided
    if texture is None:
        from pycg import image, get_assets_path
        texture = image.read(get_assets_path() / "uv.png")

    # Handle legacy Open3D mesh
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh.textures = [o3d.geometry.Image(texture)]
    # Handle tensor-based Open3D mesh
    else:
        mesh.material.material_name = 'defaultLit'
        mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(texture))
    return mesh

def mesh(mesh_or_vertices: Union[np.ndarray, "torch.Tensor", o3d.geometry.TriangleMesh],
         triangles: Union[np.ndarray, "torch.Tensor"] = None,
         color: np.ndarray = None,
         cid: Union[np.ndarray, "torch.Tensor"] = None,
         ucid: int = None, cmap: str = 'tab10',
         cfloat: np.ndarray = None, cfloat_cmap: str = 'jet', cfloat_normalize: bool = False,
         triangle_uvs: Union[np.ndarray, "torch.Tensor"] = None,
         triangle_uv_inds: Union[np.ndarray, "torch.Tensor"] = None,
         compute_vertex_normals: bool = True,
         use_new_api: bool = False):
    """Create a triangle mesh with optional coloring and texture mapping.

    This function creates an Open3D triangle mesh from input vertices and triangles, with support for
    various coloring options and texture mapping.

    Args:
        mesh_or_vertices: Either an existing Open3D mesh or vertex coordinates of shape (N, 3)
        triangles: Triangle indices of shape (M, 3). Not needed if mesh_or_vertices is an Open3D mesh
        color: Per-vertex RGB(A) colors of shape (N, 3/4). Values in [0,1] or uint8
        cid: Per-vertex color indices for categorical coloring using colormap
        ucid: Single color index for uniform mesh coloring using colormap
        cmap: Colormap name for categorical colors. Default: 'tab10'
        cfloat: Per-vertex float values for continuous coloring
        cfloat_cmap: Colormap name for continuous colors. Default: 'jet'
        cfloat_normalize: Whether to normalize cfloat values to [0,1]. Default: False
        triangle_uvs: UV coordinates for texture mapping, shape (M*3, 2) or (M, 3, 2)
        triangle_uv_inds: UV indices if using indexed UVs, shape (M, 3)
        compute_vertex_normals: Whether to compute vertex normals. Default: True
        use_new_api: Whether to return tensor-based mesh (new API). Default: False

    Returns:
        o3d.geometry.TriangleMesh or o3d.t.geometry.TriangleMesh: Created triangle mesh
    """
    # Handle input mesh or vertices
    if isinstance(mesh_or_vertices, o3d.geometry.TriangleMesh):
        assert triangles is None
        vertices = np.asarray(mesh_or_vertices.vertices).copy()
        triangles = np.asarray(mesh_or_vertices.triangles).copy()
    elif triangles is None:
        # Create triangle soup from vertices (each 3 vertices form a triangle)
        vertices = ensure_from_torch(mesh_or_vertices, 3).reshape(-1, 3)
        triangles = np.arange(vertices.shape[0]).reshape(-1, 3)
    else:
        # Use provided vertices and triangles
        vertices = ensure_from_torch(mesh_or_vertices, 2)
        triangles = ensure_from_torch(triangles, 2)

    # Handle per-vertex colors
    if color is not None:
        color = ensure_from_torch(color, 2)
        assert color.shape[0] == vertices.shape[0], f"vertex and color must have same size " \
                                                    f"{color.shape[0]}, {vertices.shape[0]}"
        if color.dtype == np.uint8:
            color = color.astype(float) / 255.

    # Create base mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Handle categorical coloring with color indices
    if cid is not None:
        cid = ensure_from_torch(cid, dim=1).astype(int)
        color = map_quantized_color(cid, cmap=cmap)

    # Handle uniform coloring with single color index
    if ucid is not None:
        u_color = map_quantized_color(ucid, cmap=cmap)
        color = np.repeat(u_color[None, :], vertices.shape[0], 0)

    # Handle continuous coloring with float values
    if cfloat is not None:
        if cfloat_normalize:
            cfloat_min, cfloat_max = cfloat.min(), cfloat.max()
            print(f"cfloat normalize: min = {cfloat_min}, max = {cfloat_max}")
            cfloat = (cfloat - cfloat_min) / (cfloat_max - cfloat_min + 1.0e-6)
        color = matplotlib.cm.get_cmap(cfloat_cmap)(cfloat)[:, :3]

    # Apply colors if any coloring method was used
    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color[:, :3])

    # Handle texture UV coordinates
    if triangle_uvs is not None:
        try:
            triangle_uvs = ensure_from_torch(triangle_uvs, dim=3, remove_batch_dim=False)
        except AssertionError:
            # Handle indexed UV coordinates
            triangle_uvs_packed = ensure_from_torch(triangle_uvs, dim=2, remove_batch_dim=False)
            assert triangle_uv_inds is not None, "Has to provide triangle_uv_inds!"
            triangle_uv_inds = ensure_from_torch(triangle_uv_inds, dim=2, remove_batch_dim=False)
            assert triangle_uv_inds.shape[0] == len(mesh.triangles)
            assert triangle_uv_inds.shape[1] == 3
            triangle_uvs = triangle_uvs_packed[triangle_uv_inds.reshape(-1)].reshape(len(mesh.triangles), 3, 2)

        assert triangle_uvs.shape[0] == len(mesh.triangles), f"{triangle_uvs.shape}"
        assert triangle_uvs.shape[1] == 3
        assert triangle_uvs.shape[2] == 2

        mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs.reshape((-1, 2)))
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(mesh.triangles, )), dtype=np.int32))

    # Convert to tensor-based mesh if requested
    if use_new_api:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Compute vertex normals if requested
    if compute_vertex_normals:
        mesh.compute_vertex_normals()
        
    return mesh


def trimesh(geom: "trimesh.Trimesh") -> o3d.geometry.TriangleMesh:
    """Convert a trimesh geometry to Open3D mesh.

    Args:
        geom: A trimesh geometry object containing vertices and faces

    Returns:
        o3d.geometry.TriangleMesh: The converted Open3D triangle mesh
    """
    return mesh(np.array(geom.vertices), np.array(geom.faces))


def colored_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> List[o3d.geometry.TriangleMesh]:
    """Create a list of colored meshes, each with a unique color based on its index.

    Args:
        meshes: List of Open3D triangle meshes to be colored

    Returns:
        List[o3d.geometry.TriangleMesh]: List of colored meshes, where each mesh has a unique color
                                        based on its position in the input list
    """
    ans_list = []
    for mesh_id, mesh in enumerate(meshes):
        ans_list.append(colored_mesh(mesh, color_id=mesh_id))
    return ans_list


def transformed_oobb(oobb: o3d.geometry.OrientedBoundingBox, iso: Isometry) -> o3d.geometry.OrientedBoundingBox:
    """Transform an oriented bounding box by an isometry transformation.

    Args:
        oobb: The oriented bounding box to transform
        iso: The isometry transformation to apply, containing rotation and translation

    Returns:
        o3d.geometry.OrientedBoundingBox: A new transformed oriented bounding box
    """
    # Create a deep copy to avoid modifying the original
    oobb = copy.deepcopy(oobb)
    # Apply rotation around origin
    oobb.rotate(iso.q.rotation_matrix, center=np.zeros((3, 1)))
    # Apply translation
    oobb.translate(iso.t)
    return oobb


def lineset(linset_or_points: Union[np.ndarray, "torch.Tensor", o3d.geometry.LineSet],
            lines: Union[np.ndarray, "torch.Tensor"] = None,
            cid: Union[np.ndarray, "torch.Tensor"] = None,
            ucid: int = 0, cmap: str = 'tab10',
            cfloat: np.ndarray = None, cfloat_cmap: str = 'jet', cfloat_normalize: bool = False,
            color: Union[np.ndarray, "torch.Tensor"] = None):
    """Create an Open3D LineSet geometry with customizable colors.

    Args:
        linset_or_points: Either an existing Open3D LineSet or points array of shape (N,3) defining line endpoints
        lines: Array of shape (M,2) defining line segments by pairs of point indices. Required if linset_or_points is points array.
        cid: Array of shape (M,) containing integer color indices for each line segment
        ucid: Single integer color index to use for all lines. Ignored if cid is provided.
        cmap: Matplotlib colormap name to use for cid/ucid coloring. Default is 'tab10'.
        cfloat: Array of shape (M,) containing float values to map to colors
        cfloat_cmap: Matplotlib colormap name to use for cfloat values. Default is 'jet'.
        cfloat_normalize: Whether to normalize cfloat values to [0,1] range. Default is False.
        color: Array of shape (M,3) containing explicit RGB colors for each line segment

    Returns:
        o3d.geometry.LineSet: Open3D LineSet geometry with specified points, lines and colors
    """
    # Handle input as either LineSet or points array
    if isinstance(linset_or_points, o3d.geometry.LineSet):
        assert lines is None
        points = np.asarray(linset_or_points.points).copy()
        lines = np.asarray(linset_or_points.lines).copy()
    else:
        points = ensure_from_torch(linset_or_points, 2)
        lines = ensure_from_torch(lines, 2)

    # Create LineSet geometry
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines))

    # Handle explicit colors
    if color is not None:
        color = ensure_from_torch(color, 2)

    # Handle color indices
    if cid is not None:
        cid = ensure_from_torch(cid, dim=1).astype(int)
        color = map_quantized_color(cid, cmap=cmap)
        assert color.shape[0] == lines.shape[0]

    # Handle uniform color index
    if ucid is not None:
        # Use ucid to create uniform color for all lines
        u_color = map_quantized_color(ucid, cmap=cmap)
        color = np.repeat(u_color[None, :], lines.shape[0], 0)

    # Handle float color values
    if cfloat is not None:
        if cfloat_normalize:
            cfloat_min, cfloat_max = cfloat.min(), cfloat.max()
            print(f"cfloat normalize: min = {cfloat_min}, max = {cfloat_max}")
            cfloat = (cfloat - cfloat_min) / (cfloat_max - cfloat_min + 1.0e-6)
        color = matplotlib.cm.get_cmap(cfloat_cmap)(cfloat)[:, :3]

    # Set colors and return
    geom.colors = o3d.utility.Vector3dVector(color)
    return geom


def ray(rays_o: Union[np.ndarray, "torch.Tensor"], 
        rays_d: Union[np.ndarray, "torch.Tensor"], 
        length: float = 3.0) -> o3d.geometry.LineSet:
    """Create line segments representing rays in 3D space.

    Args:
        rays_o: Union[np.ndarray, "torch.Tensor"] Array of shape (N,3) containing ray origin points
        rays_d: Union[np.ndarray, "torch.Tensor"] Array of shape (N,3) containing ray direction vectors 
        length: float Length of the ray segments to visualize. Default is 3.0.

    Returns:
        o3d.geometry.LineSet: Open3D LineSet geometry containing N line segments
            representing the rays, starting at rays_o and extending in direction rays_d
    """
    # Convert inputs to numpy if needed
    rays_o = ensure_from_torch(rays_o, 2)  # Shape: (N,3)
    rays_d = ensure_from_torch(rays_d, 2)  # Shape: (N,3)
    
    # Validate input shapes
    assert rays_o.shape[0] == rays_d.shape[0]
    assert rays_o.shape[1] == 3 and rays_d.shape[1] == 3
    
    # Normalize ray directions
    rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True)
    
    # Calculate ray endpoints
    rays_e = rays_o + rays_d * length
    
    # Create line segments from origins to endpoints
    return lineset(np.concatenate([rays_o, rays_e], axis=0), 
                  np.stack([np.arange(len(rays_o)), np.arange(len(rays_o)) + len(rays_o)], axis=1))


def lineset_mesh(lst: o3d.geometry.LineSet, radius: float, resolution: int = 10) -> o3d.geometry.TriangleMesh:
    """Create a triangle mesh representing a 3D tube around line segments.

    This function converts a LineSet into a tubular mesh by creating circular cross-sections
    around each line segment and connecting them with triangular faces. The tubes have
    constant radius along their length.

    Args:
        lst: o3d.geometry.LineSet
            The input line set containing line segments to be converted to tubes
        radius: float 
            Radius of the tubular mesh around each line segment
        resolution: int, optional
            Number of vertices to use for each circular cross-section. Default is 10.
            Higher values create smoother tubes but increase geometry complexity.

    Returns:
        o3d.geometry.TriangleMesh:
            A triangle mesh representing the tubular geometry around the input line segments,
            with colors preserved from the input LineSet if present.
    """
    # Extract endpoints and connectivity from line set
    end_points = np.asarray(lst.points)
    point_indices = np.asarray(lst.lines)

    # Get start and end points for each line segment
    line_starts = end_points[point_indices[:, 0]]
    line_ends = end_points[point_indices[:, 1]]

    # Compute local coordinate frame for each tube
    line_z = line_ends - line_starts  # Tube axis direction
    line_x = np.cross(line_z, [0, 1, 0])  # First basis vector perpendicular to tube axis
    line_x_norm = np.linalg.norm(line_x, axis=1, keepdims=True)
    # Handle cases where line is parallel to [0,1,0]
    bad_norm_mask = line_x_norm[:, 0] < 0.001
    line_x[~bad_norm_mask] /= line_x_norm[~bad_norm_mask]
    line_x[bad_norm_mask, 0] = 1
    # Complete orthonormal basis
    line_y = np.cross(line_z, line_x)
    line_y = line_y / np.linalg.norm(line_y, axis=1, keepdims=True)
    # Scale basis vectors by radius
    line_x *= radius
    line_y *= radius

    # We assume there is one center in the circle, to form good looking circles.
    # Generate vertices for tube cross-sections
    start_circ_verts = [line_starts + np.cos(angle) * line_x + np.sin(angle) * line_y
                        for angle in np.linspace(0.0, 2 * np.pi, resolution)] + [line_starts]
    end_circ_verts = [line_ends + np.cos(angle) * line_x + np.sin(angle) * line_y
                      for angle in np.linspace(0.0, 2 * np.pi, resolution)] + [line_ends]
    n_start = len(start_circ_verts)
    n_end = len(end_circ_verts)
    # Body s,e,e; Body s,e,s; Top circle; Bottom circle.
    face_ids = [[sid, n_start + (sid + 1) % resolution, n_start + sid] for sid in range(resolution)] + \
               [[sid, (sid + 1) % resolution, n_start + (sid + 1) % resolution] for sid in range(resolution)] + \
               [[n_start - 1, (sid + 1) % resolution, sid] for sid in range(resolution)] + \
               [[n_start + n_end - 1,n_start + sid, n_start + (sid + 1) % resolution] for sid in range(resolution)]
    
    # Combine vertices and adjust face indices for multiple tubes
    all_verts = np.stack(start_circ_verts + end_circ_verts, axis=1)
    face_ids = np.asarray(face_ids)[None, :, :] + \
               (np.arange(all_verts.shape[0]) * all_verts.shape[1])[:, None, None]

    # Handle colors if present in input LineSet
    if lst.has_colors():
        line_colors = np.asarray(lst.colors)
        line_colors = np.repeat(line_colors[:, None, :], all_verts.shape[1], axis=1)
        line_colors = line_colors.reshape(-1, 3)
    else:
        line_colors = None

    # Create and return final mesh
    return mesh(all_verts.reshape(-1, 3), face_ids.reshape(-1, 3), color=line_colors)


def wireframe(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
    """Convert a triangle mesh to a wireframe representation using line segments.

    This function extracts the edges from a triangle mesh and creates a line set representation.
    Each edge in the resulting wireframe appears only once, even if it is shared by multiple triangles.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input triangle mesh to convert to wireframe

    Returns:
        o3d.geometry.LineSet: Wireframe representation of the input mesh as a line set
    """
    # Extract vertex coordinates and triangle indices
    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create edge pairs from triangles by taking pairs of vertices
    # Each triangle contributes 3 edges: (v0,v1), (v1,v2), (v0,v2)
    wireframe_ids = np.vstack([
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]
    ])
    # Sort vertex indices for each edge to normalize edge direction
    wireframe_ids = np.sort(wireframe_ids, axis=1)
    # Remove duplicate edges that are shared between triangles
    wireframe_ids = np.unique(wireframe_ids, axis=0)

    # Create and return line set geometry
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(wireframe_ids))

    return geom


def wireframe_bbox(extent_min: Optional[Union[List[float], np.ndarray]] = None,
                   extent_max: Optional[Union[List[float], np.ndarray]] = None,
                   bbox: Optional[BoundingBox] = None,
                   solid: bool = False,
                   tube: bool = False,
                   tube_radius: float = 0.001,
                   cid: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
                   ucid: int = 0,
                   cmap: str = 'tab10',
                   cfloat: Optional[np.ndarray] = None,
                   cfloat_cmap: str = 'jet',
                   cfloat_normalize: bool = False,
                   color: Optional[Union[np.ndarray, "torch.Tensor"]] = None):
    """Create a wireframe or solid visualization of one or more bounding boxes.

    This function creates a visual representation of bounding boxes either from min/max extents
    or from a BoundingBox object. When from min/max extents, the boxes are assumed axis-aligned.

    Args:
        extent_min: Minimum coordinates of bounding box(es). Shape (N,3) or (3,)
        extent_max: Maximum coordinates of bounding box(es). Shape (N,3) or (3,)
        bbox: BoundingBox object to visualize (alternative to extent_min/max)
        solid: If True, render as solid mesh. If False, render as wireframe 
        tube: If True and solid=False, render wireframe as meshes with tubes
        tube_radius: Radius of tubes when tube=True
        cid: Integer color indices for each box. Shape (N,)
        ucid: Default color index if cid not provided
        cmap: Matplotlib colormap name for cid colors
        cfloat: Float values for coloring boxes. Shape (N,)
        cfloat_cmap: Matplotlib colormap name for cfloat values
        cfloat_normalize: If True, normalize cfloat values to [0,1]
        color: Direct RGB(A) colors for boxes. Shape (N,3) or (N,4)

    Returns:
        o3d.geometry.LineSet or o3d.geometry.TriangleMesh: Visual representation of the box(es)
    """
    # Create box geometry either from extent bounds or BoundingBox object
    if bbox is None:
        if extent_min is None:
            extent_min = [0.0, 0.0, 0.0]
        if extent_max is None:
            extent_max = [1.0, 1.0, 1.0]

        if isinstance(extent_min, list):
            extent_min = np.array(extent_min)
        if isinstance(extent_max, list):
            extent_max = np.array(extent_max)

        # Ensure (N, 3) numpy arrays
        if not isinstance(extent_max[0], Iterable):
            extent_max = extent_max[None, :]
        if not isinstance(extent_min[0], Iterable):
            extent_min = extent_min[None, :]
        extent_min = ensure_from_torch(extent_min, 2)
        extent_max = ensure_from_torch(extent_max, 2)
        assert extent_min.shape[0] == extent_max.shape[0]

        # Generate box vertices from min/max extents
        min_x, min_y, min_z = extent_min[:, 0], extent_min[:, 1], extent_min[:, 2]
        max_x, max_y, max_z = extent_max[:, 0], extent_max[:, 1], extent_max[:, 2]
        all_points = np.stack([
            min_x, min_y, min_z, min_x, min_y, max_z, min_x, max_y, min_z, min_x, max_y, max_z,
            max_x, min_y, min_z, max_x, min_y, max_z, max_x, max_y, min_z, max_x, max_y, max_z
        ], axis=1).reshape(-1, 3)
        n_box = extent_min.shape[0]

    else:
        all_points = bbox.vertices
        n_box = 1

    # Process color inputs - only one color specification can be used
    if cid is not None:
        cid = ensure_from_torch(cid, dim=1).astype(int)
        ucid = None

    if cfloat is not None:
        cfloat = ensure_from_torch(cfloat, dim=1)
        ucid = None

    if color is not None:
        color = ensure_from_torch(color)
        ucid = None

    # Create either wireframe or solid geometry
    if not solid:
        # Define edges connecting box vertices
        line_indices = np.asarray([
            [0, 1], [2, 3], [4, 5], [6, 7],  # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7],  # Horizontal edges
            [0, 2], [4, 6], [1, 3], [5, 7]   # Depth edges
        ])
        line_indices = (line_indices[None, ...] + (np.arange(n_box) * 8)[:, None, None]).reshape(-1, 2)
        geom = lineset(
            all_points, line_indices,
            cid=np.repeat(cid[:, None], repeats=12, axis=1).flatten() if cid is not None else None,
            ucid=ucid, cmap=cmap,
            cfloat=np.repeat(cfloat[:, None], repeats=12, axis=1).flatten()
            if cfloat is not None else None,
            cfloat_cmap=cfloat_cmap,
            cfloat_normalize=cfloat_normalize,
            color=np.repeat(color[:, None, :], repeats=12, axis=1).reshape((-1, color.shape[1]))
            if color is not None else None
        )
        if tube:
            geom = lineset_mesh(geom, tube_radius)
    else:
        # Define triangles for solid box faces
        cube_indices = np.asarray([
            [0, 4, 5], [0, 5, 1],  # Front face
            [4, 6, 7], [4, 7, 5],  # Right face
            [2, 7, 6], [2, 3, 7],  # Back face
            [0, 3, 2], [0, 1, 3],  # Left face
            [7, 1, 5], [3, 1, 7],  # Top face
            [2, 6, 0], [0, 6, 4]   # Bottom face
        ])
        cube_indices = (cube_indices[None, ...] + (np.arange(n_box) * 8)[:, None, None]).reshape(-1, 3)
        geom = mesh(
            all_points, cube_indices,
            cid=np.repeat(cid[:, None], repeats=8, axis=1).flatten() if cid is not None else None,
            ucid=ucid, cmap=cmap,
            cfloat=np.repeat(cfloat[:, None], repeats=8, axis=1).flatten()
            if cfloat is not None else None,
            cfloat_cmap=cfloat_cmap,
            cfloat_normalize=cfloat_normalize,
            color=np.repeat(color[:, None, :], repeats=8, axis=1).reshape((-1, color.shape[1]))
            if color is not None else None
        )

    return geom


def merged_linesets(lineset_list: list):
    merged_points = []
    merged_inds = []
    merged_colors = []
    point_acc_ind = 0
    for ls in lineset_list:
        merged_points.append(np.asarray(ls.points))
        merged_inds.append(np.asarray(ls.lines) + point_acc_ind)
        if ls.has_colors():
            merged_colors.append(np.asarray(ls.colors))
        else:
            merged_colors.append(np.zeros((len(ls.lines), 3)))
        point_acc_ind += len(ls.points)

    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack(merged_points)),
        lines=o3d.utility.Vector2iVector(np.vstack(merged_inds))
    )
    geom.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))
    return geom


def merged_entities(merged_list: list):
    """
    Group Open3D objects belong to the same type into an entire one.
    :param merged_list:
    :return: output list contains different types of merged shapes
    """
    from itertools import groupby
    from functools import reduce
    merged_list = groupby(merged_list, lambda t: type(t))
    final_list = []
    for geom_type, geom_instances in merged_list:
        if geom_type == o3d.geometry.TriangleMesh or geom_type == o3d.geometry.PointCloud:
            final_list.append(reduce(lambda x, y: x + y, geom_instances))
        elif geom_type == o3d.geometry.LineSet:
            final_list.append(merged_linesets(geom_instances))
    return final_list


def bbox_from_points(pc: np.ndarray, oriented: bool = True, solid: bool = False, z_correct_angle: float = 0.0):
    """
    Create a bounding box from a set of points, either axis-aligned or oriented along principal components.

    Args:
        pc (np.ndarray): Input point cloud of shape (N, 3) where N is number of points
        oriented (bool): If True, compute oriented bounding box using PCA. If False, compute axis-aligned box
        solid (bool): If True, create solid box. If False, create wireframe box
        z_correct_angle (float): Additional rotation angle in degrees around z-axis to apply to oriented box

    Returns:
        tuple: (geom, iso) where:
            - geom (o3d.geometry.LineSet): Line set geometry representing the bounding box
            - iso (Isometry): Isometry transform from box local coordinates to world coordinates
    """
    # Return empty box if not enough points for PCA
    if pc.shape[0] < 5:
        return o3d.geometry.LineSet(), Isometry()

    # Center the point cloud
    pc_mean = np.mean(pc, axis=0)
    pc_centered = pc - pc_mean

    if oriented:
        # Compute oriented box using PCA
        cov = pc_centered.T @ pc_centered
        try:
            # Get principal components
            u, s, vh = np.linalg.svd(cov)
        except np.linalg.LinAlgError:
            u = np.identity(3)
        # Ensure right-handed coordinate system
        if np.linalg.det(u) < 0:
            u[:, 2] = -u[:, 2]
        # Apply additional z-axis rotation if specified
        if z_correct_angle != 0.0:
            u = u @ Quaternion(axis=[0.0, 0.0, 1.0], degrees=z_correct_angle).rotation_matrix
        # Project points to principal component space
        pc_normal = pc_centered @ u
        extent_min, extent_max = np.min(pc_normal, axis=0), np.max(pc_normal, axis=0)
        geom = wireframe_bbox(extent_min, extent_max, solid)
        iso = Isometry.from_matrix(u, pc_mean)
    else:
        # Compute axis-aligned bounding box
        extent_min, extent_max = np.min(pc_centered, axis=0), np.max(pc_centered, axis=0)
        geom = wireframe_bbox(extent_min, extent_max, solid)
        iso = Isometry(t=pc_mean)

    return geom, iso


def trajectory(traj1: List[Union[Isometry, np.ndarray]], 
              traj2: Optional[List[Union[Isometry, np.ndarray]]] = None, 
              ucid: int = -1) -> o3d.geometry.LineSet:
    """Create a line set visualization of one or two trajectories.

    This function visualizes trajectories as connected line segments. It can display either a single
    trajectory or two trajectories with connecting lines between corresponding points. The trajectories
    can be specified either as lists of Isometry objects or as lists of 3D points.

    Args:
        traj1: List of trajectory points, either as Isometry objects or numpy arrays of shape (3,)
            representing 3D positions
        traj2: Optional second trajectory to visualize and connect to traj1. Must have same length as
            traj1 if provided. Default is None.
        ucid: Integer ID used to assign a unique color from the tab10 colormap. If -1, no color is
            assigned. Default is -1.

    Returns:
        o3d.geometry.LineSet: Line set geometry containing:
            - Connected segments of traj1
            - Connected segments of traj2 (if provided)
            - Red lines connecting corresponding points between trajectories (if traj2 provided)
    """
    # Extract translation components if trajectories contain Isometry objects
    if len(traj1) > 0 and isinstance(traj1[0], Isometry):
        traj1 = [t.t for t in traj1]
    if traj2 and isinstance(traj2[0], Isometry):
        traj2 = [t.t for t in traj2]

    # Create line set for first trajectory
    traj1_lineset = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(traj1)),
                                         lines=o3d.utility.Vector2iVector(np.vstack((np.arange(0, len(traj1) - 1),
                                                                                     np.arange(1, len(traj1)))).T))
    
    # Apply uniform coloring if ucid specified
    if ucid != -1:
        color_map = np.asarray(matplotlib.cm.get_cmap('tab10').colors)
        traj1_lineset.paint_uniform_color(color_map[ucid % 10])

    # Handle optional second trajectory
    if traj2 is not None:
        assert len(traj1) == len(traj2), "Trajectories must have equal length"
        
        # Create line set for second trajectory
        traj2_lineset = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(traj2)),
                                             lines=o3d.utility.Vector2iVector(np.vstack((np.arange(0, len(traj2) - 1),
                                                                                         np.arange(1, len(traj2)))).T))
        
        # Create red lines connecting corresponding points between trajectories
        traj_diff = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack((np.asarray(traj1), np.asarray(traj2)))),
            lines=o3d.utility.Vector2iVector(np.arange(2 * len(traj1)).reshape((2, len(traj1))).T))
        traj_diff.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]).repeat(len(traj_diff.lines), axis=0))

        # Merge all line sets
        traj1_lineset = merged_linesets([traj1_lineset, traj2_lineset, traj_diff])
    return traj1_lineset


def grid(direction: str = "XY", size: float = 1.0, count: int = 7) -> o3d.geometry.LineSet:
    """Creates a grid of lines in a specified plane.

    Creates a grid of evenly spaced lines in a plane defined by two axes. The grid is centered at the origin
    and extends size/2 units in each direction along both axes.

    Args:
        direction (str): Two-character string specifying the plane for the grid ('XY', 'YZ', or 'XZ').
            Case insensitive. Default is 'XY'.
        size (float): Total size of the grid in each direction. Default is 1.0.
        count (int): Number of lines to create in each direction. Default is 7.

    Returns:
        o3d.geometry.LineSet: Line set geometry representing the grid, colored in gray.
    """
    # Dictionary mapping axis labels to array indices
    DIR_DICT = {"X": 0, "x": 0, "Y": 1, "y": 1, "Z": 2, "z": 2}
    dir_x = DIR_DICT[direction[0]]
    dir_y = DIR_DICT[direction[1]]

    line_ends = []
    # Create horizontal lines
    for i in range(count):
        a_start = [0.0, 0.0, 0.0]
        a_end = [0.0, 0.0, 0.0]
        a_start[dir_x] = -size / 2.
        a_end[dir_x] = size / 2.
        a_start[dir_y] = a_end[dir_y] = -size / 2. + size / (count - 1) * i
        line_ends += [a_start, a_end]
    
    # Create vertical lines
    for i in range(count):
        b_start = [0.0, 0.0, 0.0]
        b_end = [0.0, 0.0, 0.0]
        b_start[dir_y] = -size / 2.
        b_end[dir_y] = size / 2.
        b_start[dir_x] = b_end[dir_x] = -size / 2. + size / (count - 1) * i
        line_ends += [b_start, b_end]
    
    # Convert to numpy array and create line indices
    line_ends = np.asarray(line_ends)
    line_inds = np.arange(len(line_ends)).reshape(-1, 2)

    # Create and return the line set with gray color
    grd_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_ends),
        lines=o3d.utility.Vector2iVector(line_inds))
    grd_lineset.paint_uniform_color([0.5, 0.5, 0.5])
    return grd_lineset


def show_segmentation_motion_interactive(pc: np.ndarray, segm: np.ndarray, motions: List[List[Isometry]],
                                         oobb: List[o3d.geometry.OrientedBoundingBox] = None, body_id: int = 0):
    cur_frame = 0
    cur_geometries = None

    def show_frame(vis):
        nonlocal cur_geometries

        draw_list = [
            pointcloud(pc[cur_frame], segm[cur_frame], is_sphere=True),
            *[frame(t, 0.2) for t in motions[cur_frame]]
        ]
        if oobb is not None:
            draw_list += [transformed_oobb(ob, t) for ob, t in zip(oobb, motions[cur_frame])]
        draw_list = [obj.transform(motions[cur_frame][body_id].inv().matrix) for obj in draw_list]
        if cur_geometries is not None:
            for obj in cur_geometries:
                vis.remove_geometry(obj, reset_bounding_box=False)
        for obj in draw_list:
            vis.add_geometry(obj, reset_bounding_box=cur_geometries is None)
        cur_geometries = draw_list

    def next_frame(vis):
        nonlocal cur_frame
        cur_frame = min(cur_frame + 1, len(pc) - 1)
        show_frame(vis)
        return True

    def prev_frame(vis):
        nonlocal cur_frame
        cur_frame = max(0, cur_frame - 1)
        show_frame(vis)
        return True

    engine = o3d.visualization.VisualizerWithKeyCallback()
    engine.create_window(window_name="Interactive", width=1024, height=768, visible=True)

    engine.register_key_callback(key=ord("K"), callback_func=prev_frame)
    engine.register_key_callback(key=ord("L"), callback_func=next_frame)
    show_frame(engine)
    engine.run()
    engine.destroy_window()


def depth_pointcloud(depth: np.ndarray, normal: np.ndarray = None, rgb: np.ndarray = None, fx: float = None, fy: float = None, 
                     cx: float = None, cy: float = None, depth_scale: float = 1000.0, pose: Isometry = Isometry(), 
                     compute_normal: bool = False, use_numpy: bool = False, numpy_norm_ray: bool = False):
    """Convert a depth image to a point cloud with optional normal and color information.

    Args:
        depth (np.ndarray): Input depth image of shape (H,W)
        normal (np.ndarray, optional): Input normal map of shape (H,W,3). Only used when use_numpy=True
        rgb (np.ndarray, optional): Input RGB image of shape (H,W,3)
        fx (float, optional): Focal length in x direction. If None, assumes 90 degree FOV
        fy (float, optional): Focal length in y direction. If None, assumes 90 degree FOV
        cx (float, optional): Principal point x coordinate. If None, uses image center
        cy (float, optional): Principal point y coordinate. If None, uses image center
        depth_scale (float, optional): Scale factor to convert depth values. Applied if depth is uint16 type
        pose (Isometry, optional): Camera pose transformation to apply to point cloud
        compute_normal (bool, optional): Whether to compute normals if not provided. Default False
        use_numpy (bool, optional): Whether to use numpy implementation instead of Open3D. Default False
        numpy_norm_ray (bool, optional): Whether to normalize ray directions in numpy implementation. Default False

    Returns:
        o3d.geometry.PointCloud: Resulting point cloud with optional normals and colors
    """
    img_h, img_w = depth.shape

    # Convert depth to float32 meters if needed
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / depth_scale

    if depth.dtype == float:
        depth = depth.astype(np.float32)

    # Validate normal input
    if normal is not None:
        assert use_numpy, "You have to set use_numpy=True to ensure a correct building"
        if compute_normal:
            print("Warning: re-computing normals even if it's provided...")
        assert normal.shape[0] == img_h and normal.shape[1] == img_w and normal.shape[2] == 3

    # Set default intrinsics if not provided
    if cx is None or cy is None:
        cx = img_w / 2
        cy = img_h / 2

    if fx is None or fy is None:
        fx = fy = cx        # Assume 90 degrees fovx.

    if use_numpy:
        # Create pixel coordinate grid and unproject to 3D rays
        xx, yy = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))
        mg = np.concatenate((xx.reshape(1, -1), yy.reshape(1, -1)), axis=0)
        mg_homo = np.vstack((mg, np.ones((1, mg.shape[1]))))
        pc = np.matmul(np.linalg.inv(np.array([
            [fx, 0, cx], [0, fy, cy], [0, 0, 1]
        ])), mg_homo)
        if numpy_norm_ray:
            pc /= np.linalg.norm(pc, axis=0)
        depth_flat = depth.ravel()
        pc = depth_flat[np.newaxis, :] * pc
        
        # Filter out invalid points (zero or infinite depth)
        pc_mask = np.logical_and(np.isfinite(depth_flat), depth_flat > 0.0)
        pc = pc[:, pc_mask].T
        if rgb is not None:
            rgb = rgb.reshape(-1, rgb.shape[-1])[pc_mask, :]
        if normal is not None:
            normal = normal.reshape(-1, normal.shape[-1])[pc_mask, :]
        pcd = pointcloud(pc, color=rgb, normal=normal)
        pcd.transform(pose.matrix)
    else:
        # Use Open3D implementation
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width=img_w, height=img_h, fx=fx, fy=fy, cx=cx, cy=cy)
        depth = o3d.geometry.Image(depth)
        if rgb is not None:
            assert rgb.shape == (img_h, img_w, 3)
            rgbd_image = o3d.geometry.RGBDImage()
            rgbd_image.depth = depth
            rgbd_image.color = o3d.geometry.Image(rgb)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, pose.inv().matrix)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic, pose.inv().matrix)

    # Compute normals if requested
    if compute_normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16))
        pcd.orient_normals_towards_camera_location(pose.t)

    return pcd


def colored_triangle_soup(mesh: o3d.geometry.TriangleMesh, color: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create a triangle mesh where each face has a unique color by duplicating vertices.
    
    This function converts a triangle mesh with per-face colors into a new mesh where each
    triangle has its own set of vertices, allowing for sharp color boundaries between faces.
    This is useful for visualization where you want distinct colors per face without color
    interpolation.

    Args:
        mesh: o3d.geometry.TriangleMesh
            Input triangle mesh
        color: np.ndarray
            Array of shape (N,3) containing RGB colors for each face,
            where N matches the number of triangles in the mesh

    Returns:
        o3d.geometry.TriangleMesh:
            New triangle mesh with duplicated vertices allowing per-face colors
    """
    # Verify color array matches number of faces
    assert len(mesh.triangles) == color.shape[0]

    # Get face indices and vertex positions
    faces = np.asarray(mesh.triangles)
    xyz = np.asarray(mesh.vertices)

    # Create new vertices by duplicating each vertex used by each face
    soup_xyz = xyz[faces.ravel()]
    # Create new face indices pointing to the duplicated vertices
    soup_faces = np.arange(soup_xyz.shape[0]).reshape(-1, 3)
    # Duplicate colors for each vertex of each face
    soup_colors = np.tile(np.expand_dims(color, 1), [1, 3, 1]).reshape(-1, 3)
    
    # Create new mesh with duplicated geometry
    soup = o3d.geometry.TriangleMesh()
    soup.vertices = o3d.utility.Vector3dVector(soup_xyz)
    soup.vertex_colors = o3d.utility.Vector3dVector(soup_colors)
    soup.triangles = o3d.utility.Vector3iVector(soup_faces)

    return soup


def from_file(path: Union[str, Path], compute_normal: bool = True, load_obj_textures: bool = True,
              multipart_force_merge: bool = True) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, List[o3d.geometry.TriangleMesh]]:
    """Load 3D geometry data from various file formats into Open3D objects.

    This function supports loading multiple 3D file formats including point clouds, meshes, and textured models.
    It handles different file extensions and formats appropriately, converting them into Open3D geometry objects.

    Args:
        path: Union[str, Path]
            Path to the input 3D geometry file
        compute_normal: bool, optional
            Whether to compute vertex normals for mesh geometry. Default is True.
        load_obj_textures: bool, optional
            For OBJ/GLB files, if True, return a list of TriangleMesh objects, each with its own texture.
        multipart_force_merge: bool, optional
            For OBJ/GLB files with multiple parts, whether to merge them into a single mesh.
            Default is True.

    Returns:
        Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, List[o3d.geometry.TriangleMesh]]:
            - For point cloud formats: Returns Open3D PointCloud
            - For mesh formats: Returns Open3D TriangleMesh
            - For textured OBJ/GLB with load_obj_textures=True: Returns list of TriangleMesh if multiple parts exist
            
    Raises:
        FileNotFoundError: If input file does not exist
        NotImplementedError: If file format is not supported
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Determine geometry type:
    suffix = path.suffix
    if suffix in [".xyz", ".xyzn", ".xyzrgb", ".pts", ".pcd"]:
        geom = o3d.io.read_point_cloud(str(path))
    elif suffix in [".bnpts", ".npts"]:
        # M.Kazhdan format of input data (point cloud)
        if suffix[1] == "b":
            data = np.fromfile(path, dtype=np.float32).reshape(-1, 6)
        else:
            data = np.genfromtxt(path)
        geom = pointcloud(data[:, :3], normal=data[:, 3:])
    elif suffix in [".stl", ".off", ".gltf"]:
        # In the future: load PBR material in gltf to render in filament
        geom = o3d.io.read_triangle_mesh(str(path))
    elif suffix in [".obj", ".glb"]:
        """
        Open3D loader does not support materials and textures very well, with the following limitations:
            1. kd is not loaded.
            2. texture (kd_map) can be loaded, but if some textures are empty, then it refuses to display
        any texture: https://github.com/isl-org/Open3D/issues/4916
        For now, we only regard kd and kd_map in MLT files,
            if load_obj_textures == True, we will then return a list of triangle meshes, each either with textures,
        or with per-vertex colors.
            else, we return a mesh with per-vertex colors, the one with textures will be the average color of the
        texture image.
        """
        import trimesh
        obj_components = trimesh.load(path, force="mesh" if multipart_force_merge else None)

        if isinstance(obj_components, trimesh.Trimesh):
            all_components = [obj_components]
        else:
            all_components = obj_components.geometry.values()

        o3d_components = []
        tex_coord_warning = False
        for comp_trimesh in all_components:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(comp_trimesh.vertices))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(comp_trimesh.faces))

            if comp_trimesh.visual.kind == 'texture':
                if comp_trimesh.visual.uv is not None:
                    tri_uv = comp_trimesh.visual.uv[comp_trimesh.faces.ravel()]
                    if not np.all(tri_uv >= 0.0) or not np.all(tri_uv <= 1.0):
                        if not tex_coord_warning:
                            logger.warning(f"Texture coordinates tiling exceed 0-1!")
                            tex_coord_warning = True
                    tri_uv = tri_uv % 1.0
                    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uv)

                material = comp_trimesh.visual.material
                if isinstance(material, trimesh.visual.material.PBRMaterial):
                    tex_img = np.asarray(material.baseColorTexture)
                else:
                    tex_img = material.image
                kd_multiplier = None
                if tex_img is not None:
                    tex_img = np.asarray(tex_img)
                    if load_obj_textures:
                        # Test pass for PNG textures.
                        tex_img = np.ascontiguousarray(np.flip(tex_img, axis=0))
                        o3d_mesh.textures = [o3d.geometry.Image(tex_img)]
                        o3d_mesh.triangle_material_ids = o3d.utility.IntVector(
                            np.zeros((len(o3d_mesh.triangles, )), dtype=np.int32))
                    else:
                        kd_multiplier = np.mean(tex_img.reshape(-1, tex_img.shape[-1]).astype(float) / 255., axis=0)[:3]
                        logger.warning(f"Detected texture of size {tex_img.shape}, using mean color {kd_multiplier}")
                        if len(kd_multiplier) < 3:
                            # Get rid of some strange 2-channel textures.
                            kd_multiplier = None

                if hasattr(material, "diffuse"):
                    # Always set v-color
                    kd_color = material.diffuse[:3].astype(float)[None, :] / 255.
                    if kd_multiplier is not None:
                        kd_color = kd_color * kd_multiplier[None, :]
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                        np.repeat(kd_color, len(o3d_mesh.vertices), axis=0))

            else:
                logger.warning(f"Unsupported material type {type(comp_trimesh.visual)}!")

            o3d_components.append(o3d_mesh)

        if load_obj_textures:
            geom = o3d_components[0] if len(o3d_components) == 1 else o3d_components
        else:
            geom = merged_entities(o3d_components)[0]

    elif suffix == ".ply":
        from plyfile import PlyData, PlyElement
        # Determine filetype by peaking into ply header
        with path.open("rb") as f:
            header_data = PlyData._parse_header(f)
        # Check if file contains faces to determine if it's a mesh or point cloud
        element_keys = [t.name for t in header_data]
        if 'face' in element_keys and header_data['face'].count > 0:
            geom = o3d.io.read_triangle_mesh(str(path))
        else:
            geom = o3d.io.read_point_cloud(str(path))
    elif suffix == ".npz":
        data = np.load(path)
        geom = pointcloud(data['points'], normal=data['normals'] if 'normals' in data.keys() else None)
    elif suffix == ".dae":
        import collada
        # A very simple loader for now:
        data = collada.Collada(str(path))
        primitive = data.geometries[0].primitives[0]
        verts = primitive.vertex
        triangles = primitive.index[:, :, 0]
        geom = mesh(verts, triangles)
    else:
        raise NotImplementedError(f"Un-recognized file type {suffix}.")

    # Compute normals if requested and applicable
    if isinstance(geom, o3d.geometry.TriangleMesh):
        if compute_normal: # and not geom.has_vertex_normals():
            geom.compute_vertex_normals()
    elif isinstance(geom, list):
        if isinstance(geom[0], o3d.geometry.TriangleMesh):
            if compute_normal:
                for g in geom:
                    g.compute_vertex_normals()

    return geom

def to_file(geom: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh], path: Union[str, Path]) -> None:
    """Save an Open3D geometry object to a file.

    This function saves point clouds or triangle meshes to various file formats.
    For USD files, it creates a scene and exports the geometry.
    For other formats, it uses Open3D's built-in writers.

    Args:
        geom: Open3D geometry object to save. Must be either a PointCloud or TriangleMesh.
        path: Output file path as string or Path object. Parent directories will be created
            if they don't exist.

    Returns:
        None, except for USD files where it returns the result of scene.export()

    Raises:
        NotImplementedError: If geometry type is not supported
    """
    # Convert path to Path object for consistent handling
    path = Path(path)
    # Create parent directories if needed
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    # Handle USD files specially using Scene
    if path.suffix.startswith('.usd'):
        import pycg.render as render
        scene = render.Scene().add_object(geom, name=path.stem)
        return scene.export(path, geometry_only=True)

    # Use appropriate Open3D writer based on geometry type
    if isinstance(geom, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(str(path), geom)
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(str(path), geom)
    else:
        raise NotImplementedError(type(geom))


class RayDistanceQuery:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_id = self.scene.add_triangles(mesh)

    def compute_occupancy(self, points):
        """ For (N, 3) array output (N,) bool occupancy, 1 is inside, 0 is outside """
        points = ensure_from_torch(points, 2)
        points = o3d.core.Tensor.from_numpy(points.astype(np.float32))
        occupancy = self.scene.compute_occupancy(points)
        return occupancy.numpy() > 0.5


# Matplotlib Stuff.
#

def compare_plots(ax, loss_name, meters):
    """
    Compare the loss in meters.
    :param ax:
    :param loss_name:
    :param meters: [ (meter_name, meter_obj), ... ]
    :return:
    """
    from .exp import AverageMeter
    from functools import reduce

    for meter in meters:
        assert isinstance(meter[1], AverageMeter)

    all_loss_dict = [t[1].loss_dict for t in meters]
    all_meter_names = [t[0] for t in meters]

    all_loss_names = set(reduce(lambda x, y: x + y, [list(t.keys()) for t in all_loss_dict]))
    assert loss_name in all_loss_names, f"Available loss-names are {all_loss_names}"

    # ax.set_title(loss_name)
    for mdict, mname in zip(all_loss_dict, all_meter_names):
        if loss_name in mdict.keys():
            line, = ax.plot(mdict[loss_name])
            line.set_label(mname)
    # ax.legend(title='Method', shadow=True)
    ax.legend()
