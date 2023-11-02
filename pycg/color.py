"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from typing import Union, List
import matplotlib.colors
import matplotlib.cm
import numpy as np

"""
Distinct Color maps:
    Generated via: http://phrogz.net/css/distinct-colors.html
    Use HSV color-mode:
        - Hue: 0-360
        - Saturation: 30%-80%
        - Value: Middle (50%-90%)
"""
CMAP_DISTINCT_MIDDLE_51 = ['#db2e2e', '#82521b', '#52c229', '#237ca8', '#75428f', '#9c2121', '#e8ae6b', '#66a84d',
                           '#6bbee8', '#871e8f', '#a84d4d', '#cf982b', '#3c824f', '#317ae8', '#c75fcf', '#e87b6b',
                           '#a88a4d', '#31e87a', '#537ab5', '#e831c3', '#e86231', '#e8cf6b', '#26b572', '#4d5aa8',
                           '#821b60', '#9c4121', '#e8dc31', '#6be8be', '#1e1e8f', '#db65b4', '#b56d53', '#8a8f42',
                           '#4da896', '#4931e8', '#c2297a', '#e87a31', '#c3e831', '#2edbd0', '#8c6be8', '#e83162',
                           '#b57a53', '#8ea823', '#26acb5', '#7226b5', '#e86b8c', '#c27a29', '#aee86b', '#238ea8',
                           '#ab31e8', '#a84d66']
CMAP_METRO = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
              '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080']
CMAP_SHENGYU = [[0.00, 0.65, 0.93], [0.84, 0.00, 0.00], [0.55, 0.24, 1.00], [0.01, 0.53, 0.00], [0.00, 0.67, 0.78],
                [0.60, 1.00, 0.00], [1.00, 0.50, 0.82], [0.42, 0.00, 0.31], [1.00, 0.65, 0.19], [0.00, 0.00, 0.62],
                [0.53, 0.44, 0.41], [0.00, 0.29, 0.26], [0.31, 0.16, 0.00], [0.00, 0.99, 0.81], [0.74, 0.72, 1.00],
                [0.58, 0.71, 0.48], [0.75, 0.02, 0.73], [0.15, 0.40, 0.64], [0.16, 0.00, 0.25], [0.86, 0.70, 0.69],
                [1.00, 0.96, 0.56], [0.31, 0.27, 0.36], [0.64, 0.49, 0.00], [1.00, 0.44, 0.40], [0.25, 0.51, 0.43],
                [0.51, 0.00, 0.05], [0.64, 0.48, 0.70], [0.20, 0.31, 0.00], [0.61, 0.89, 1.00], [0.92, 0.00, 0.47],
                [0.18, 0.00, 0.04], [0.37, 0.56, 1.00], [0.00, 0.78, 0.13], [0.35, 0.00, 0.67], [0.00, 0.12, 0.00],
                [0.60, 0.28, 0.00], [0.59, 0.62, 0.65], [0.61, 0.26, 0.36], [0.00, 0.12, 0.20], [0.78, 0.77, 0.00],
                [1.00, 0.82, 1.00], [0.00, 0.75, 0.60], [0.22, 0.08, 1.00], [0.18, 0.15, 0.15], [0.87, 0.35, 1.00],
                [0.75, 0.91, 0.75], [0.50, 0.27, 0.60], [0.32, 0.31, 0.24], [0.85, 0.40, 0.00], [0.39, 0.45, 0.22],
                [0.76, 0.45, 0.53], [0.43, 0.45, 0.54], [0.50, 0.62, 0.01], [0.75, 0.55, 0.40], [0.39, 0.20, 0.22],
                [0.79, 0.80, 0.85], [0.42, 0.92, 0.51], [0.13, 0.25, 0.41], [0.64, 0.50, 1.00], [1.00, 0.01, 0.80],
                [0.46, 0.74, 0.99], [0.85, 0.76, 0.51], [0.81, 0.64, 0.81], [0.43, 0.31, 0.00], [0.00, 0.41, 0.45],
                [0.28, 0.62, 0.37], [0.58, 0.78, 0.75], [0.98, 1.00, 0.00], [0.75, 0.33, 0.27], [0.00, 0.40, 0.24],
                [0.36, 0.31, 0.66], [0.33, 0.13, 0.39], [0.31, 0.37, 1.00], [0.49, 0.56, 0.47], [0.73, 0.03, 0.98],
                [0.55, 0.57, 0.76], [0.70, 0.00, 0.21], [0.53, 0.38, 0.49], [0.62, 0.00, 0.46], [1.00, 0.87, 0.77],
                [0.32, 0.03, 0.00], [0.10, 0.03, 0.00], [0.30, 0.54, 0.71], [0.00, 0.87, 0.87], [0.78, 1.00, 0.98],
                [0.19, 0.21, 0.08], [1.00, 0.15, 0.28], [1.00, 0.59, 0.67], [0.02, 0.00, 0.10], [0.79, 0.38, 0.69],
                [0.76, 0.64, 0.22], [0.49, 0.31, 0.23], [0.98, 0.62, 0.47], [0.34, 0.40, 0.39], [0.82, 0.58, 1.00],
                [0.18, 0.12, 0.41], [0.25, 0.11, 0.20], [0.69, 0.58, 0.60]]
CMAP_PPT_DEFAULT = [[0.278, 0.329, 0.408], [0.906, 0.902, 0.902], [0.302, 0.447, 0.745], [0.875, 0.510, 0.267],
                    [0.647, 0.647, 0.647], [0.965, 0.757, 0.263], [0.412, 0.604, 0.816], [0.494, 0.671, 0.333]]
CMAP_PPT_BLUE_WARM = [[0.141, 0.161, 0.310], [0.698, 0.796, 0.961], [0.310, 0.404, 0.655], [0.435, 0.612, 0.804],
                      [0.255, 0.498, 0.812], [0.510, 0.561, 0.655], [0.420, 0.627, 0.675], [0.608, 0.569, 0.624]]
CMAP_PPT_BLUE = [[0.129, 0.251, 0.416], [0.875, 0.937, 0.973], [0.188, 0.435, 0.753], [0.263, 0.612, 0.831],
                 [0.369, 0.804, 0.843], [0.373, 0.796, 0.620], [0.557, 0.780, 0.435], [0.671, 0.753, 0.361]]
CMAP_PPT_BLUE2 = [[0.235, 0.353, 0.447], [0.878, 0.890, 0.898], [0.314, 0.671, 0.875], [0.255, 0.510, 0.757],
                  [0.384, 0.796, 0.835], [0.396, 0.718, 0.600], [0.318, 0.525, 0.345], [0.443, 0.631, 0.624]]
CMAP_PPT_BLUE_GREEN = [[0.216, 0.208, 0.267], [0.816, 0.859, 0.898], [0.310, 0.576, 0.714], [0.439, 0.706, 0.745],
                       [0.522, 0.733, 0.659], [0.490, 0.545, 0.557], [0.549, 0.671, 0.710], [0.255, 0.510, 0.757]]
CMAP_PPT_GREEN = [[0.290, 0.369, 0.322], [0.886, 0.871, 0.824], [0.400, 0.608, 0.278], [0.580, 0.714, 0.290],
                  [0.765, 0.804, 0.337], [0.259, 0.576, 0.471], [0.404, 0.702, 0.761], [0.231, 0.533, 0.678]]
CMAP_PPT_YELLOW_GREEN = [[0.290, 0.369, 0.322], [0.886, 0.875, 0.808], [0.643, 0.788, 0.322], [0.451, 0.635, 0.282],
                         [0.349, 0.643, 0.455], [0.412, 0.745, 0.643], [0.412, 0.694, 0.800], [0.435, 0.757, 0.957]]
CMAP_PPT_YELLOW = [[0.220, 0.188, 0.169], [0.894, 0.871, 0.859], [0.969, 0.796, 0.278], [0.922, 0.592, 0.243],
                   [0.773, 0.561, 0.302], [0.867, 0.463, 0.200], [0.835, 0.329, 0.200], [0.584, 0.424, 0.420]]
CMAP_PPT_YELLOW_ORANGE = [[0.294, 0.235, 0.192], [0.976, 0.933, 0.804], [0.898, 0.643, 0.286], [0.612, 0.404, 0.322],
                          [0.686, 0.553, 0.510], [0.741, 0.600, 0.451], [0.624, 0.584, 0.471], [0.718, 0.471, 0.227]]
CMAP_PPT_ORANGE = [[0.400, 0.435, 0.333], [0.812, 0.867, 0.914], [0.843, 0.529, 0.208], [0.694, 0.365, 0.216],
                   [0.498, 0.345, 0.267], [0.592, 0.514, 0.365], [0.757, 0.737, 0.529], [0.588, 0.624, 0.541]]
CMAP_PPT_VIOLET2 = [[0.361, 0.196, 0.373], [0.914, 0.898, 0.922], [0.525, 0.196, 0.545], [0.569, 0.365, 0.800],
                    [0.439, 0.380, 0.824], [0.392, 0.376, 0.698], [0.369, 0.643, 0.906], [0.380, 0.514, 0.835]]
CMAP_PPT_NEUTRAL = [[0.451, 0.376, 0.341], [0.914, 0.867, 0.776], [0.604, 0.710, 0.812], [0.820, 0.518, 0.322],
                    [0.651, 0.667, 0.522], [0.824, 0.702, 0.412], [0.518, 0.651, 0.616], [0.580, 0.549, 0.549]]
CMAP_PPT_PAPER = [[0.275, 0.298, 0.169], [0.996, 0.980, 0.808], [0.659, 0.706, 0.584], [0.910, 0.655, 0.349],
                  [0.882, 0.741, 0.294], [0.780, 0.584, 0.651], [0.596, 0.529, 0.737], [0.525, 0.620, 0.749]]
CMAP_PPT_SUBTITLE = [[0.369, 0.369, 0.369], [0.867, 0.867, 0.867], [0.325, 0.537, 0.686], [0.667, 0.710, 0.271],
                     [0.914, 0.588, 0.212], [0.514, 0.514, 0.514], [0.961, 0.769, 0.267], [0.812, 0.361, 0.212]]
CMAP_PPT_FLOW = [[0.133, 0.153, 0.263], [0.733, 0.859, 0.969], [0.322, 0.408, 0.761], [0.478, 0.792, 0.937],
                 [0.714, 0.906, 0.416], [0.482, 0.796, 0.690], [0.937, 0.525, 0.239], [0.875, 0.314, 0.204]]
CMAP_PPT_VIEWPOINT = [[0.196, 0.196, 0.196], [0.886, 0.871, 0.824], [0.886, 0.518, 0.200], [0.576, 0.200, 0.224],
                      [0.173, 0.341, 0.475], [0.357, 0.514, 0.290], [0.361, 0.290, 0.459], [0.733, 0.600, 0.384]]

_hex_array_to_np = lambda x: np.asarray([matplotlib.colors.to_rgb(t) for t in x])
_canonicalize_ppt_color = lambda x: np.asarray([x[tidx] for tidx in [2, 3, 4, 5, 6, 7, 0, 1]])

ADDITIONAL_CMAPS = {
    'random': np.random.RandomState(0).uniform(0., 1., (1000, 3)),
    'distinct_middle_51': _hex_array_to_np(CMAP_DISTINCT_MIDDLE_51),
    'metro': _hex_array_to_np(CMAP_METRO),
    'shengyu': np.asarray(CMAP_SHENGYU),
    'ppt_default': _canonicalize_ppt_color(CMAP_PPT_DEFAULT),
    'ppt_blue_warm': _canonicalize_ppt_color(CMAP_PPT_BLUE_WARM),
    'ppt_blue': _canonicalize_ppt_color(CMAP_PPT_BLUE),
    'ppt_blue2': _canonicalize_ppt_color(CMAP_PPT_BLUE2),
    'ppt_orange': _canonicalize_ppt_color(CMAP_PPT_ORANGE),
    'ppt_paper': _canonicalize_ppt_color(CMAP_PPT_PAPER),
    'ppt_viewpoint': _canonicalize_ppt_color(CMAP_PPT_VIEWPOINT)
}

_void_color = [0.0, 0.0, 0.0]


def set_void_color(color=None):
    assert len(color) == 3, "Color must be 3-dimensional!"
    _void_color[0] = color[0]
    _void_color[1] = color[1]
    _void_color[2] = color[2]


def shuffle_custom_cmaps(seed: int = 0):
    for cmap in ADDITIONAL_CMAPS.values():
        cmap[:, ] = cmap[np.random.RandomState(seed).permutation(cmap.shape[0])]


def get_cmap_array(cmap: str):
    if "@" in cmap:
        cmap, cmap_processor = cmap.split("@")
    else:
        cmap, cmap_processor = cmap, None

    if cmap in ADDITIONAL_CMAPS.keys():
        color_map = ADDITIONAL_CMAPS[cmap]
    else:
        color_map = np.asarray(matplotlib.cm.get_cmap(cmap).colors)

    if cmap_processor is not None:
        if cmap_processor == "shuffle":
            color_map = color_map[np.random.RandomState(0).permutation(color_map.shape[0])]
        elif cmap_processor == "double-lighter":
            lighter_map = 0.5 * color_map + 0.5 * np.ones_like(color_map)
            color_map = np.hstack([color_map, lighter_map]).reshape((-1, 3))
        elif cmap_processor == "double-darker":
            darker_map = 0.7 * color_map
            color_map = np.hstack([color_map, darker_map]).reshape((-1, 3))
        else:
            raise NotImplementedError("Unknown cmap processor: {}".format(cmap_processor))

    return color_map


def map_quantized_color(cid: Union[int, np.ndarray], cmap: str = 'tab10'):
    color_map = get_cmap_array(cmap)

    single_value = False
    if isinstance(cid, int) or isinstance(cid, np.int64):
        cid = np.full((1, ), cid, dtype=int)
        single_value = True

    cid[cid >= color_map.shape[0]] = cid[cid >= color_map.shape[0]] % (color_map.shape[0])
    # color id < 0 will be assigned void color.
    color_map = np.vstack([color_map, np.array(_void_color)[None, :]])
    cid[cid < 0] = color_map.shape[0] - 1
    color = color_map[cid]
    return color if not single_value else color[0]
