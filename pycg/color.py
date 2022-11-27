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

_hex_array_to_np = lambda x: np.asarray([matplotlib.colors.to_rgb(t) for t in x])

ADDITIONAL_CMAPS = {
    'random': np.random.RandomState(0).uniform(0., 1., (1000, 3)),
    'distinct_middle_51': _hex_array_to_np(CMAP_DISTINCT_MIDDLE_51),
    'metro': _hex_array_to_np(CMAP_METRO),
    'shengyu': np.asarray(CMAP_SHENGYU)
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


def map_quantized_color(cid: Union[int, np.ndarray], cmap: str = 'tab10'):
    if cmap in ADDITIONAL_CMAPS.keys():
        color_map = ADDITIONAL_CMAPS[cmap]
    else:
        color_map = np.asarray(matplotlib.cm.get_cmap(cmap).colors)

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
