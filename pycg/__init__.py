"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from pathlib import Path
from .isometry import Isometry, ScaledIsometry

__author__ = "Jiahui Huang"
__license__ = "MIT"
__version__ = '0.4.0'

__all__ = ["Isometry", "ScaledIsometry"]


def get_assets_path():
    return Path(__file__).parent / "assets"
