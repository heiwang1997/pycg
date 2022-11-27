"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from pathlib import Path

__author__ = "Jiahui Huang"
__license__ = "MIT"
__version__ = '0.2.0'


def get_assets_path():
    return Path(__file__).parent / "assets"
