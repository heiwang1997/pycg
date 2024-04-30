from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from pycg.isometry import Isometry
from pycg.camera import CameraIntrinsic
from pycg.vis import wireframe_bbox


class Renderable(ABC):
    """
    Interface for objects that can be rendered in the viewport.
        i.e. can be directly added to vis.show_3d(), or scene.add_object().
    The rendering process is controlled entirely by the user.
    """
    @abstractmethod
    def render(self, pose: Isometry, intrinsic: CameraIntrinsic) -> np.ndarray:
        """
        Main render function.
        Note that if the renderable is moved, we will move pose inversely here.
        """
        pass

    @abstractmethod
    def get_extent(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum and maximum coordinates of the object, results are (3, ), (3, ) numpy arrays.
        """
        pass

    def get_proxy_geometry(self):
        """
        Proxy geometry to display, must match get_extent() results.
        """
        return wireframe_bbox(*self.get_extent(), solid=False, ucid=-1)
