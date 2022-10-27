from open3d import *
from pycg.exp import logger

if hasattr(visualization.VisualizerWithKeyCallback, 'register_view_refresh_callback'):
    is_custom_build = True
else:
    is_custom_build = False
    logger.warning("Customized build of Open3D is not detected. pycg might not function correctly")
