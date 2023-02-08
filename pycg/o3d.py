from pycg.exp import logger

OPTION_1 = "(recommended, using customized Open3D that enables view sync, animation, ...) \n\n" \
           "pip install python-pycg[full] -f https://pycg.s3.ap-northeast-1.amazonaws.com/packages/index.html\n\n"
OPTION_2 = "(using official Open3D) \n\npip install python-pycg[all]\n\n"

try:
    from open3d import *
except ImportError:
    logger.error(f"Open3D not installed! You can try either the following 2 options: \n"
                 f" 1. {OPTION_1}\n"
                 f" 2. {OPTION_2}")
    raise


if hasattr(visualization.VisualizerWithKeyCallback, 'register_view_refresh_callback'):
    is_custom_build = True
else:
    is_custom_build = False
    logger.warning("Customized build of Open3D is not detected, to resolve this you can do:\n"
                   f"{OPTION_1}")


def get_resource_path():
    import open3d
    from pathlib import Path
    return Path(open3d.__path__[0]) / "resources"
