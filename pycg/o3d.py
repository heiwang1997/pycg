from pycg.exp import logger

O3D_INSTRUCTION = "pip install open3d_pycg_cpu -f https://pycg.huangjh.tech/packages/index.html"

try:
    from open3d_pycg import *
    is_custom_build = True
except ImportError:
    try:
        from open3d import *
        logger.warning(f"Customized build of Open3D is not detected, to resolve this you can do: \n{O3D_INSTRUCTION}")
        is_custom_build = False
    except ImportError:
        raise ImportError(f"Open3D not installed, to resolve this you can do: {O3D_INSTRUCTION}")


def get_resource_path():
    try:
        import open3d_pycg
        path = open3d_pycg.__path__[0]
    except ImportError:
        import open3d
        path = open3d.__path__[0]
    from pathlib import Path
    return Path(path) / "resources"
