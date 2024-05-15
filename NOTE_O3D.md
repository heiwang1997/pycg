It seems that there are two problems:
1. You cannot import `open3d` package after importing `open3d_pycg`. There is some name collisions.
2. You cannot call `pycg.render_filament`, since internally it constructs `rendering.OffscreenRenderer` object, which dynamically looks for `open3d` package which is not installed or cannot be imported.

```python
import sys
import site
import os.path

from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_file_location


class MyMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if "open3d" not in fullname or "open3d_pycg" in fullname:
            return None

        fullname = fullname.replace("open3d", "open3d_pycg")

        print(fullname, path, target)

        if path is None or path == "":
            path = site.getsitepackages() # top level import -- 
        if "." in fullname:
            *parents, name = fullname.split(".")
        else:
            name = fullname

        for entry in path:
            if os.path.isdir(os.path.join(entry, name)):
                # this module has child modules
                filename = os.path.join(entry, name, "__init__.py")
                submodule_locations = [os.path.join(entry, name)]
            else:
                filename = os.path.join(entry, name + ".py")
                submodule_locations = None
            # if not os.path.exists(filename):
                # continue

            return spec_from_file_location(fullname, filename,
                submodule_search_locations=submodule_locations)

        print(f"No idea how to import {fullname}")
        return None # we don't know how to import this

# class MyLoader(Loader):
#     def __init__(self, filename):
#         self.filename = filename

#     def create_module(self, spec):
#         return None # use default module creation semantics

#     def exec_module(self, module):
#         with open(self.filename) as f:
#             data = f.read()

#         # manipulate data some way...

#         exec(data, vars(module))

# def install():
"""Inserts the finder into the import machinery"""
# print(sys.meta_path)
sys.meta_path.insert(0, MyMetaFinder())
# sys.meta_path.append(MyMetaFinder())

from pycg import vis

from pathlib import Path
import os, re
import torch
import pycg.o3d as o3d


renderer = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
# import o3d.visualization.rendering

# print(o3d)
print(o3d.utility.Vector3dVector())
```