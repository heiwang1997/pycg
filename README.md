# PyCG: Toolbox for CG-related visualizations and computations

[![Publish to PyPI.org](https://github.com/heiwang1997/pycg/actions/workflows/publish.yml/badge.svg)](https://github.com/heiwang1997/pycg/actions/workflows/publish.yml)
[![Documentation Status](https://readthedocs.org/projects/pycg/badge/?version=latest)](https://pycg.readthedocs.io/en/latest/?badge=latest)

I'm writing the document of this. Please do not use it unless you are someone internal.

## Install

To install a full-fledged version, use:
```bash
pip install -U python-pycg[full] -f https://pycg.s3.ap-northeast-1.amazonaws.com/packages/index.html
```

> Note that the need for the extra index URL if for our [customized version of Open3D](https://github.com/heiwang1997/Open3D) (with support for multi-window camera/light synchronization, animation maker and visualizer, scalar analyzer, etc).

If you don't want to use our customized Open3D, simply do:
```shell
pip install -U python-pycg[all]
```

By default, Open3D will not be installed if you do `pip install python-pycg` directly, but all other non-visualization-related functions should work!

For developers, clone this repository and install it if you want to use newest features:
```shell
git clone --recursive https://github.com/heiwang1997/pycg
pip install .
```
