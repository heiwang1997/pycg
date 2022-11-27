# PyCG: Toolbox for CG-related visualizations and computations

[![Publish to PyPI.org](https://github.com/heiwang1997/pycg/actions/workflows/publish.yml/badge.svg)](https://github.com/heiwang1997/pycg/actions/workflows/publish.yml)
[![Documentation Status](https://readthedocs.org/projects/pycg/badge/?version=latest)](https://pycg.readthedocs.io/en/latest/?badge=latest)

I'm writing the document of this. Please do not use it unless you are someone internal.

## Install

You could either do:
```shell
pip install -U python-pycg --extra-index-url http://eagle.huangjh.tech:8080/simple --trusted-host eagle.huangjh.tech
```

> Note that the need for the extra index URL if for our [customized version of Open3D](https://github.com/heiwang1997/Open3D) (with support for multi-window camera/light synchronization, animation maker and visualizer, scalar analyzer, etc). If you don't want this, it's still fine. You can use the normal Open3D.

Or clone this repository and install it if you want to use newest features:
```shell
git clone --recursive https://github.com/heiwang1997/pycg
python setup.py develop
```
