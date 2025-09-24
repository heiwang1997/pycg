# PyCG: Toolbox for CG-related visualizations and computations

[![Publish to PyPI.org](https://github.com/heiwang1997/pycg/actions/workflows/publish.yml/badge.svg)](https://github.com/heiwang1997/pycg/actions/workflows/publish.yml)
[![Documentation Status](https://readthedocs.org/projects/pycg/badge/?version=latest)](https://pycg.readthedocs.io/en/latest/?badge=latest)

PyCG is there for people to accelerate their 3D visualizations and computations, aiming at implementing complicated functions with as few lines as possible.
The toolbox is created from the perspective of researchers and would hopefully accelerate your daily pipeline.

## Install

```shell
pip install -U python-pycg[all]
```

For developers, clone this repository and install it if you want to use newest features:
```shell
git clone --recursive https://github.com/heiwang1997/pycg
pip install -e .[all]
```

## Using PyCG

PyCG contains many submodules which could be easily imported via `from pycg import xxx`.
These different modules span a wide range of functionalities from visualizing 3D assets to creating nice html tables for comparing results.
Please refer to the individual documentation of each submodule to get started.

- 3D Visualization `pycg.vis`
- Render 3D Scenes `pycg.render`
- Handling 3D Transformations `pycg.isometry`
- Compress Videos `pycg.video`
- Compress PDF Files `pycg.pdf`
- Experiment Utilities `pycg.exp`
- Manipulating Images `pycg.image`
- ... and so on to be added!

## Gallery

![](docs/demo/render.png)

<video src="https://github.com/heiwang1997/pycg/raw/master/docs/demo/scene_show.mp4" controls autoplay></video>

<video src="https://github.com/heiwang1997/pycg/raw/master/docs/demo/selection.mp4" controls autoplay></video>

<video src="https://github.com/heiwang1997/pycg/raw/master/docs/demo/animation_light.mp4" controls autoplay></video>

<video src="https://github.com/heiwang1997/pycg/raw/master/docs/demo/animation_arrow.mp4" controls autoplay></video>
