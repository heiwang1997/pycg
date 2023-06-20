# Welcome to PyCG

PyCG is there for people to accelerate their 3D visualizations and computations, aiming at implementing complicated functions with as few lines as possible.
The toolbox is created from the perspective of researchers and would hopefully accelerate your daily pipeline.

## Using PyCG

PyCG contains many submodules which could be easily imported via `from pycg import xxx`.
These different modules span a wide range of functionalities from visualizing 3D assets to creating nice html tables for comparing results.
Please refer to the individual documentation of each submodule to get started.

- [3D Visualization `pycg.vis`](vis.md)
- [Render 3D Scenes `pycg.render`](render.md)
- [Handling 3D Transformations `pycg.isometry`](isometry.md)
- [Compress Videos `pycg.video`](video.md)
- [Compress PDF Files `pycg.pdf`](pdf.md)
- [Experiment Utilities `pycg.exp`](exp.md)
- [Manipulating Images `pycg.image`](image.md)
<!-- - [Make Fancy HTMLs `pycg.html`](html.md) -->
- ... and so on to be added!

## Why (not) PyCG?

PyCG is yet another wrapper library that exposes many handy functionalities. 
It is never intended to be used in production and please expect breaking API changes along the development.
However, you could borrow many implementation ideas from this library!

## License

PyCG is an open-source project for free. Note there is no warranty for this software.
Send a PR if you want to contribute your code!
