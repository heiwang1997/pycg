# pycg.vis

`vis` module is intended for fast creation and visualizing 3D assets for debugging and demonstrations.
It is a shallow warpper of Open3D, with a more intuitive interface.
Functions in this module are divided into two categories: asset functions that return drawable 3D instance, and utility functions that do some jobs.

The following visualization could be achieved under few lines of `PyCG` code, please refer to `examples/` directory on how it could be used.

<video src="../demo/scene_show.mp4" controls autoplay></video>

<video src="../demo/selection.mp4" controls autoplay></video>

## Asset functions

**Legend**: 🐬 - supports batched values with shape `(N, 3)`, meaning you can create multiple instances at a time.

- `vis.transparent(geom)`: return a transparent version of `geom`.
- `vis.text(text)`: return a 3D geometry (could be either mesh or point cloud) displaying `text`.
- `vis.pointflow(src, flow, dst)`: visualize 3D scene flow from `src` to `dst`, with scene flow being `flow`.
- `vis.frame(iso)`: return a coordinate frame with pose `iso`.
- `vis.camera(iso)`: return a camera frustum geometry with pose `iso`.
- `vis.arrow(base, target)`: 🐬 return 3D arrows pointing from `base` to `target`.
- `vis.pointcloud(pc)`: return a point cloud, normal, color, ... could be provided as additional arguments.
- `vis.mesh(v, f)`: return a triangle mesh.
- `vis.wireframe(bottom_pos, top_pos)`: 🐬 return a bounding box ranging from `bottom_pos` to `top_pos`.

## Utility functions

- `vis.show_3d([geom_a, geom_b], [geom_c, geom_d], ...)`: display two windows (cameras are synchronized), one window with `geom_a` and `geom_b`, the other window showing `geom_c` and `geom_d`. The return values are two `render.Scene` object, where you could render to a numpy image (e.g. using blender) by simply calling `.render_blender()`. 
- `vis.from_file(path)`: load geometry from `path`.
- `vis.to_file(geom, path)`: save `geom` to `path`.
