# pycg.render

`render` module mainly consists of a `Scene` class, with many utility functions to manage windows, tackle animations, etc. It supports rendering using different backends such as `Open3D`, `blender`, `polyscope`, or `pyrender`.

![](demo/render.png)

## `Scene` class

Frequently-used methods:

- `add_object` / `add_light`
- `remove_object` / `remove_light`
- `auto_plane`: Add a ground plane given the current scene geometry.
- `render_blender`: Render the current scene using blender.
- `render_opengl`: Render the current scene using OpenGL.

### Using Themes

Themes determine the render style. In PyCG we provide many predefined themes, including `ThemeAngela`, `ThemeDiffuseShadow`, `ThemeNKSR`.
You can call the `apply` method to use the theme in corresponding scenes.

## Other classes

- `CameraIntrinsic` stores intrinsic parameter of perspective cameras.
- `SceneObject` holds an object that is display-able and render-able.
- `SceneLight` holds a light that could either be a sun or point light.

## Animation Support

Please refer to the corresponding examples in `examples/` folder for more details.

<video src="../demo/animation_light.mp4" controls autoplay></video>

<video src="../demo/animation_arrow.mp4" controls autoplay></video>
