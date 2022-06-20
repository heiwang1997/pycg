from open3d.visualization import gui

from pycg import render, vis
from pycg.isometry import Isometry
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # Create a scene with 4 boxes
    scene = render.Scene()
    bbox_geom = vis.wireframe_bbox([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], solid=True)

    scene.add_object(vis.colored_mesh(bbox_geom, ucid=0),
                     pose=Isometry(t=[0., 0., 0.]), name='cube0', attributes={'smooth_shading': False})
    scene.add_object(vis.colored_mesh(bbox_geom, ucid=1),
                     pose=Isometry(t=[0., 1.5, 0.]), name='cube1', attributes={'smooth_shading': False})
    scene.add_object(vis.colored_mesh(bbox_geom, ucid=2),
                     pose=Isometry(t=[0., 3., 0.]), name='cube2', attributes={'smooth_shading': False})
    scene.add_object(vis.colored_mesh(bbox_geom, ucid=3),
                     pose=Isometry(t=[0., 4.5, 0.]), name='cube3', attributes={'smooth_shading': False})

    scene.quick_camera()

    # For each box, we create some data ys
    x = np.linspace(0, 10, 100)
    ys = [np.sin(x), np.cos(x), np.sin(x / 2), np.cos(x / 4)]

    # Prepare matplotlib window
    plt.ion()
    figure, ax = plt.subplots(figsize=(4, 4))
    line1, = ax.plot(x, ys[0])

    # Prepare open3d window
    #   vis_manager.run = build_engines + {looped run_step}
    vis_manager = render.vis_manager
    vis_manager.add_scene(scene)
    vis_manager.build_engines(use_new_api=True)
    scene_window = vis_manager.get_scene_engine(scene)
    scene_window.mouse_mode = gui.SceneWidget.Controls.PICK_GEOMETRY

    # Event loop combining matplotlib and open3d
    last_selected_name = ""
    while True:
        vis_manager.run_step()

        # Get a list of selected geometries. Use Ctrl+click to select.
        cur_selected = scene_window.scene.get_model_geometry_names()
        if len(cur_selected) > 0:
            cur_selected = cur_selected[0]
            if cur_selected != last_selected_name:
                last_selected_name = cur_selected

                cur_idx = int(last_selected_name[4])
                line1.set_xdata(x)
                line1.set_ydata(ys[cur_idx])

                # Update the figure canvas
                figure.canvas.draw()

        figure.canvas.flush_events()

    # Please see mars-project for a SLAM system example.
