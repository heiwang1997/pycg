from pycg import vis, image
import numpy as np


if __name__ == '__main__':
    # Load bunny and show it.
    bunny_geom = vis.from_file("assets/bunny.obj")
    bunny_geom = vis.colored_mesh(bunny_geom, ucid=0)
    bunny_geom2 = vis.colored_mesh(bunny_geom, ucid=1)
    bunny_pcd = vis.pointcloud(np.asarray(bunny_geom.vertices), is_sphere=True, sphere_radius=0.002)
    vis.show_3d([bunny_geom], [bunny_geom2], [bunny_pcd], use_new_api=True)

    # Render bunny using different renderers.
    s1 = vis.show_3d([bunny_geom], show=False)
    img_filament = s1.render_filament()
    img_opengl = s1.render_opengl()
    image.show(img_filament, img_opengl, subfig_size=4)
