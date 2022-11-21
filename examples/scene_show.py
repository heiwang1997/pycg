import copy, os

from pycg import animation, render, vis, image
from pycg.animation import InterpType
from pycg.isometry import Isometry


if __name__ == '__main__':
    bunny_geom = vis.from_file("assets/bunny.obj")
    bunny_geom = vis.colored_mesh(bunny_geom, ucid=0)

    bunny_geom2 = copy.deepcopy(bunny_geom)
    vis.show_3d([bunny_geom], [bunny_geom2], use_new_api=False)

    s1, s2 = vis.show_3d([bunny_geom], [bunny_geom], show=False)
    img = s1.render_filament()
    image.show(img)

    s1.preview(use_new_api=True)
    vis.show_3d([bunny_geom2], show=True, use_new_api=True)
