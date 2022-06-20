from pycg.render import Scene
from pycg import vis
from pycg.isometry import Isometry

if __name__ == '__main__':
    # Create a scene with two bunnies and two lights (sun and point)
    #   We don't use a theme to show clear effects.

    bunny_geom = vis.from_file("assets/bunny.obj")
    bunny_geom_1 = vis.colored_mesh(bunny_geom, ucid=0)
    bunny_geom_2 = vis.colored_mesh(bunny_geom, ucid=1)

    # Create a scene with a bunny inside
    scene = Scene()
    scene.viewport_shading = 'LIT'

    # Setting the up axis, so the default configurations could fit well
    scene.up_axis = '+Y'
    scene.add_object(bunny_geom_1,
                     name='bunny1',
                     pose=Isometry(t=[0.1, 0.0, 0.0]),
                     attributes={"material.metallic": 0.8, "material.roughness": 0.2})
    scene.add_object(bunny_geom_2,
                     name='bunny2',
                     pose=Isometry(t=[-0.1, 0.0, 0.0]),
                     attributes={"material.roughness": 1.0})

    scene.quick_camera(w=800, h=800, plane_angle=-90.0, fill_percent=0.8)
    scene.auto_plane(scale=1.0, dist_ratio=0.0)

    # Setup lights: no background (ibr/env) light
    scene.ambient_color[-1] = 0.0
    scene.add_light_sun('sun', light_energy=2.0)
    scene.add_light_point('point', pos=[0.0, 0.1, 0.1], energy=2.0)

    scene.export("out/exported_2_bunny.usdc")

    scene.preview(use_new_api=True)

