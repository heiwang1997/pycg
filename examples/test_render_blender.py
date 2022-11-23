from pycg import vis, render

if __name__ == '__main__':
    s = vis.show_3d([vis.from_file("assets/bunny.obj")], show=False)
    s.setup_blender_and_detach()
