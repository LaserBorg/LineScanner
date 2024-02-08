import open3d as o3d
import copy


def init_visualizer(width=800, height=800, left=1000, point_size=1.5, backface=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, left=left)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.light_on = False
    render_option.mesh_show_back_face = backface

    # view_control = vis.get_view_control()
    # view_control.set_zoom(0.2)
    # view_control.set_front((0.0, 0.0, 0.01))
    # view_control.set_lookat((0.0, 0.0, -1.0))
    # view_control.set_up((0.0, 1.0, 0.0))
    return vis

def update_visualizer(vis, object):
    # Update 3D-view each line
    vis.update_geometry(object)
    vis.poll_events()
    vis.update_renderer()

def static_visualizer(object_list, width=800, height=800, left=1000, point_size=1.5):
    vis = init_visualizer(width=width, height=height, left=left, point_size=point_size)

    # Add the geometry to the visualization window
    for object in object_list:
        vis.add_geometry(object)

    vis.run()
    vis.destroy_window()

def draw_registration_result(mesh1, mesh2, transformation, uniform_colors=False):
    mesh1_temp = copy.deepcopy(mesh1)
    mesh2_temp = copy.deepcopy(mesh2)

    if uniform_colors:
        mesh1_temp.paint_uniform_color([1, 0.706, 0])
        mesh2_temp.paint_uniform_color([0, 0.651, 0.929])

    mesh1_temp.transform(transformation)
    o3d.visualization.draw_geometries([mesh1_temp, mesh2_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556],
                                      mesh_show_back_face=True)
