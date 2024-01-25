# http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html
# http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html

import open3d as o3d
import numpy as np
from libs.pointcloud import export_pointcloud

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    source = o3d.io.read_point_cloud("export/laser1a_720.pcd")
    target = o3d.io.read_point_cloud("export/laser1b_720.pcd")

    # # downsample
    # source = source.voxel_down_sample(voxel_size=0.01)
    # target = target.voxel_down_sample(voxel_size=0.01)

    # # calculate normals
    # source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # translate
    source.translate((50, 0, 100))  # 3dsMax: [-50, 100, 0]

    # rotate
    euler_rotation = [0.0, 20.0, 0]  # 3dsMax: [0, 0, -20]
    R = source.get_rotation_matrix_from_xyz(np.deg2rad(euler_rotation))
    source.rotate(R, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.1  # 0.05
    icp_iteration = 100

    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        # vis.capture_screen_image(f"temp_{i}.jpg")
    vis.destroy_window()

    source +=target

    export_pointcloud(source, "export/icp", type="ply")

    # o3d.visualization.RenderOption.light_on = False  # TODO not working
    o3d.visualization.draw_geometries([source], width=800, height=800, left=1000,
                                      mesh_show_back_face=False, zoom=0.2, up=[0.0, 1.0, 0.0],
                                      front=[0.0, 0.0, 0.01], lookat=[0.0, 0.0, -1.0])
