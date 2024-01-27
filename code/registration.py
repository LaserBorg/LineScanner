'''
http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html
http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html

https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
'''

import open3d as o3d
import numpy as np
from libs.pointcloud import *


voxel_size = 3
icp_threshold = 0.05
icp_iterations = 200
max_iteration = 1000


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # .Debug
    
    source = load_pointcloud("export/laser1a_720.pcd", voxel_size=voxel_size)
    target = load_pointcloud("export/laser1b_720.pcd", voxel_size=voxel_size)

    source = estimate_normals(source, radius=voxel_size * 2)
    target = estimate_normals(target, radius=voxel_size * 2)

    # # TEST: translate and rotate using known values
    # translate = (50, 0, 100)
    # rotate    = (0.0, 20.0, 0)
    # source    = transform_pointcloud(source, translate=translate, euler_rotate_deg=rotate)


    ########################################
    # GLOBAL REGISTRATION
    ########################################

    fpfh_source = fpfh_from_pointcloud(source, voxel_size)
    fpfh_target = fpfh_from_pointcloud(target, voxel_size)

    # # fast global registration
    # global_registration = fast_global_registration(source, target, fpfh_source, fpfh_target, voxel_size)

    # ransac global registration
    global_registration = ransac_global_registration(source, target, fpfh_source, fpfh_target, voxel_size, 
                                                  max_iteration=1000000, confidence=0.9)

    source = transform_pointcloud(source, transform=global_registration.transformation)

    print("Global registration")
    static_visualizer([source, target])



    ########################################
    # ICP REGISTRATION
    ########################################

    initial_transformation = np.identity(4)

    for i in range(icp_iterations):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, icp_threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        
        initial_transformation = reg_p2l.transformation
        source.transform(reg_p2l.transformation)

        # vis.capture_screen_image(f"temp_{i}.jpg")

    source +=target

    export_pointcloud(source, "export/icp", type="ply")

    print("ICP registration")
    static_visualizer([source, target])
