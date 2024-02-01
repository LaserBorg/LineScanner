'''
http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html
http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html

https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
'''

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

from libs.pointcloud import load_pointcloud, estimate_normals, fpfh_from_pointcloud, ransac_global_registration, transform_pointcloud, static_visualizer, export_pointcloud


def get_transform_vectors(transform_M):
    # Extract translation (top-right 3x1 sub-matrix)
    translation = transform_M[:3, 3]

    # Extract rotation (top-left 3x3 sub-matrix), make a copy to avoid read only error
    rotation_M = np.array(transform_M[:3, :3])
    # Convert rotation matrix to Euler angles
    r = R.from_matrix(rotation_M)
    euler_angles = r.as_euler('xyz', degrees=True)

    return translation, euler_angles


voxel_size = 3
icp_threshold = 0.01
max_iteration = 1000
verbose = False


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # .Debug
    
    source = load_pointcloud("export/laser1a_720.pcd", voxel_size=voxel_size, verbose=verbose)
    target = load_pointcloud("export/laser1b_720.pcd", voxel_size=voxel_size, verbose=verbose)

    source = estimate_normals(source, radius=voxel_size * 2)
    target = estimate_normals(target, radius=voxel_size * 2)


    # GROUNDTRUTH
    groundtruth_translation = (50, 0, 100)
    groundtruth_euler       = (0.0, 20.0, 0)
    # groundtruth_source    = transform_pointcloud(source, translate=translate, euler_rotate_deg=rotate)


    ########################################
    # GLOBAL REGISTRATION (using RANSAC or FAST)
    ########################################

    fpfh_source = fpfh_from_pointcloud(source, voxel_size)
    fpfh_target = fpfh_from_pointcloud(target, voxel_size)

    # # fast global registration
    # global_registration = fast_global_registration(source, target, fpfh_source, fpfh_target, voxel_size)

    # ransac global registration
    global_registration = ransac_global_registration(source, target, fpfh_source, fpfh_target, voxel_size, 
                                                  max_iteration=1000000, confidence=0.9)

    ransac_transformation = global_registration.transformation
    ransac_translation, ransac_euler = get_transform_vectors(ransac_transformation)
    print(f"[GR] translate:\t{ransac_translation}\t(gt: {groundtruth_translation})")
    print(f"[GR] rotate:\t{ransac_euler}\t(gt: {groundtruth_euler})")


    # print("Global registration")
    transformed_source = transform_pointcloud(source, transform=ransac_transformation)
    static_visualizer([transformed_source, target])


    # ########################################
    # # ICP REGISTRATION
    # ########################################

    # # Make sure the normals are correctly estimated
    # source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))


    # # Perform ICP registration
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source, target, icp_threshold, np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))


    # # visualize source and target
    # export_pointcloud(source + target, "export/icp", type="ply")
    # static_visualizer([source, target])

    # icp_transformation = reg_p2l.transformation
    # icp_translation, icp_euler = get_transform_vectors(icp_transformation)
    # print(f"[ICP] translate:\t{icp_translation}")
    # print(f"[ICP] rotate:\t{icp_euler}")


    # # combine ransac global registration and ICP transforms
    # translation, euler = get_transform_vectors(np.dot(ransac_transformation, icp_transformation))
    # print(f"[combined] translate:\t{translation}\t(gt: {groundtruth_translation})")
    # print(f"[combined] rotate:\t{euler}\t(gt: {groundtruth_euler})")
