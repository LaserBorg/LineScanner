"""
https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
"""

import open3d as o3d
import numpy as np
import copy


def visualize_simple(mesh1, mesh2, transformation, uniform_colors=True):
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

def estimate_point_normals(pcd, radius=0.1, max_nn=30):
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    pcd.estimate_normals(search_param=search_param)
    return pcd

def fpfh_from_pointcloud(pcd, radius=0.25, max_nn=100):
    ''' Fast Point Feature Histograms (FPFH) descriptor'''
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    return o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

def preprocess_point_cloud(pcd, voxel_size):
    # downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # estimate normals
    radius_normal = voxel_size * 2
    pcd_down = estimate_point_normals(pcd_down, radius=radius_normal, max_nn=30)
    
    # compute FPFH feature
    radius_feature = voxel_size * 5
    pcd_fpfh = fpfh_from_pointcloud(pcd_down, radius=radius_feature, max_nn=100)
    
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    source = o3d.io.read_point_cloud("code/experiments/test_data/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("code/experiments/test_data/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    visualize_simple(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
                o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.9))

    return result


voxel_size = 0.05  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
visualize_simple(source_down, target_down, result_ransac.transformation)
