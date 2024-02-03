"""
https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
"""

import open3d as o3d
import numpy as np
import os
import copy
import time


def draw_registration_result(source, target, transformation, draw_uniform=True):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    if draw_uniform:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    
def preprocess_point_cloud(pcd, voxel_size):
    # downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # compute FPFH feature
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, use_fast=False, max_iteration=1000000, confidence=0.9):
    if use_fast:
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, 
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))

    else: # RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
                o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence))
        
    return result

def ICP_registration(source, target, distance_threshold, transform, use_p2l=True, p2p_max_iteration=200):
    if use_p2l: # point-to-plane ICP
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        
        return o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transform, icp_method)
    
    else: # point-to-point ICP
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=p2p_max_iteration)

        return o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transform, icp_method, convergence_criteria)
    
def evaluate_registration(source, target, threshold, transform=None):
    source_temp = copy.deepcopy(source)

    if transform is None:
        transform = np.identity(4)
        source_temp.transform(transform)

    return o3d.pipelines.registration.evaluate_registration(source_temp, target, threshold, transform)


basedir = "code/experiments/test_data"
path0 = os.path.join(basedir, "cloud_bin_0.pcd")
path1 = os.path.join(basedir, "cloud_bin_1.pcd")
path2 = os.path.join(basedir, "cloud_bin_2.pcd")

source = o3d.io.read_point_cloud(path0)
target = o3d.io.read_point_cloud(path1)



voxel_size = 0.05  # means 5cm for this dataset
gr_max_iteration = 1000000
gr_confidence = 0.9

icp_threshold = 0.02 #  voxel_size * 0.4
p2p_max_iteration = 200


# # testing: distort initial pose of source
# trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# source.transform(trans_init)

# downsample, compute normals, and compute FPFH feature
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

draw_registration_result(source, target, np.identity(4))


# ----------------------------------
# RANSAC Global registration

start = time.time()
distance_threshold = voxel_size * 1.5
reg_ransac = global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, 
                                    use_fast=False, max_iteration=gr_max_iteration, confidence=gr_confidence)

print(f"RANSAC global registration took {time.time() - start:.3f} sec.")

# print(reg_ransac)
draw_registration_result(source_down, target_down, reg_ransac.transformation)


# # Fast global registration
# start = time.time()
# distance_threshold = voxel_size * 0.5
# reg_fast = global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, 
#                                   use_fast=True, max_iteration=gr_max_iteration, confidence=gr_confidence)

# print(f"fast global registration took {time.time() - start:.3f} sec.")

# # print(reg_fast)
# draw_registration_result(source_down, target_down, reg_fast.transformation)



# print(evaluate_registration(source, target, icp_threshold, transform=reg_ransac.transformation))
draw_registration_result(source, target, reg_ransac.transformation)


# ----------------------------------
# Local refinement using ICP


# # P2P
# p2p_max_iteration = 200
# start = time.time()
# reg_p2p = ICP_registration(source, target, icp_threshold, 
#                               reg_ransac.transformation, use_p2l=False, p2p_max_iteration=p2p_max_iteration)

# print(f"P2P ICP took {time.time() - start:.3f} sec.")
# # print(reg_p2p)
# draw_registration_result(source, target, reg_p2p.transformation)


# P2L
start = time.time()
reg_p2l = ICP_registration(source, target, icp_threshold, 
                              reg_ransac.transformation, use_p2l=True)

print(f"P2L ICP took {time.time() - start:.3f} sec.")
# print(reg_p2l)
draw_registration_result(source, target, reg_p2l.transformation)
