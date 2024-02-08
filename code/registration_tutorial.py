"""
https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
"""

import open3d as o3d
import numpy as np
import os
import time

from libs.visualization import draw_registration_result
from libs.registration import global_registration, ICP_registration
from libs.pointcloud import preprocess_point_cloud

basedir = "code/experiments/test_data"
path0 = os.path.join(basedir, "cloud_bin_0.pcd")
path1 = os.path.join(basedir, "cloud_bin_1.pcd")
path2 = os.path.join(basedir, "cloud_bin_2.pcd")

# TODO: 0 <> 2 not working with P2L
source = o3d.io.read_point_cloud(path2)
target = o3d.io.read_point_cloud(path0)



voxel_size = 0.05  # means 5cm for this dataset
gr_max_iteration = 1000000
gr_confidence = 0.9

icp_threshold = 0.03 #  voxel_size * 0.4
# p2p_max_iteration = 200


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
transformation = reg_ransac.transformation
draw_registration_result(source, target, transformation)


# ----------------------------------
# Local refinement using ICP


# P2P
p2p_max_iteration = 200
start = time.time()
reg_p2p = ICP_registration(source, target, icp_threshold, 
                              reg_ransac.transformation, use_p2l=False, p2p_max_iteration=p2p_max_iteration)

print(f"P2P ICP took {time.time() - start:.3f} sec.")
# print(reg_p2p)
draw_registration_result(source, target, reg_p2p.transformation)


# P2L
start = time.time()
reg_p2l = ICP_registration(source, target, icp_threshold, 
                              reg_ransac.transformation, use_p2l=True)

print(f"P2L ICP took {time.time() - start:.3f} sec.")
# print(reg_p2l)
draw_registration_result(source, target, reg_p2l.transformation)
