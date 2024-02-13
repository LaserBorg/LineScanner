"""
http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html
http://www.open3d.org/docs/latest/tutorial/Basic/transformation.html

https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
"""

import open3d as o3d
import os
import time

from lib.transformation import get_transform_vectors, transform
from lib.pointcloud import set_verbosity, preprocess_point_cloud, export_pointcloud
from lib.registration import global_registration, ICP_registration
from lib.visualization import visualize  # visualize_simple


# # LINESCANNER GROUND-TRUTH
# groundtruth_translation = (50, 0, 100)
# groundtruth_euler       = (0.0, 20.0, 0)
# # groundtruth_source    = transform(source, translate=translate, euler_rotate_deg=rotate)

set_verbosity()

voxel_size = 0.05  # meter units                    # 3 # cm units
gr_max_iteration = 1000000
gr_confidence = 0.9

icp_threshold = voxel_size * 0.4
p2p_max_iteration = 200

verbose = False

basedir = "code/experiments/test_data"              # "export"
path0 = os.path.join(basedir, "cloud_bin_0.pcd")    # "laser1a_720.pcd"
path1 = os.path.join(basedir, "cloud_bin_1.pcd")    # "laser1b_720.pcd"
path2 = os.path.join(basedir, "cloud_bin_2.pcd")

# TODO: 0 <> 2 not working with P2L
source = o3d.io.read_point_cloud(path2)
target = o3d.io.read_point_cloud(path0)

# downsample, compute normals, and compute FPFH feature
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

visualize([source, target], uniform_colors=True)
visualize([source_down, target_down], uniform_colors=True)
# visualize_simple(source, target, np.identity(4))
# visualize_simple(source_down, target_down, np.identity(4))



########################################
# GLOBAL REGISTRATION 
########################################
# FAST

# start = time.time()
# distance_threshold = voxel_size * 0.5
# reg_fast = global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, 
#                                   use_fast=True, max_iteration=gr_max_iteration, confidence=gr_confidence)

# print(f"FAST global registration took {time.time() - start:.3f} sec.")
# # print(reg_fast)

# visualize([source_down, target_down], transformation=reg_fast.transformation, uniform_colors=True)
# # visualize_simple(source_down, target_down, reg_fast.transformation)


# # print(evaluate_registration(source, target, icp_threshold, transform=reg_ransac.transformation))

# visualize([source, target], transformation=reg_fast.transformation, uniform_colors=True)
# # visualize_simple(source, target, reg_fast.transformation)


# ########################################
# RANSAC

start = time.time()
distance_threshold = voxel_size * 1.5
reg_ransac = global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, 
                                    use_fast=False, max_iteration=gr_max_iteration, confidence=gr_confidence)

print(f"\nRANSAC global registration took {time.time() - start:.3f} sec.")
# print(reg_ransac)

ransac_translation, ransac_euler = get_transform_vectors(reg_ransac.transformation)
print(f"[RANSAC] translate:\t{ransac_translation})")
print(f"[RANSAC] rotate:\t{ransac_euler})")

visualize([source, target], transformation=reg_ransac.transformation, uniform_colors=True)
# visualize_simple(source_down, target_down, reg_ransac.transformation)



########################################
# ICP REGISTRATION
########################################
# # P2P

# start = time.time()
# reg_p2p = ICP_registration(source, target, icp_threshold, 
#                               reg_ransac.transformation, use_p2l=False, p2p_max_iteration=p2p_max_iteration)

# print(f"P2P ICP took {time.time() - start:.3f} sec.")
# # print(reg_p2p)

# visualize([source, target], transformation=reg_p2p.transformation, uniform_colors=True)
# # visualize_simple(source, target, reg_p2p.transformation)


# ########################################
# P2L

start = time.time()
reg_p2l = ICP_registration(source, target, icp_threshold, 
                              reg_ransac.transformation, use_p2l=True)

print(f"\nP2L ICP took {time.time() - start:.3f} sec.")
# print(reg_p2l)

icp_translation, icp_euler = get_transform_vectors(reg_p2l.transformation)
print(f"[P2L ICP] translate:\t{icp_translation}")
print(f"[P2L ICP] rotate:\t{icp_euler}")

visualize([source, target], transformation=reg_p2l.transformation, uniform_colors=True)
# visualize_simple(source, target, reg_p2l.transformation)



########################################
# EXPORT
########################################

export_pointcloud(source + target, "export/icp", type="ply")
