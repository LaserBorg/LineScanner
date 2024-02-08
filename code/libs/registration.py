import open3d as o3d
import numpy as np
import copy


def fpfh_from_pointcloud(pcd, radius=5, max_nn=100):
    ''' Fast Point Feature Histograms (FPFH) descriptor'''
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    return o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

def fast_global_registration(source, target, source_fpfh, target_fpfh, voxel_size=1):
    distance_threshold = voxel_size * 1.5

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance = distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source, target, source_fpfh, target_fpfh, option = option)
    return result

def ransac_global_registration(source, target, source_fpfh, target_fpfh, voxel_size=1, max_iteration=1000000, confidence=0.9, ransac_n=3, similarity_threshold=0.9):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence))
    return result

# TODO: replace fast_global_registration and ransac_global_registration with global_registration

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

def ICP_registration(source, target, distance_threshold, transformation, use_p2l=True, p2p_max_iteration=200):
    if use_p2l: # point-to-plane ICP
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        
        return o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transformation, icp_method)
    
    else: # point-to-point ICP
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=p2p_max_iteration)

        return o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transformation, icp_method, convergence_criteria)

def evaluate_registration(source, target, threshold, transformation=None):
    source_temp = copy.deepcopy(source)

    if transformation is None:
        transformation = np.identity(4)
        source_temp.transform(transformation)

    return o3d.pipelines.registration.evaluate_registration(source_temp, target, threshold, transformation)
