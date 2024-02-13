import open3d as o3d
import numpy as np
import copy


def global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold, use_fast=False, max_iteration=4000000, confidence=0.9):
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
