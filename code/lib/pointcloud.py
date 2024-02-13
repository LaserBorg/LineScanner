import numpy as np
import open3d as o3d
import copy


def set_verbosity():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # .Debug

def sample_poisson_disk(pcd, count=1000000):
    return pcd.sample_points_poisson_disk(count)

def estimate_point_normals(pcd, radius=0.1, max_nn=30):
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    pcd.estimate_normals(search_param=search_param)
    return pcd

def export_pointcloud(pcd, savepath, type="pcd", write_ascii=True, compressed=True):
    if type == "pcd" or type == "ply":
        if write_ascii:
            compressed = False
        o3d.io.write_point_cloud(savepath+"."+type, pcd, write_ascii=write_ascii, compressed=compressed),

    elif type == "csv":
        if not isinstance(pcd, np.ndarray):
            array = np.asarray(pcd.points)
        np.savetxt(savepath+"."+type, array, delimiter=",")

def fpfh_from_pointcloud(pcd, radius=5, max_nn=100):
    ''' Fast Point Feature Histograms (FPFH) descriptor'''
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    return o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

def preprocess_point_cloud(pcd, voxel_size=0):
    # downsample
    if voxel_size > 0:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = copy.deepcopy(pcd)
    
    # estimate normals
    pcd_down = estimate_point_normals(pcd_down, radius=voxel_size*2, max_nn=30)

    # compute FPFH feature
    pcd_fpfh = fpfh_from_pointcloud(pcd_down, radius=voxel_size*5, max_nn=100)
    
    return pcd_down, pcd_fpfh


if __name__ == "__main__":
    from visualization import visualize
    from mesh import mesh_from_ball_pivoting, mesh_optimize

    pcd = o3d.io.read_point_cloud("export/laser1a_720.pcd")
    pcd_down = preprocess_point_cloud(pcd, voxel_size=0)  # if voxel_size > 0: return pcd_down, pcd_fpfh

    mesh = mesh_from_ball_pivoting(pcd)
    mesh = mesh_optimize(mesh, count=1000000)
    visualize([mesh])
