import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # .Debug


def load_pointcloud(path, voxel_size=0, verbose=False):
    pcd = o3d.io.read_point_cloud(path)
    size_orig = len(pcd.points)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        size_downsampled = len(pcd.points)
        
        if verbose:
            print("reduced from", size_orig, "to", size_downsampled, "points")
    return pcd

def transform_pointcloud(pcd, transform=None, translate=None, euler_rotate_deg=None, pivot=(0,0,0)):
    if transform is not None:
        pcd.transform(transform)
    
    if translate is not None:
        pcd.translate(translate)

    if euler_rotate_deg is not None:
        euler_rotate_rad = np.deg2rad(euler_rotate_deg)
        rotation_matrix = pcd.get_rotation_matrix_from_xyz(euler_rotate_rad)
        pcd.rotate(rotation_matrix, center=pivot)

    return pcd

def estimate_normals(pcd, radius=5, max_nn=30):
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

def scatterplot(pcd):
    array = np.asarray(pcd.points)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    # matplotlib is Z-up, I am Y-up
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_zlabel("Y")

    x = array[:, 0:1]
    y = array[:, 1:2]
    z = array[:, 2:3]

    axis.scatter(x, z, y, marker=".", s=1)

    limit = max(y)
    axis.set_xlim3d(-limit/2, limit/2)
    axis.set_ylim3d(-limit, 0)
    axis.set_zlim3d(0, limit)

    plt.show()


def init_visualizer(width=800, height=800, left=1000, point_size=1.5, backface=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, left=left)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.light_on = False
    render_option.mesh_show_back_face = backface

    # view_control = vis.get_view_control()
    # view_control.set_zoom(0.2)
    # view_control.set_front((0.0, 0.0, 0.01))
    # view_control.set_lookat((0.0, 0.0, -1.0))
    # view_control.set_up((0.0, 1.0, 0.0))
    return vis

def static_visualizer(object_list, width=800, height=800, left=1000, point_size=1.5):
    vis = init_visualizer(width=width, height=height, left=left, point_size=point_size)

    # Add the geometry to the visualization window
    for object in object_list:
        vis.add_geometry(object)

    vis.run()
    vis.destroy_window()

def update_visualizer(vis, object):
    # Update 3D-view each line
    vis.update_geometry(object)
    vis.poll_events()
    vis.update_renderer()


def meshing_ball_pivoting(pcd):
    '''https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba'''
    pcd = estimate_normals(pcd)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist * 2
    ball_radii = o3d.utility.DoubleVector([radius, radius * 2])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, ball_radii)
    return mesh

def mesh_optimize(mesh, decimate=1000000):
    mesh = mesh.simplify_quadric_decimation(decimate)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh
    
def fpfh_from_pointcloud(pcd, voxel_size=1, max_nn=100):
    ''' Fast Point Feature Histograms (FPFH) descriptor'''
    radius_feature = voxel_size * 5
    params = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn)
    return o3d.pipelines.registration.compute_fpfh_feature(pcd, params)


# GLOBAL REGISTRATION

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



if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("export/laser1a_720.pcd")
    mesh = meshing_ball_pivoting(pcd)
    mesh = mesh_optimize(mesh, decimate=1000000)
    static_visualizer([mesh])
