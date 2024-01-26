import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def estimate_normals(pcd, radius=0.1, max_nn=30):
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    pcd.estimate_normals(search_param=search_param)
    return pcd


# TODO: exports do not overwrite files -> need to delete if existing
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


# def pointcloud_to_mesh(pcd):
#     pcd.estimate_normals()
#
#     # https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
#
#     mesh = mesh.simplify_quadric_decimation(500000)
#     mesh.remove_degenerate_triangles()
#     mesh.remove_duplicated_triangles()
#     mesh.remove_duplicated_vertices()
#     mesh.remove_non_manifold_edges()
#
#     return mesh
