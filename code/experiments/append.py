import open3d as o3d
import numpy as np


# open3D visualisation
vis = o3d.visualization.Visualizer()
vis.create_window()


pcd = o3d.geometry.PointCloud()
new_pcd = o3d.geometry.PointCloud()

points = np.random.rand(100, 3)  # np.array([[0.0, 10.0, 10.0], [2.0, 5.0, 2.0]], dtype=np.float64)
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(points)
vis.add_geometry(pcd)

for i in range(200):
    points = np.random.rand(100, 3)
    colors = np.random.rand(100, 3)

    # append pcd to pcd
    new_pcd.points = o3d.utility.Vector3dVector(points)
    new_pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd += new_pcd

    # # append numpy-array to pcd
    # o3d.utility.Vector3dVector.extend(pcd.points, points)

    # update 3D-view each line
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
