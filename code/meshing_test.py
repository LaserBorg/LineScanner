"""
http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html

normals: http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
"""

import open3d as o3d
from lib.mesh import mesh_from_alpha_shape, estimate_mesh_normals
from lib.pointcloud import sample_poisson_disk


bunny = o3d.data.BunnyMesh()
mesh  = o3d.io.read_triangle_mesh(bunny.path)
estimate_mesh_normals(mesh)

pcd = sample_poisson_disk(mesh, count=1000)
o3d.visualization.draw_geometries([mesh, pcd])

mesh = mesh_from_alpha_shape(pcd)
estimate_mesh_normals(mesh)
o3d.visualization.draw_geometries([mesh, pcd], mesh_show_back_face=True)
