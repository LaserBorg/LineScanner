import numpy as np
import copy
from scipy.spatial.transform import Rotation as R


def get_transform_vectors(transform_M):
    # Extract translation (top-right 3x1 sub-matrix)
    translation = transform_M[:3, 3]

    # Extract rotation (top-left 3x3 sub-matrix), make a copy to avoid read only error
    rotation_M = np.array(transform_M[:3, :3])
    # Convert rotation matrix to Euler angles
    r = R.from_matrix(rotation_M)
    euler_angles = r.as_euler('xyz', degrees=True)

    return translation, euler_angles

def transform(pcd, transformation=None, translate=None, euler_rotate_deg=None, pivot=(0,0,0)):
    pcd_temp = copy.deepcopy(pcd)
    
    if transformation is not None:
        pcd_temp.transform(transformation)
    
    if translate is not None:
        pcd_temp.translate(translate)

    if euler_rotate_deg is not None:
        euler_rotate_rad = np.deg2rad(euler_rotate_deg)
        rotation_matrix = pcd_temp.get_rotation_matrix_from_xyz(euler_rotate_rad)
        pcd_temp.rotate(rotation_matrix, center=pivot)

    return pcd_temp
