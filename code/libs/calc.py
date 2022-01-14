import numpy as np


def triangulate(pixel, topleft_corner, camera_pos, laser_pos, plane_normal):
    # Pixel vector relative to image topleft_corner point
    rayDirection = np.array([pixel[0] + topleft_corner[0], topleft_corner[1] - pixel[1], topleft_corner[2]])

    dotProduct = plane_normal.dot(rayDirection)

    # check if parallel or in-plane
    almost_zero = 1e-6
    if abs(dotProduct) < almost_zero:
        print("[WARNING] no intersection at line", pixel[1])
        return np.array([0, 0, 0])
    else:
        w = camera_pos - laser_pos
        si = -plane_normal.dot(w) / dotProduct
        intersection = w + si * rayDirection + laser_pos

        if intersection[2] > 0:
            return intersection
        # print("[WARNING] intersection behind camera")
        return np.array([0, 0, 0])


def sort_numpy_by_column(array, column=0):
    return array[array[:, column].argsort()]
