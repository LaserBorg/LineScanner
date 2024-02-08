import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D


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
