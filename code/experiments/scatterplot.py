'''
tutorial:
https://realpython.com/python-opencv-color-spaces/

original bei matlab
https://www.mathworks.com/help/images/image-segmentation-using-the-color-thesholder-app.html
'''

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt





points = np.zeros(shape=(10, 4))


x = points[:, 1:2]
y = points[:, 2:3]
z = points[:, 3:4]

# 3D scatterplot
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(x, y, z, marker=".")
axis.set_xlabel("X")
axis.set_ylabel("Y")
axis.set_zlabel("Z")
plt.show()