"""
rotate_bound forked from jrosebr1:imutils (MIT license)
https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
"""

import numpy as np
import cv2


def subtract_images(current_frame, previous_frame, return_RGB=True):
    img = cv2.subtract(current_frame, previous_frame).astype(np.float64)
    average = np.mean(img, axis=2).clip(max=255).astype(np.uint8)

    return cv2.merge([average, average, average]) if return_RGB else average


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the angle to rotate clockwise),
    # then grab the sine and cosine (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
