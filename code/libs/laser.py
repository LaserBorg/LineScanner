import numpy as np
import cv2


def find_laser(img, channel=2, threshold=180, preview_on_black=False, texture=None):
    # extract single color channel
    img_channel = img[:, :, channel]

    # if no texture given, use desaturated image
    if not isinstance(texture, np.ndarray):
        texture = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    if preview_on_black:
        preview_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    else:
        preview_img = cv2.cvtColor(img_channel, cv2.COLOR_GRAY2BGR)

    # # initialize array of 2D (+ later 3D) coordinates + RGB values for this frame
    pointlist = np.zeros(shape=(img_channel.shape[0], 8))

    # vertical laser line -> work through all rows and save x and RGB
    for y in range(img.shape[0]):
        # crop to current row
        row = img_channel[y:y + 1, :][0]
        # search for brightness maximum, else return -1
        x = find_line_maximum(row, threshold=threshold)
        if x < 0.5:  # if nothing found, skip line
            continue
        
        # screenspace coordinates (2D) at [0:2]
        pointlist[y][0] = x
        pointlist[y][1] = y
        # RGB values at [5:8] (reversed opencv order)
        pointlist[y][5] = texture[int(y), int(x)][2]
        pointlist[y][6] = texture[int(y), int(x)][1]
        pointlist[y][7] = texture[int(y), int(x)][0]
        
        # draw current point into preview image
        preview_img = draw_point(preview_img, (x, y))
    return pointlist, preview_img


def find_line_maximum(row, threshold=180, verbose=False):
    max_indices = np.nonzero(row > threshold)

    if len(max_indices[0]) == 0:
        if verbose:
            print(f"row {row}: < threshold")
        return 0

    else:
        # initialize array for pixel intensities
        intensities = np.zeros(len(max_indices[0]))

        # once for every element in max_indices
        for n in range(len(max_indices[0])):
            intensity = row[max_indices[0][n]]
            intensities[n] = intensity

        # WEIGHTED AVERAGE
        weights = intensities/255
        max_index = np.sum(max_indices * weights) / np.sum(weights)

        if verbose:
            print(f"max_index: {max_index} (value: {row[max_index]})")
        return max_index


def draw_point(image, pos, color=(255, 0, 0)):
    cv2.line(image, (int(pos[0]), int(pos[1])), (int(pos[0]+2), int(pos[1])), color, 1)
    return image
