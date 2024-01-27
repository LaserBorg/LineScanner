import numpy as np
import cv2
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def find_laser(img, channel=2, threshold=180, preview_on_black=False, texture=None, desaturate=False):

    def find_line_maxima(row, threshold=180, distance=5, multi=False):
        if multi is False:
            max_indices = np.nonzero(row > threshold)
            if len(max_indices[0]) == 0:
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
                return max_index

        # find multiple Maxima
        else:
            row = smooth_line(row, blur=3)
            maxima, _ = find_peaks(row, height=threshold, distance=distance)
            if len(maxima) == 0:
                return 0
            else:
                return maxima
    
    def smooth_line(array, blur=5, window='hamming'):
            """
            https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
            'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            """
            s = np.r_[array[blur - 1:0:-1], array, array[-2:-blur - 1:-1]]
            if window == 'flat':
                w = np.ones(blur, 'd')
            else:
                w = eval('np.' + window + '(blur)')
            array = np.convolve(w / w.sum(), s, mode='valid')
            array = array[blur - 1:]
            return array

    def draw_point(image, pos, color=(255, 0, 0)):
        cv2.line(image, (int(pos[0]), int(pos[1])), (int(pos[0]+2), int(pos[1])), color, 1)
        return image

    def plot_row(row, points):
        length = len(row)
        x = np.linspace(0, length, length, endpoint=False)
        plt.plot(x, row, "bo", ms=3)
        plt.plot(x, row, "b")
        plt.plot(x[points], row[points], "rD", ms=4, label="selected")
        plt.legend()
        plt.show()

    # extract single color channel for maxima search
    img_channel = img[:, :, channel]

    # if texture is not numpy, use image
    if not isinstance(texture, np.ndarray):
        texture = img
    # if texture is grayscale, convert to RGB
    elif texture.shape[2] != 3:
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    if desaturate:
        texture = cv2.cvtColor(cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

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
        x = find_line_maxima(row, threshold=threshold)

        # TODO: now there could be multiple maxima
        if isinstance(x, np.ndarray):
            x = x[0]

        if x < 0.5:  # if nothing found, skip line
            continue

        # plot_row(row, x)
        
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
