'''
https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array#9667121
https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
https://stackoverflow.com/questions/53466504/finding-singulars-sets-of-local-maxima-minima-in-a-1d-numpy-array-once-again
'''
import numpy as np
from scipy.signal import find_peaks  # 2
# from scipy.signal import argrelmax, argrelmin  # 3
# from scipy.signal import square


# DATA
x = np.linspace(0, 10, 100, endpoint=False)

# 1 interference
data = .2 * np.sin(10 * x) + np.exp(-np.abs(2 - x / 3) ** 2)

# 2 square function
sig = np.sin(2 * np.pi * x)
# data = square(2 * np.pi * x / 2, duty=(sig + 1)/2)


# SMOOTH
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


data = smooth_line(data)


# EXTREMA

# # 1 detects both plateau flanks
# maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1
# minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1

# # 2 detects center of plateau
maxima, _ = find_peaks(data, height=0.2, distance=5)

# # 3 does not detect plateaus!
# maxima = argrelmax(data)
# minima = argrelmin(data)


# VISUALIZE
def plot_row(row, points):
    from matplotlib import pyplot as plt
    length = len(row)
    x = np.linspace(0, length, length, endpoint=False)
    print(x)
    plt.plot(x, row, "bo", ms=3)
    plt.plot(x, row, "b")
    plt.plot(x[points], row[points], "rD", ms=4, label="selected")
    plt.legend()
    plt.show()


plot_row(data, maxima)
