# https://stackoverflow.com/questions/30057046/weighted-mean-in-numpy-python#30057626

import numpy as np

values =      np.array([1,2,3,4,5, 417, 418, 419, 420, 421, 422, 423, 424])
intensities = np.array([1,1,1,1,1,   1,   1,  20,  50,  60,  80,  90, 255])

# normalized brightness as weights
weights = intensities/255
print("weights", weights)

weighted_average = np.sum(values * weights) / np.sum(weights)
print("weighted average", weighted_average)

average = np.sum(values) / len(values)
print("average", average)