
# https://moonbooks.org/Articles/How-to-remove-array-rows-that-contain-only-0-in-python/

import numpy as np
im = np.array([ [0,0,0,0,0,0],
   [0,0,1,1,1,0],
   [0,0,0,0,0,0],
   [0,0,1,0,1,0],
   [0,0,0,0,0,0]])

data = im[~np.all(im == 0, axis=1)]

print(data)


