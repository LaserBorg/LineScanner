# https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

import numpy as np


a = np.array([[5,2,3],[4,5,6],[3,6,4]])

def sort_numpy_by_column(array, column=0):
	return array[array[:,column].argsort()]

a = sort_numpy_by_column(a, column=2)
print(a)
