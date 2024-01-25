#Based on http://geomalgorithms.com/a05-_intersect-1.html
#https://stackoverflow.com/questions/5666222/3d-line-plane-intersection

import numpy as np
import math

def triangulate(pixel, root, rayPoint, planePoint, planeNormal):
	#Pixel vector relative to image root point
	rayDirection = np.array([pixel[0]+root[0], root[1]-pixel[1], root[2]])
	# print(rayDirection)

	ndotu = planeNormal.dot(rayDirection)

	# check if parallel or in-plane
	almost_zero=1e-6
	if abs(ndotu) < almost_zero:
		print ("[WARNING] no intersection or line is within plane")
		return False
	else:
		w = rayPoint - planePoint
		si = -planeNormal.dot(w) / ndotu
		Psi = w + si * rayDirection + planePoint

		if Psi[2] > 0:
			print ("3D position: ", Psi)
			return Psi
		else:
			print("[WARNING] intersection behind camera ?!")
			return False


pixel = (320, 240)
dims = (640, 480)
c = 10 # cm Camera|Laser
beta_degree = -5. # Laser Angle
fov_degree = 60. # Camera horizontal Field of View


#ONCE
deg2rad = math.pi/180
fov = fov_degree * deg2rad
lens_length = dims[0]/(2*math.tan(fov/2))
# print('lens', lens_length)
root = np.array([-dims[0]/2, dims[1]/2, lens_length])
# print('root', root)

#Laser position
planePoint = np.array([c, 0, 0])
#Camera position
rayPoint = np.array([0, 0, 0])


#EACH FRAME
beta = beta_degree * deg2rad
planeNormal = np.array([-1, 0, math.tan(beta)]) #Laser plane normal vector


#EACH LINE
point = triangulate(pixel, root, rayPoint, planePoint, planeNormal)
