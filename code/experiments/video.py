import cv2

def canny(gray):
	# https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
	# Find Canny edges
	edged = gray.copy()
	edged = cv2.Canny(gray, 30, 200)
	contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	return edged, contours, hierarchy


videopath = "images/laser1a_720.mp4"


camera = cv2.VideoCapture(videopath)

while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	if grabbed is False:
		break
	# cv2.imshow("Frame", frame)

	# Grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	edged, contours, hierarchy = canny(gray)

	cv2.imshow('Canny Edges After Contouring', edged)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
