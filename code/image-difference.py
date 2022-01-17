# https://stackoverflow.com/questions/21425992/how-to-subtract-two-images-using-python-opencv2-to-get-the-foreground-object
# https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html

import cv2
import numpy as np

video_path = "../images/laser1a.mp4"
cap = cv2.VideoCapture(video_path)

old_frame = np.zeros((960, 540, 3), np.uint8)

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_LINEAR)

        img = cv2.subtract(frame, old_frame)
        B, G, R = cv2.split(img.astype(np.float64))

        average = (R + B + G) / 2
        average = average.clip(max=255).astype(np.uint8)

        img = cv2.merge([average, average, average])
        cv2.imshow('difference', img)

        old_frame = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
