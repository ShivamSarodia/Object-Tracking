import cv2 as cv
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == 1:
        print("Callback!")

cv.namedWindow("Display")
cv.setMouseCallback("Display", mouse_callback)

while True:
    img = np.zeros((512,512,3), np.uint8)
    cv.imshow("Display", img)

    if cv.waitKey(1) == ord("q"):
        break
