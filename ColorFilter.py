import cv2
import numpy as np

cap = cv2.VideoCapture("v1.mp4")

while (1):
    _, frame = cap.read()
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Threshold of blue in HSV space
    lower_woodcolor = np.array([76, 54, 18])
    upper_woodcolor = np.array([255, 255, 239])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_woodcolor, upper_woodcolor)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    cv2.waitKey(0)

cv2.destroyAllWindows()
cap.release()
