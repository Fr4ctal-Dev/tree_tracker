import numpy as np
import cv2

cap = cv2.VideoCapture("v1.mp4")

#object detector
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



    cv2.normalize(frame, hsv, -3000, 2000, cv2.NORM_MINMAX)

    # ColorRange
    lower_woodcolor = np.array([50, 0, 0])
    upper_woodcolor = np.array([255, 255, 255])

    # Colormask
    colormask = cv2.inRange(hsv, lower_woodcolor, upper_woodcolor)



    #mask
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 20, 255,cv2.THRESH_BINARY)

    # apply image dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

#Region of Interrest
    ROI = np.array([[(0, 0), (640, 0), (640, 480), (0, 480), (0,350),(200,400),(450,100),(0,100)]], dtype=np.int32)
    region_of_interest = cv2.fillPoly(colormask, ROI, 255)
    region_of_interest_image = cv2.bitwise_and(colormask, region_of_interest)
    

    cv2.imshow('frame',frame)
    cv2.imshow('colormask',colormask)
    cv2.imshow('Mask',region_of_interest_image)


    if cv2.waitKey(15) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()