import numpy as np
import cv2
from mog2.tracker import *

parameters = {
    "history" : 2000000,
    "threshold": 18,
    "erodeKernel" : 4,
    "dilateKernel": 2,

    "normaliseAlpha": -500,
    "normaliseBeta": 500,

    "dilateIterations": 1,
    "erodeIterations" : 1,

    "area": 500

}

roi_input_mode = True
roi_points = []



# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("..\\stable1.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=parameters["history"], varThreshold=parameters["threshold"])

while True:
    ret, frame = cap.read()
    ret, original = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame

    # 1. Object Detection

    # Region of Interrest
    ROI = np.array([[(0, 0), (640, 0), (640, 480), (0, 480), (0, 350), (200, 400), (450, 100), (0, 100)]],
                   dtype=np.int32)
    region_of_interest = cv2.fillPoly(frame, ROI, 255)
    region_of_interest_image = cv2.bitwise_and(frame, region_of_interest)

    erodeKernel = np.ones((parameters["erodeKernel"], parameters["erodeKernel"]), np.uint8)
    dilateKernel = np.ones((parameters["dilateKernel"], parameters["dilateKernel"]), np.uint8)

    cv2.normalize(region_of_interest, roi, parameters["normaliseAlpha"], parameters["normaliseBeta"], cv2.NORM_MINMAX)
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(mask, erodeKernel, iterations=parameters["erodeIterations"])
    dilated = cv2.dilate(eroded, dilateKernel, iterations=parameters["dilateIterations"])
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_contours = cv2.drawContours(dilated, contours, -1, (0,255,0), 3)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 2000:
            cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(original, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Original", original)
    cv2.imshow('eroded', eroded)
    cv2.imshow('roiimg', region_of_interest_image)

    key = cv2.waitKey(50)
    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()