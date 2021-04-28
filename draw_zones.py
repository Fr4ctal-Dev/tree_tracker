import numpy as np
import cv2
from mog2.tracker import *
from cv2 import *

# wildcard import above does not import "private" variables like __version__
# this makes them available
globals().update(importlib.import_module("cv2.cv2").__dict__)

parameters = {
    "history": 2000000,


    "area": 500,

    "roiCount": 8
    "roiCount": 8,
    "zonePoints": 4,
    "zoneCount:": 4

}

roi_input_mode = True
roi_points = []
zone_input_mode = False
z1_points = []
z2_points = []
z3_points = []
z4_points = []
z5_points = []
zone_shape = []
zones_area = [z1_points,z2_points,z3_points,z4_points,z5_points]
mouseclick = None


def select_roi(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roi_pts, inputMode
    global frame, roi_pts, inputMode, roi_input_mode, zone_input_mode, z1_points, z2_points, z3_points

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if roi_input_mode and event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < parameters["roiCount"]:
        roi_points.append((x, y))
        cv2.circle(original, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", original)

        if len(roi_points) == parameters["roiCount"] - 1:
            inputMode = False

    elif zone_input_mode and event == cv2.EVENT_LBUTTONDOWN and len(z1_points) < parameters["zonePoints"]:
        z1_points.append((x, y))
        cv2.circle(original, (x, y), 4, (255, 0, 0), 5)
        cv2.imshow("frame", original)
    elif zone_input_mode and event == cv2.EVENT_LBUTTONDOWN and len(z2_points) < parameters["zonePoints"]:
        z2_points.append((x, y))
        cv2.circle(original, (x, y), 4, (0, 0, 255), 5)
        cv2.imshow("frame", original)
    elif zone_input_mode and event == cv2.EVENT_LBUTTONDOWN and len(z3_points) < parameters["zonePoints"]:
        z3_points.append((x, y))
        cv2.circle(original, (x, y), 4, (255, 0, 255), 5)
        cv2.imshow("frame", original)
    elif zone_input_mode and event == cv2.EVENT_LBUTTONDOWN and len(z4_points) < parameters["zonePoints"]:
        z4_points.append((x, y))
        cv2.circle(original, (x, y), 4, (255, 255, ), 5)
        cv2.imshow("frame", original)
    elif zone_input_mode and event == cv2.EVENT_LBUTTONDOWN and len(z5_points) < parameters["zonePoints"]:
        z5_points.append((x, y))
        cv2.circle(original, (x, y), 4, (0, 255, 255), 5)
        cv2.imshow("frame", original)

# Create tracker object
tracker = EuclideanDistTracker()

object_detector = cv2.createBackgroundSubtractorMOG2(history=parameters["history"],
                                                     varThreshold=parameters["threshold"])

roi_input_mode = True
roi_points = []

color = (0,255,0)
while True:

    ret, frame = cap.read()
    ret, original = cap.read()
    height, width, _ = frame.shape


    if len(roi_points) < parameters["roiCount"]:
        # indicate that we are in input mode and clone the
        # frame
        inputMode = True
        orig = frame.copy()




        # keep looping until 4 reference ROI points have
        # been selected; press any key to exit ROI selction
        # mode once parameters["roiCount"] points have been selected
        while len(roi_points) < parameters["roiCount"]:
            cv2.imshow("frame", original)
            if inputMode:
                cv2.putText(original, "Mark the Zone of Interest -> Exclude unwanted Areas", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)


            cv2.waitKey(0)

        # determine the top-left and bottom-right points

        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        roi_box = (tl[0], tl[1], br[0], br[1])

    # if the 'q' key is pressed, stop the loop
    elif key == ord("q"):
        break

    roi_area = roi_points.reshape(-1, parameters["roiCount"], 2)




    roi_area = np.array(roi_area)
    frame = cv2.fillPoly(original, roi_area, color=(0,0,0, 50))
    frame = cv2.fillPoly(original, roi_area, color=(0, 0, 0, 50))
    region_of_interest_image = cv2.bitwise_and(original, frame)

    # Process Image
    cv2.normalize(frame, frame, parameters["normaliseAlpha"], parameters["normaliseBeta"], cv2.NORM_MINMAX)
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    frame = cv2.erode(mask, erodeKernel, iterations=parameters["erodeIterations"])
    frame = cv2.dilate(frame, dilateKernel, iterations=parameters["dilateIterations"])
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_contours = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # ZoneSelection
    if not inputMode:
        color = (255, 255, 0)
        cv2.putText(original, "Mark up to 5 Tracking Zones - Enter to proceed", (20, 80),cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)
        inputMode = True
        roi_input_mode = False
        zone_input_mode = True
        # inputMode = False
        cv2.waitKey(0)
    colorcounter = 0
    for zn_points in zones_area:
        # determine the top-left and bottom-right points
        zn_points = np.array(zn_points)
        s = zn_points.sum(axis=1)
        tl = zn_points[np.argmin(s)]
        br = zn_points[np.argmax(s)]
        zone_shape = zn_points.reshape(-1, parameters["zonePoints"], 2)
        zone_shape = np.array(zone_shape)
        for i in range(len(zone_shape)):
            if colorcounter==0:
                color = (255, 0, 0)
            elif colorcounter == 1:
                color = (0, 0, 255)
            elif colorcounter == 2:
                color = (255, 0, 255)
            elif colorcounter == 3:
                color = (255, 255, 0)
            elif colorcounter == 4:
                color = (0, 255, 255)
            cv2.drawContours(original, zone_shape, i, color, 5)

        region_of_interest_image = cv2.bitwise_and(original, frame)
        colorcounter = colorcounter+1

    # Run Detection
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 2000:
            cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(original, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)

    key = cv2.waitKey(50)



    # Video Players
    cv2.imshow("Original", original)
    # cv2.imshow('filledpoly', filled_poly)
