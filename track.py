# USAGE
# python track.py --video video/sample.mov

# import the necessary packages
import numpy as np
import argparse
import cv2

# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
ZnPts = []
inputMode = False
ROISelected = True
ZoneAmount = 0
ZoneArr = [0,0,0]


def ROIOverlay():
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX
    # inserting text on video
    cv2.putText(frame, 'Mark your Region of Interest', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)


# Fuction getROI Matteo -> Set ROISelected to True


def selectZones(ev1, x, y, flags, param):
    global frame, inputMode, ZoneAmount, ZoneArr

    pts = np.array([0])
    # for ZnPts in ZoneArr:
    if inputMode and ev1 == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        if len(pts) <= 3:
            pts = np.append(pts, (x, y))
            cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
            cv2.imshow("frame", frame)

        if len(pts) == 3:
            # pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 255), 3)
            cv2.imshow("frame", frame)


def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, ZnPts, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(ZnPts) < 4:
        ZnPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def main():
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, ZnPts, Z1Pts, Z2Pts, Z3Pts, Z4Pts, Z5Pts, inputMode, ROISelected, ZoneAmount

    camera = cv2.VideoCapture('sample.mov')

    # setup the mouse callback
    cv2.namedWindow("frame")
    # cv2.setMouseCallback("frame", selectROI)
    cv2.setMouseCallback("frame", selectZones)

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None

    # keep looping over the frames
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # if the see if the ROI has been computed
        # if roiBox is not None:
        # convert the current frame to the HSV color space
        # and perform mean shift
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

        # apply cam shift to the back projection, convert the
        # points to a bounding box, and then draw them

        font = cv2.FONT_HERSHEY_SIMPLEX
        # inserting text on video
        cv2.putText(frame, 'Press I Key -> Select the amount of Zones (max 5 Zones)', (50, 50), font, 1, (0, 255, 255),
                    2, cv2.LINE_4)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(30) & 0xFF

        # handle if the 'i' key is pressed, then go into ROI
        # selection mode
        for ZnPts in ZoneArr:
            if key == ord("i") and len(ZoneArr) < 4:
                # indicate that we are in input mode and clone the
                # frame
                inputMode = True
                orig = frame.copy()

                # keep looping until 4 reference ROI points have
                # been selected; press any key to exit ROI selction
                # mode once 4 points have been selected
                while len(ZnPts) < 4:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(0)

                # determine the top-left and bottom-right points
                ZnPts = np.array(ZnPts)
                s = ZnPts.sum(axis=1)
                tl = ZnPts[np.argmin(s)]
                br = ZnPts[np.argmax(s)]

                # grab the ROI for the bounding box and convert it
                # to the HSV color space
                roi = orig[tl[1]:br[1], tl[0]:br[0]]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

                # compute a HSV histogram for the ROI and store the
                # bounding box
                roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                roiBox = (tl[0], tl[1], br[0], br[1])

            # if the 'q' key is pressed, stop the loop
            elif key == ord("q"):
                break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
