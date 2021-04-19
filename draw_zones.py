import cv2
import time
import numpy as np

tpPointsChoose = []
drawing = False
tempFlag = False
timesClicked = 0


def draw_ROI(event, x, y, flags, param):
    global point1, tpPointsChoose, pts, drawing, tempFlag, timesClicked

    if event == cv2.EVENT_LBUTTONDOWN and timesClicked <= 4:
        timesClicked = timesClicked + 1
        tempFlag = True
        drawing = False
        point1 = (x, y)
        tpPointsChoose.append((x, y))  # Used to draw points

        if timesClicked == 4:
            drawing = True
            pts = np.array([tpPointsChoose[timesClicked]], np.int32)
            pts1 = tpPointsChoose[1:len(tpPointsChoose[timesClicked-1])]
            print(pts1)


    if event == cv2.EVENT_MBUTTONDOWN:
        tempFlag = False
        drawing = True
        tpPointsChoose = []


cv2.namedWindow('video')
cv2.setMouseCallback('video', draw_ROI)
cap = cv2.VideoCapture('sample.mov')  # File name and format
fps = cap.get(cv2.CAP_PROP_FPS)
size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps: {}\nsize: {}".format(fps, size))
vfps = 0.7 / fps  # Delayed playback, adjusted according to computing power
while (True):
    # capture frame-by-frame
    ret, frame = cap.read()
    # display the resulting frame

    if (tempFlag == True and drawing == False and timesClicked < 4):  # Mouse click
            cv2.circle(frame, point1, 5, (0, 255, 0), 2)
            for i in range(len(tpPointsChoose) - 1):
                cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i+1], (255, 0, 0), 2)

    if (tempFlag == True and drawing == True) and timesClicked == 4:  # Mouse right click
            cv2.polylines(frame, [pts], True, (0, 0, 255), thickness=2)

    if (tempFlag == False and drawing == True):  # Middle mouse button
            for i in range(len(tpPointsChoose) - 1):
                cv2.line(frame, tpPointsChoose[i], tpPointsChoose[i+1], (0, 0, 255), 2)

    time.sleep(vfps)
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
        break
# when everything done , release the capture
cap.release()
cv2.destroyAllWindows()
