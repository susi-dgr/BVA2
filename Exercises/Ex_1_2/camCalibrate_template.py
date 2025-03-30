# -*- coding: utf-8 -*-
"""
#from https://learnopencv.com/camera-calibration-using-opencv/
"""

import cv2
import numpy as np
import os
import glob
import time

CHECKERBOARD_TYPE = "CHECKERBOARD"
CIRCLE_GRID_TYPE = "CIRCLE_GRID"

pattern_type = CIRCLE_GRID_TYPE

# Defining the dimensions of checkerboard
if pattern_type == CHECKERBOARD_TYPE:
    CHECKERBOARD = (6, 9)
else:
    CHECKERBOARD = (4, 11)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
outFilePath = "img\\"
numOfFrames = 10  # frames to record
timeDelayInSecs = 0.8

inVideoPath = 0

capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('unable to open video: ' + args.input)
    exit(0)

frameCount = 0

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Thresholds for better blob detection
blobParams.minThreshold = 5
blobParams.maxThreshold = 250

# Filter by Area (size of blobs)
blobParams.filterByArea = True
blobParams.minArea = 100   # Detect smaller blobs
blobParams.maxArea = 6000  # Prevent filtering out valid large blobs

# Filter by Circularity (ensures round blobs)
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.7

# Filter by Convexity (ensures smooth blobs)
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.9

# Filter by Inertia (prevents elongated detections)
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.4

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

while frameCount < numOfFrames:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCopy = frame.copy()
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    filePathToWrite = outFilePath + 'img' + str(frameCount) + ".png"
    cv2.imwrite(filePathToWrite, frameCopy)
    frameCount = frameCount + 1
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    if pattern_type == CHECKERBOARD_TYPE:
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        ret, corners = cv2.findCirclesGrid(gray, CHECKERBOARD, None,
                                                    cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, blobDetector=blobDetector)

    #    """
    #    If desired number of corner are detected,
    #    we refine the pixel coordinates and display
    #    them on the images of checker board
    #    """

    if ret == True:
        print('pattern found')
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        frameCopy = cv2.drawChessboardCorners(frameCopy, CHECKERBOARD, corners2, ret)
        filePathToWrite = outFilePath + 'imgCB' + str(frameCount) + ".png"
        cv2.imwrite(filePathToWrite, frameCopy)
        cv2.imshow('img', frameCopy)
    else:
        print('pattern NOT found')
    time.sleep(timeDelayInSecs)

cv2.destroyAllWindows()

h, w = frameCopy.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret_RMSerr, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# RMS err should be between 0.1 and 1 pixel
if ret_RMSerr < 2.0:
    print('CAMERA CALIBRATED!!! ERR=' + str(ret_RMSerr))
else:
    print('camera calibration FAILED!! ERR=' + str(ret_RMSerr))

print("Camera matrix : \n")
print(mtx)
print("lens distortion : \n")
print(dist)
print('extrinsic positions for ALL detected shapes')
print("ROTATION rvecs : \n")
print(rvecs)
print("TRANSLATION tvecs : \n")
print(tvecs)

"""
Performing Undistortion 
and drawing Vector field
"""

images = glob.glob(outFilePath + 'img[0-9].png')

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # save img
    cv2.imwrite(fname + '_calibresult.png', dst)

    # Visualization of the distortion by vector field
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)

    step = 10  # step size for the vector field
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Get the original and distorted pixel positions
            original = (x, y)
            distorted = (int(map_x[y, x]), int(map_y[y, x]))

            # Draw the vectors line
            cv2.arrowedLine(img, original, distorted, (0, 0, 255), 1, tipLength=0.4)

    cv2.imwrite(fname + '_calibresult_vectors.png', img)


