# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from enum import Enum

# Helper function for checkerboard overlay visualization
def checkerboard_overlay(original, undistorted, square_size=20):
    """
    Creates a checkerboard overlay of two images
    """
    h, w = original.shape[:2]
    checkerboard = np.zeros_like(original)

    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            # Doing a floor division to determine if the square is even or odd to alternate between the two images
            if (x // square_size + y // square_size) % 2 == 0:
                # If the square is even, use the original image
                checkerboard[y:y+square_size, x:x+square_size] = original[y:y+square_size, x:x+square_size]
            else:
                # If the square is odd, use the undistorted image
                checkerboard[y:y+square_size, x:x+square_size] = undistorted[y:y+square_size, x:x+square_size]

    return checkerboard


class PatternType(Enum):
    CHECKERBOARD = 1
    CIRCLE_GRID = 2

class VisualizationType(Enum):
    VECTOR_FIELD = 1
    HEATMAP = 2
    CHECKERBOARD_DIFF = 3
    ALL = 4

# Pattern Type
pattern_type = PatternType.CHECKERBOARD

# Visualization Type
visualization_type = VisualizationType.ALL

# Defining the dimensions of checkerboard / circle grid
if pattern_type == PatternType.CHECKERBOARD:
    CHECKERBOARD = (6, 9) # checkerboard
else:
    CHECKERBOARD = (4, 11) # asymmetrical circle grid

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
numOfFrames = 50  # frames to record
timeDelayInSecs = 0.8 # delay between frames

inVideoPath = 0

capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('unable to open video: ' + args.input)
    exit(0)

frameCount = 0

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Filter by Area (size of blobs)
blobParams.filterByArea = True
blobParams.minArea = 100
blobParams.maxArea = 6000

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

# Iterate over each image
while frameCount < numOfFrames:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCopy = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)

    # Save frame
    filePathToWrite = outFilePath + 'img' + str(frameCount + 1) + ".png"
    cv2.imwrite(filePathToWrite, frameCopy)

    frameCount = frameCount + 1

    # Find the chessboard corners or the circles
    if pattern_type == PatternType.CHECKERBOARD:
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
Undistorting the images
and drawing Vector field
"""

h, w = frameCopy.shape[:2]
images = [outFilePath + f'img{i}.png' for i in range(1, 11)]

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # undistort
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # save img
    cv2.imwrite(fname + '_calibresult.png', undistorted)

    # Vector field Visualization
    if visualization_type == VisualizationType.VECTOR_FIELD:
        map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)
        step = 15
        for y in range(0, h, step):
            for x in range(0, w, step):
                original = (x, y)
                distorted = (int(map_x[y, x]), int(map_y[y, x]))
                # Drawing the vector from original to distorted point
                cv2.arrowedLine(img, original, distorted, (0, 0, 255), 1, tipLength=0.2)
        cv2.imwrite(fname + '_vector_field.png', img)

    # Heatmap Visualization
    elif visualization_type == VisualizationType.HEATMAP:
        diff = cv2.absdiff(img, undistorted)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        cv2.imwrite(fname + '_heatmap.png', heatmap)

    # Checkerboard Visualization
    elif visualization_type == VisualizationType.CHECKERBOARD_DIFF:
        checkerboard_result = checkerboard_overlay(img, undistorted)
        cv2.imwrite(fname + '_checkerboard_diff.png', checkerboard_result)

    # All of the above Visualizations
    else:
        diff = cv2.absdiff(img, undistorted)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        cv2.imwrite(fname + '_heatmap.png', heatmap)

        checkerboard_result = checkerboard_overlay(img, undistorted)
        cv2.imwrite(fname + '_checkerboard_diff.png', checkerboard_result)

        map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)
        step = 15
        for y in range(0, h, step):
            for x in range(0, w, step):
                original = (x, y)
                distorted = (int(map_x[y, x]), int(map_y[y, x]))
                cv2.arrowedLine(img, original, distorted, (0, 0, 255), 1, tipLength=0.2)
        cv2.imwrite(fname + '_vector_field.png', img)


