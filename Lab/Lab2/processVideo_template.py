# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:05:18 2020

@author: P21702
"""

# adapted from https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2
import numpy as np
from args import args

inVideoPath = "vtest.avi"

capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened:
    print('unable to open: ' + args.input)
    exit(0)

frameCount = 0
delayInMS = 500

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frameCopy = frame.copy()

    # allocate in case of frame #0
    if frameCount == 0:
        cumulatedFrame = np.zeros(frameCopy.shape)
        cumulatedFrame = cumulatedFrame + frameCopy
        frameCount += 1
    else:
        cumulatedFrame = cumulatedFrame + frameCopy
        frameCount += 1

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(frameCount), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)

    # calculate average frame
    avgFrame = cumulatedFrame / (frameCount, frameCount, frameCount)
    maxVal = np.max(cumulatedFrame)
    avgVal = np.average(cumulatedFrame)
    print("iter " + str(frameCount) + " max= " + str(maxVal) + " avg= " + str(avgVal))
    avgFrame = avgFrame.astype('uint8')
    cv2.imshow('AvgFrame', avgFrame)

    diffFrame = (avgFrame - frameCopy).astype('int8')
    diffFrame = np.abs(diffFrame)
    diffFrame = diffFrame.astype('uint8')
    cv2.imshow('DiffFrame', diffFrame)
    blueImg = diffFrame[:, :, 0]
    greenImg = diffFrame[:, :, 1]
    redImg = diffFrame[:, :, 2]
    threshold = 30
    segmentedImgIdx = np.where((blueImg > threshold) | (greenImg > threshold) | (redImg > threshold))

    binaryResImg = frameCopy.copy()
    binaryResImg[segmentedImgIdx] = 255
    cv2.imshow('Binary Frame', binaryResImg)

    keyboard = cv2.waitKey(delayInMS)
