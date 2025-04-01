import cv2
import numpy as np
from enum import Enum

# Background subtraction methods
class BackgroundSubtractionMethod(Enum):
    CUSTOM = 1
    MOG2 = 2
    KNN = 3

background_subtraction_method = BackgroundSubtractionMethod.MOG2

# Video input
# inVideoPath = "videos/vtest.avi"
inVideoPath = "videos/wuerfel.mp4"
capture = cv2.VideoCapture(inVideoPath)
if not capture.isOpened():
    print("Unable to open video file")
    exit(0)

# Parameters
alpha = 0.02  # Background update rate
training_frames = 20  # Number of frames used for background initialization
frame_count = 0
background = None
heatmap = None
heatmap_alpha = 0.01  # Learning rate for heatmap update

# OpenCV background subtractors
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2()
fgbg_knn = cv2.createBackgroundSubtractorKNN()

# Training phase: Collect initial background model if using custom method
training_images = []
if background_subtraction_method == BackgroundSubtractionMethod.CUSTOM:
    while frame_count < training_frames:
        ret, frame = capture.read()
        if not ret:
            break
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)  # Reduce noise
        training_images.append(frame_gray.astype(np.float32))
        frame_count += 1

    # Compute median background model from training frames
    if training_images:
        background = np.median(np.array(training_images), axis=0).astype(np.float32)

    # Reset video to start processing after training phase
    capture.set(cv2.CAP_PROP_POS_FRAMES, training_frames)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    #Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)  # Reduce noise

    # Custom Background Subtraction Method
    if background_subtraction_method == BackgroundSubtractionMethod.CUSTOM:
        # Convert difference to grayscale for better object detection
        background_uint8 = cv2.convertScaleAbs(background)
        diff_frame = cv2.absdiff(background_uint8.astype(np.uint8), frame_gray)

        # Thresholding to create a binary moving object mask
        _, fg_mask = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

        # Update background only where no motion is detected
        motion_free = cv2.bitwise_not(fg_mask)
        background = cv2.accumulateWeighted(frame_gray.astype(np.float32), background, alpha, mask=motion_free)

    # OpenCV Background Subtraction Methods MOG = Mixture of Gaussians
    elif background_subtraction_method == BackgroundSubtractionMethod.MOG2:
        fg_mask = fgbg_mog2.apply(frame)

    # OpenCV Background Subtraction Methods KNN = K-Nearest Neighbors
    elif background_subtraction_method == BackgroundSubtractionMethod.KNN:
        fg_mask = fgbg_knn.apply(frame)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Apply red color to detected objects on the original frame
    frame_with_objects = frame.copy()
    frame_with_objects[fg_mask == 255] = [0, 0, 255]  # Red color overlay

    # Update heatmap
    if heatmap is None:
        heatmap = np.zeros_like(fg_mask, dtype=np.float32)
    heatmap = cv2.addWeighted(heatmap, 1 - heatmap_alpha, fg_mask.astype(np.float32), heatmap_alpha, 0)

    # Normalize and colorize heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Display results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Movement Heatmap", heatmap_color)
    cv2.imshow("Detected Objects", frame_with_objects)

    if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
        break