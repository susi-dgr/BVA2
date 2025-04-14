import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Ultralytics YOLO11n model
model = YOLO("yolo11n.pt")

# image paths
image_dir = "test_images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# results
results_data = []

for img_path in image_paths:
    image = cv2.imread(img_path)

    # YOLO detection
    results = model.predict(img_path, classes=[32])  # 32 is 'sports ball' in COCO
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx_bbox = (x1 + x2) // 2
        cy_bbox = (y1 + y2) // 2

        # crop the detected region
        cropped = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # ---- Segmentation ----
        # color threshold (tennis ball -> yellowish green)
        lower_yellow = np.array([0, 80, 80])
        upper_yellow = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # ---- Ball and Center detection ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)

            # fit a circle to the contour
            (x_circle, y_circle), radius = cv2.minEnclosingCircle(c)
            center_circle = (int(x_circle), int(y_circle))
            radius = int(radius)

            # circular mask
            circle_mask = np.zeros_like(mask)
            cv2.circle(circle_mask, center_circle, radius, 255, -1)

            # updating centers using circle center
            cx_seg = center_circle[0] + x1
            cy_seg = center_circle[1] + y1

            # distance between bbox center and segmentation center
            distance = np.sqrt((cx_bbox - cx_seg) ** 2 + (cy_bbox - cy_seg) ** 2) # euclidean distance
            results_data.append({
                "image": os.path.basename(img_path),
                "bbox_center": (cx_bbox, cy_bbox),
                "seg_center": (cx_seg, cy_seg),
                "distance": distance
            })

            # ---- Visualization ----
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 4, 1)
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.title("Cropped Ball Region")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Segmentation Mask")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(circle_mask, cmap='gray')
            plt.title("Circular Mask")
            plt.axis('off')

            # bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # bbox diagonals
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.line(image, (x1, y2), (x2, y1), (255, 0, 0), 1)

            # segmentation center
            cv2.circle(image, (cx_seg, cy_seg), 1, (0, 0, 255), -1)

            # segmentation circle
            cv2.circle(image, (cx_seg, cy_seg), radius, (0, 255, 255), 1)

            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.title("BBox + Segmentation Circle")
            plt.axis('off')

            plt.suptitle(f"Fitted Circle: {os.path.basename(img_path)}")
            plt.tight_layout()
            plt.show()

    # image with visualization
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected centers for image {os.path.basename(img_path)}")
    plt.show()

# ---- Statistics ----
# some key numbers
distances = [d["distance"] for d in results_data]
print("\n--- Center Analysis ---")
print(f"Number of tennis balls analyzed: {len(distances)}")
print(f"Min distance: {np.min(distances):.2f} pixels")
print(f"Max distance: {np.max(distances):.2f} pixels")
print(f"Mean distance: {np.mean(distances):.2f} pixels")
print(f"Std deviation: {np.std(distances):.2f} pixels")

# histogram of distances
plt.hist(distances, bins=10, color='gray', edgecolor='black')
plt.title("Center Distance Between BBox Center and Segmentation Center")
plt.xlabel("Distance (pixels)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# centers comparison as scatter plot
plt.figure(figsize=(10, 6))
for result in results_data:
    plt.scatter(result["bbox_center"][0], result["bbox_center"][1], color='blue', label='BBox Center')
    plt.scatter(result["seg_center"][0], result["seg_center"][1], color='red', label='Segmentation Center')
plt.title("Center Point Comparison")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()







