import cv2
import numpy as np
import matplotlib.pyplot as plt

inPath = "color_monkey.jpg"
outPath = "grey_monkey.jpg"

img = cv2.imread(inPath)
print("shape " + str(img.shape)) # img (256, 256, 1) != (256, 256)

# convert to grey
grayImg = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) # attention BGR and not RGB
print("shape " + str(grayImg.shape))

minVal = np.min(grayImg)
maxVal = np.max(grayImg)
avgVal = np.min(grayImg)

print("min " + str(minVal) + ",max " + str(maxVal) + ",avg " + str(avgVal))

# view gray image
plot = plt.imshow(grayImg)
plt.show()

plt.imshow(grayImg, cmap= "hot")
plt.show()

# finally write to file system
cv2.imwrite(outPath, grayImg)

# apply low-pass filter for smoothing, radius = 1 ==> 3x3
kernel = np.ones((3,3), np.float32) / 9.0
filtered = cv2.filter2D(grayImg, -1, kernel)
plt.imshow(filtered)
plt.show()

# or apply gaussian blur
filtered2 = cv2.GaussianBlur(grayImg, (11, 11), 0)
plt.imshow(filtered2)
plt.show()

# image arithmetics
avgImg = (grayImg + filtered) / 2.0
edgeImg = np.abs(grayImg - filtered2) # high-pass from low-pass
plt.imshow(edgeImg, cmap="gray")
plt.show()

# deep copy
resImg = grayImg.copy()
resImg[:][10:40] = 255
print("white rect")
plt.imshow(resImg, cmap="gray")
plt.show()

# segment via logic operations
subImageRegion = (grayImg < 60) & (grayImg < 160)
resImg2 = grayImg.copy()
resImg2[subImageRegion] = 255
plt.imshow(resImg2, cmap="gray")
plt.show()

# transforms
width = grayImg.shape[1] # attention!!! breite und höhe vertauscht
height = grayImg.shape[0]
rotDegree = 22.45
rotM = cv2.getRotationMatrix2D((width / 2, height / 2), rotDegree, 1)
rotatedImage = cv2.warpAffine(grayImg, rotM, (width, height))
plt.imshow(rotatedImage, cmap="gray")
plt.show()

# noise
noise = np.random.normal(0, 1, grayImg.shape)
noisyImg = np.abs(grayImg + 40.0 * noise)
plt.imshow(noisyImg, cmap="gray")
plt.show()

# resample
resizedImg = cv2.resize(grayImg, (203, 156), interpolation=cv2.INTER_LINEAR)
plt.imshow(resizedImg, cmap="gray")
plt.show()
