# Movement Detection in Videos

## Aim
The aim of this project is to develop an algorithm that detects moving objects in static video scenes. 
The approach involves building a background model from the initial frames and identifying changes 
in the following frames to detect moving objects. Three different background substraction methods are being looked at: 
a custom approach and two other implementations from the OpenCV library. (using `MOG2`and `KNN`)

## Implementation
### Background Model
A good background model is essential for detecting moving objects accurately.
- In a training phase the first `n` frames are captured to build a background model.
- The frames are converted to grayscale with `cvtColor` and to reduce noise a Gaussian blur is applied with `GaussianBlur`.
- After obtaining every frame the median of the pixel values is calculated to get the background model.

### Moving Object Detection
After calculating the background model the video is reset to the first frame to start the detection.
- The frame is again converted to grayscale and blurred with a Gaussian blur.
- The absolute difference between the background model and the current frame is calculated.
- A threshold is applied to the difference image to get a binary image which shows the moving objects.
- The background is updated using `accumulateWeighted` to adapt to changes in the scene. 
This is only done where no moving objects are detected to avoid that objects become part of the background 
when staying in the same place for a long time.
- Morphological operations are applied to the binary image to remove noise and to connect close objects. 
This is done by using `morphologyEx` with the `MORPH_CLOSE` (removes small white spots) and `MORPH_OPEN` (removes small black spots) flags.
- The binary image is used to create a mask which is applied to the original frame to show the moving objects in red.

### Heatmap
To visualize the movement of objects a heatmap is created. The functions `normalize` and `applyColorMap` are used to create a heatmap from each frame's binary image.

### OpenCV Background Subtraction
I used the two OpenCV implementations `MOG2` and `KNN` to compare the results with my custom approach.

- `MOG2` is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It uses a method to model each pixel as a mixture of Gaussians.
- `KNN` is a K-Nearest Neighbors approach to background subtraction. It uses a method to model each pixel as a mixture of K-nearest neighbors.

## Results
### Pedestrian video 
#### Custom Approach
<img src="img/img.png" width="1000">

We can see that the moving object detection works okay. The background model is not perfect and the detection is not 
always accurate. For example a moving object that has a similar color to the background is not always detected.
However, if some people stand still for a while it does not mess up the background model and they are still detected as a moving object.

#### MOG2
<img src="img/img_1.png" width="1000">

The detection in this method is closer to the object, or more "tight-fitting," compared to the custom approach.

#### KNN
<img src="img/img_2.png" width="1000">

The results of this background subtraction seems to be a bit worse than the other two, since rather large parts of the moving objects are not detected.

### Rolling dice video
#### Custom Approach
<img src="img/img_5.png" width="1000">

Here, you can observe that objects with colors similar to the background 
(such as the hand and the wood, as well as the dice numbers and fabric) are not marked as moving objects. 
This issue is less pronounced in the KNN method. 
Additionally, the shadow is significantly more pronounced in this method compared to the others.

#### MOG2
<img src="img/img_3.png" width="1000">

The objects were only shortly detected. Once they stopped moving they were not marked as moving objects anymore. This happened rather fast. 
Sometimes some dices were not detected at all. 

#### KNN 
<img src="img/img_4.png" width="1000">

Here the objects were detected a bit longer than with the MOG2 approach.

### Notes on Training phase 
I found it important to have a fitting training phase without objects, otherwise the background model 
would not be accurate and would for example have some objects in it as can be seen here at the bottom:
<img src="img/img_6.png" width="1000">

or here in the middle:
<img src="img/img_7.png" width="1000">

## Used Libraries
- OpenCV: image processing and background subtraction
- Numpy: mathematical operations on arrays
- enum: enumerations for the background subtraction methods
