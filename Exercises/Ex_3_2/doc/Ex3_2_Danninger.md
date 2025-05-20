# Richardson Lucy Deconvolution

## Aim
The goal of this project is to implement an iterative image restoration algorithm using 
the Richardson-Lucy Deconvolution (RLD) technique from scratch in Python, with minimal dependencies.
The algorithm attempts to recover a sharp image A from a blurred and noisy observation B, 
given a known Point Spread Function (PSF) or kernel K.

## Implementation
### Richardson-Lucy Deconvolution
The core iterative algorithm is implemented in the `richardson_lucy(b, k, iterations=30, init_img=None)` function with:
- `b`: blurred (and noisy) input image
- `k`: Point Spread Function (PSF) or kernel used for blurring
- `init_img`: initial guess for the sharp image
- `iterations`: number of iteration steps to perform

There are a few main steps to this algorithm:
#### Preprocessing:
Before entering the iteration loop, some essential preprocessing is done to ensure 
numerical stability and consistent data types:
- The input image `b` is converted to a float32 type for no round-off errors during calculations and 
a small epsilon value is added to avoid division by zero: `b = b.astype(np.float32) + 1e-6`
- The initial guess image `init_img` is also converted to float32 type: `a_est = init_img.astype(np.float32)`
- The kernel `k` is normalized to ensure that the sum of its elements equals 1. 
This prevents unintended changes in overall image brightness during the deconvolution process: `k = k / k.sum()`
- The kernel is flipped horizontally and vertically to create `k_mirror`, which is used in the deconvolution process: `k_mirror = k[::-1, ::-1]`

#### Step1: Convolve current estimate with PSF
The current estimate of the image is blurred using the same PSF K:
```python
conv = cv2.filter2D(a_est, -1, k, borderType=cv2.BORDER_REFLECT)
```
#### Step 2: Avoid division by near-zero values
To avoid instability in the ratio calculation, small values are clipped, `borderType=cv2.BORDER_REFLECT` is used to 
handle edge effects during convolution by mirroring the image at the borders. 
This avoids introducing artificial edges or abrupt transitions that can occur 
when padding with zeros or constant values. 
```python
conv[conv < 1e-6] = 1e-6
```

#### Step 3: Calculate the ratio
The observed image is divided by the blurred estimate to get the ratio:
```python
ratio = b / conv
```

#### Step 4: Convolve the ratio with the flipped PSF and update the estimate
This provides a correction factor and the estimate is updated by element-wise multiplication:
```python
correction = cv2.filter2D(ratio, -1, k_mirror, borderType=cv2.BORDER_REFLECT)
a_est *= correction
```

#### Step 5: Update the estimate
This is an optional step that stops the algorithm early if the estimate converges, 
the previous estimate is stored before and the difference between the current and previous estimates is calculated:
```python
delta = np.linalg.norm(a_est - prev)
if delta < 1e-2:
    print(f"Converged at iteration {i}")
    break
```

### PSF Kernel Definition
With the function `get_kernels(kernel_size)`, the following kernels are generated with `kernel_size` as input:
//TODO how are they calculated?
- **Mean Kernel**: A uniform kernel that averages pixel values in a local neighborhood. 
It divides the sum of all pixel values in the region equally.
- **Gaussian Kernel**: A Gaussian kernel that weights pixel values based on their distance from the center. 
The standard deviation kernel_size / 3 ensures the kernel decays smoothly from center to edge.
- **Motion Kernel**: A kernel that simulates motion blur in a specific direction, in this case horizontal. 
It spreads intensity evenly across a horizontal line of kernel_size pixels.
```python
"mean": np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2),
"gaussian": cv2.getGaussianKernel(kernel_size, kernel_size / 3) @
            cv2.getGaussianKernel(kernel_size, kernel_size / 3).T,
"motion_horizontal": np.ones((1, kernel_size)) / kernel_size,
```

### Noise
For each kernel, a noise is added to the blurred image using different noise levels (standard deviations):
```python
noise_levels = [0, 5, 10, 20]
```

### Initial Guess Image
Four different initial guess images are generated:
- **Random**: A random image generated using `np.random.rand(*b_noisy.shape) * 255`
- **Observed**: The observed blurred image `b_noisy.copy()`
- **Gray127**: A gray image with all pixel values set to 127 `np.full_like(b_noisy, 127)`
- **Custom**: A custom image. This can be any image, that is loaded.

