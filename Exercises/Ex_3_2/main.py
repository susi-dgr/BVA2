import cv2
import numpy as np
import matplotlib.pyplot as plt

# Richardson-Lucy Deconvolution Implementation
def richardson_lucy(B, K, iterations=30, init_img=None):
    if init_img is None:
        raise ValueError("Initial image (init_img) must be provided.")

    B = B.astype(np.float32) + 1e-6  # avoid divide by zero
    A_est = init_img.astype(np.float32)
    K = K / K.sum()
    K_mirror = K[::-1, ::-1]

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        conv = cv2.filter2D(A_est, -1, K, borderType=cv2.BORDER_REFLECT)
        ratio = B / (conv + 1e-6)
        correction = cv2.filter2D(ratio, -1, K_mirror, borderType=cv2.BORDER_REFLECT)
        A_est *= correction

    return np.clip(A_est, 0, 255).astype(np.uint8)

# Sample PSF kernels
def get_kernels():
    return {
        "mean": np.ones((5, 5), dtype=np.float32) / 25,
        "gaussian": cv2.getGaussianKernel(9, 2) @ cv2.getGaussianKernel(9, 2).T,
        "motion_horizontal": np.eye(1, 9, dtype=np.float32),
    }

# Test pipeline with display
def test_rld_on_image(img_path):
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernels = get_kernels()
    custom_A_est = cv2.imread("img/circle.jpg", cv2.IMREAD_GRAYSCALE)

    for name, K in kernels.items():
        print(f"\n\nTesting kernel: {name}")
        blurred = cv2.filter2D(original, -5, K)

        # Add Gaussian noise
        noise = np.random.normal(0, 5, blurred.shape).astype(np.float32)
        B_noisy = np.clip(blurred + noise, 0, 255).astype(np.uint8)

        for init in ["blurred", "gray127", "random", "custom"]:
            print(f"\nUsing init mode: {init}")

            # Generate initial image
            if init == "random":
                init_img = np.random.rand(*B_noisy.shape) * 255
            elif init == "gray127":
                init_img = np.full_like(B_noisy, 127)
            elif init == "blurred":
                init_img = B_noisy.copy()
            elif init == "custom":
                init_img = custom_A_est.copy()
            else:
                continue

            # Run Richardson-Lucy with provided init_img
            result = richardson_lucy(B_noisy, K, iterations=40, init_img=init_img)

            # Plot A, B, A′₀, A′
            fig, axs = plt.subplots(1, 4, figsize=(20, 6))
            axs[0].imshow(original, cmap='gray')
            axs[0].set_title("Original Image A")
            axs[0].axis('off')

            axs[1].imshow(B_noisy, cmap='gray')
            axs[1].set_title(f"Blurred + Noise B ({name})")
            axs[1].axis('off')

            axs[2].imshow(init_img, cmap='gray', vmin=0, vmax=255)
            axs[2].set_title(f"Initial A′ ({init})")
            axs[2].axis('off')

            axs[3].imshow(result, cmap='gray')
            axs[3].set_title(f"Reconstructed A′")
            axs[3].axis('off')

            plt.suptitle(f"RLD Deconvolution – Kernel: {name} – Init: {init}", fontsize=14)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Example usage
    test_rld_on_image("img/lena.jpg")
