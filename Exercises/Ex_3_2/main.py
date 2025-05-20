import cv2
import numpy as np
import matplotlib.pyplot as plt

# Richardson-Lucy Deconvolution
def richardson_lucy(b, k, iterations=30, init_img=None):
    if init_img is None:
        raise ValueError("Initial image (init_img) must be provided.")

    b = b.astype(np.float32) + 1e-6 # avoid divide-by-zero
    a_est = init_img.astype(np.float32)
    k = k / k.sum() # normalize PSF
    k_mirror = k[::-1, ::-1] # flip kernel for correction step

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        prev = a_est.copy() # prev estimate for convergence check

        # Step 1: blur current estimate with PSF
        conv = cv2.filter2D(a_est, -1, k, borderType=cv2.BORDER_REFLECT)

        # Step 2: avoid division by very small numbers
        conv[conv < 1e-6] = 1e-6

        # Step 3: compute ratio image (B / blurred estimate)
        ratio = b / conv

        # Step 4: convolve ratio with flipped PSF and update estimate
        correction = cv2.filter2D(ratio, -1, k_mirror, borderType=cv2.BORDER_REFLECT)
        a_est *= correction

        # Step 5: convergence check to stop early if the change is small
        delta = np.linalg.norm(a_est - prev)
        if delta < 1e-2:
            print(f"Converged at iteration {i}")
            break

    return np.clip(a_est, 0, 255).astype(np.uint8)


# Generate example PSF kernels
def get_kernels(kernel_size):
    return {
        "mean": np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2),
        "gaussian": cv2.getGaussianKernel(kernel_size, kernel_size / 3) @
                    cv2.getGaussianKernel(kernel_size, kernel_size / 3).T,
        "motion_horizontal": np.ones((1, kernel_size)) / kernel_size,
    }


# Test RLD with various blur kernels, noise levels, and initializations
def test_rld_on_image(img_path):
    kernel_size = 5
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernels = get_kernels(kernel_size)
    custom_a_est = cv2.imread("img/circle.jpg", cv2.IMREAD_GRAYSCALE)
    noise_levels = [0, 5, 10, 20]

    for name, K in kernels.items():
        print(f"\n\nTesting kernel: {name}")
        blurred = cv2.filter2D(original, -1, K) # simulate blur (B = A * K)

        for noise_std in noise_levels:
            print(f"\nNoise level = {noise_std}")
            # add Gaussian noise to blurred image
            noise = np.random.normal(0, noise_std, blurred.shape).astype(np.float32)
            b_noisy = np.clip(blurred + noise, 0, 255).astype(np.uint8)

            for init in ["observed", "gray127", "random", "custom"]:
                print(f"\nUsing init mode: {init}")

                # choose initial guess for A'
                if init == "random":
                    init_img = np.random.rand(*b_noisy.shape) * 255
                elif init == "observed":
                    init_img = b_noisy.copy()
                elif init == "gray127":
                    init_img = np.full_like(b_noisy, 127)
                elif init == "custom":
                    init_img = custom_a_est.copy()
                else:
                    continue

                # Richardson-Lucy Deconvolution
                result = richardson_lucy(b_noisy, K, iterations=100, init_img=init_img)

                # plot results
                fig, axs = plt.subplots(1, 4, figsize=(20, 6))
                axs[0].imshow(original, cmap='gray')
                axs[0].set_title("Original Image A")
                axs[0].axis('off')

                axs[1].imshow(b_noisy, cmap='gray')
                axs[1].set_title(f"Blurred + Noise B\n({name}, σ={noise_std})")
                axs[1].axis('off')

                axs[2].imshow(init_img, cmap='gray')
                axs[2].set_title(f"Initial A′ ({init})")
                axs[2].axis('off')

                axs[3].imshow(result, cmap='gray')
                axs[3].set_title(f"Reconstructed A′")
                axs[3].axis('off')

                plt.suptitle(f"RLD Deconvolution – Kernel: {name} – Init: {init}", fontsize=14)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    test_rld_on_image("img/tripod.png")
