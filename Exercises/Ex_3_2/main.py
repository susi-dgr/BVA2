import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Richardson-Lucy Deconvolution
def richardson_lucy(b, k, iterations=30, init_img=None):
    if init_img is None:
        raise ValueError("Initial image must be provided.")

    b = b.astype(np.float32) + 1e-6
    a_est = init_img.astype(np.float32)
    k = k / k.sum()
    k_mirror = k[::-1, ::-1]

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        prev = a_est.copy()

        # Step 1: blur current estimate with PSF
        conv = cv2.filter2D(a_est, -1, k, borderType=cv2.BORDER_REFLECT)

        # Step 2: compute ratio image (B / blurred estimate)
        ratio = b / (conv + 1e-6)

        # Step 3: convolve ratio with flipped PSF and update estimate
        correction = cv2.filter2D(ratio, -1, k_mirror, borderType=cv2.BORDER_REFLECT)
        a_est *= correction
        a_est = np.clip(a_est, 1e-6, 255)

        # Step 4: convergence check to stop early if the change is small
        delta = np.linalg.norm(a_est - prev)
        print(f"Delta: {delta:.4f}")
        if delta < 1:
            print(f"Converged at iteration {i+1}")
            return np.clip(a_est, 0, 255).astype(np.uint8), i+1  # return image + iteration

    return np.clip(a_est, 0, 255).astype(np.uint8), iterations  # return image + max iteration if no convergence

# Generate example PSF kernels
def get_kernels(kernel_size, custom_kernel_path=None):
    motion_asym = np.random.rand(1, kernel_size).astype(np.float32)
    motion_asym /= motion_asym.sum()

    kernels = {
        "mean": np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2),
        "gaussian": cv2.getGaussianKernel(kernel_size, kernel_size / 3) @
                    cv2.getGaussianKernel(kernel_size, kernel_size / 3).T,
        "motion_horizontal": np.ones((1, kernel_size * 2), dtype=np.float32) / (kernel_size * 2),
        "motion_asymmetric": motion_asym,
    }

    # load custom kernel if path is given
    if custom_kernel_path is not None:
        custom_kernel = cv2.imread(custom_kernel_path, cv2.IMREAD_GRAYSCALE)
        if custom_kernel is not None:
            kernels["custom"] = custom_kernel.astype(np.float32) / np.sum(custom_kernel)
        else:
            print(f"Warning: Custom kernel at '{custom_kernel_path}' could not be loaded.")

    return kernels

# Test RLD with various blur kernels, noise levels, and initializations
def test_rld_on_image(img_path):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    kernel_size = 10
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernels = get_kernels(kernel_size)

    for kernel in kernels.values():
        # show kernels
        plt.imshow(kernel, cmap='gray')
        plt.show()

    custom_a_est = cv2.imread("img/donald.jpg", cv2.IMREAD_GRAYSCALE)

    # Resize custom init to match the current image
    if custom_a_est.shape != original.shape:
        custom_a_est = cv2.resize(custom_a_est, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)

    noise_levels = [0, 5, 10]

    for name, K in kernels.items():
        print(f"\n\nTesting kernel: {name}")
        blurred = cv2.filter2D(original, -1, K) # simulate blur (B = A * K)

        for noise_std in noise_levels:
            print(f"\nNoise level = {noise_std}")
            # add Gaussian noise to blurred image
            noise = np.random.normal(0, noise_std, blurred.shape).astype(np.float32)
            b_noisy = np.clip(blurred + noise, 0, 255).astype(np.uint8)

            for init in ["random", "observed", "gray127", "custom"]:
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
                result, iteration = richardson_lucy(b_noisy, K, iterations=30, init_img=init_img)

                # plot results
                fig, axs = plt.subplots(1, 4, figsize=(20, 6))
                axs[0].imshow(original, cmap='gray')
                axs[0].set_title("Original Image A")
                axs[0].axis('off')

                axs[1].imshow(b_noisy, cmap='gray')
                axs[1].set_title(f"Blurred + Noise B\n({name}, σ={noise_std})")
                axs[1].axis('off')

                axs[2].imshow(init_img, cmap='gray', vmin=0, vmax=255)
                axs[2].set_title(f"Initial A′ ({init})")
                axs[2].axis('off')

                axs[3].imshow(result, cmap='gray')
                axs[3].set_title(f"Reconstructed A′\n(Iterations: {iteration})")
                axs[3].axis('off')

                plt.suptitle(f"RLD Deconvolution – Kernel: {name} – Initial Guess: {init}", fontsize=14)
                plt.tight_layout()

                fig_name = f"output/RLD_{img_name}_plot_{name}_noise{noise_std}_init{init}_iter{iteration}.png"

                plt.savefig(fig_name)
                print(f"Saved plot to {fig_name}")

                plt.show()


if __name__ == "__main__":
    test_rld_on_image("img/car.jpg")
