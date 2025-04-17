import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib import animation
from numba import njit, prange
from scipy.stats import gaussian_kde

def ColourDist(refColor, currColor):
    """
    Calculates the Euclidean distance between two points in 3D RGB space.
    refColor and currColor are length-3 arrays [r, g, b].
    """
    return np.sqrt(np.sum((refColor - currColor) ** 2))


def GaussianWeight(dist, bandwidth):
    """
    Calculates the Gaussian kernel weight given the distance and bandwidth.
    w = exp( -(dist^2) / (2 * bandwidth^2) )
    """
    return np.exp(- (dist ** 2) / (2 * (bandwidth ** 2)))


def create_visualization_frames(shifted, shape, iteration, bandwidth):
    """
    Create 2D and 3D visualization frames for the current iteration.

    :param shifted: (N, 3) ndarray of pixel colors (float)
    :param shape: tuple (h, w) representing the original image dimensions.
    :param iteration: current iteration (integer) to include in the title.
    :param bandwidth: float, the bandwidth used.
    :return: tuple (frame_2d, frame_3d) as RGB images.
    """
    h, w = shape

    # --- 2D Visualization (reconstructed image) ---
    image_2d = shifted.reshape(h, w, 3)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(np.clip(image_2d, 0, 255).astype(np.uint8))
    ax.set_title(f'Iteration {iteration + 1} - 2D (bandwidth: {bandwidth})', fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame_2d = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    frame_2d = cv2.cvtColor(frame_2d, cv2.COLOR_BGR2RGB)
    plt.close(fig)

    # --- 3D Visualization (scatter plot in RGB space) ---
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shifted[:, 0], shifted[:, 1], shifted[:, 2], c=shifted / 255.0, s=2)
    ax.set_title(f'Iteration {iteration + 1} - 3D (bandwidth: {bandwidth})', fontsize=12)

    # Set axis limits
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    # Remove tick marks but keep axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('Red', fontsize=10)
    ax.set_ylabel('Green', fontsize=10)
    ax.set_zlabel('Blue', fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame_3d = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    frame_3d = cv2.cvtColor(frame_3d, cv2.COLOR_BGR2RGB)
    plt.close(fig)

    return frame_2d, frame_3d


@njit(parallel=True)
def mean_shift_iteration(shifted, bandwidth):
    """
    Parallelized single Mean Shift iteration.

    :param shifted: (N, 3) array of pixel values (float)
    :param bandwidth: scalar bandwidth value.
    :return: new_shifted: updated pixel values after one iteration.
    """
    N = shifted.shape[0]
    new_shifted = np.empty_like(shifted)
    for i in prange(N):
        num0 = 0.0
        num1 = 0.0
        num2 = 0.0
        denom = 0.0
        for j in range(N):
            diff0 = shifted[j, 0] - shifted[i, 0]
            diff1 = shifted[j, 1] - shifted[i, 1]
            diff2 = shifted[j, 2] - shifted[i, 2]
            dist_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
            weight = np.exp(-dist_sq / (2.0 * bandwidth * bandwidth))
            num0 += shifted[j, 0] * weight
            num1 += shifted[j, 1] * weight
            num2 += shifted[j, 2] * weight
            denom += weight
        new_shifted[i, 0] = num0 / denom
        new_shifted[i, 1] = num1 / denom
        new_shifted[i, 2] = num2 / denom
    return new_shifted


@njit
def compute_max_shift(old, new):
    """
    Computes the maximum Euclidean difference between two arrays of pixel values.

    :param old: (N, 3) array from previous iteration.
    :param new: (N, 3) array with updated pixel values.
    :return: maximum shift distance (float)
    """
    max_shift = 0.0
    N = old.shape[0]
    for i in range(N):
        diff0 = new[i, 0] - old[i, 0]
        diff1 = new[i, 1] - old[i, 1]
        diff2 = new[i, 2] - old[i, 2]
        shift = (diff0 * diff0 + diff1 * diff1 + diff2 * diff2) ** 0.5
        if shift > max_shift:
            max_shift = shift
    return max_shift


def MeanShiftColorPixel(inPixels, bandwidth, max_iter=40, epsilon=1e-3, shape=None, record_frames=False):
    """
    Perform Mean Shift in RGB space on all pixels using a parallelized inner loop.

    :param inPixels: (N, 3) ndarray of pixels (each row is [r, g, b]).
    :param bandwidth: float, bandwidth (sigma) for the Gaussian kernel.
    :param max_iter: int, maximum number of iterations.
    :param epsilon: float, convergence threshold.
    :param shape: tuple (h, w) representing the original image dimensions.
    :param record_frames: bool, if True then record intermediate 2D and 3D visualizations.
    :return: If record_frames is False: (N, 3) ndarray of converged pixel colors.
             If record_frames is True: tuple (shifted, frames_2d, frames_3d).
    """
    shifted = inPixels.copy().astype(np.float64)
    frames_2d = []
    frames_3d = []

    for iteration in range(max_iter):
        new_shifted = mean_shift_iteration(shifted, bandwidth)
        max_shift_distance = compute_max_shift(shifted, new_shifted)
        print(f"Iteration {iteration + 1}, max shift: {max_shift_distance:.4f}")

        shifted = new_shifted
        if record_frames and shape is not None:
            frame2d, frame3d = create_visualization_frames(shifted, shape, iteration, bandwidth)
            frames_2d.append(frame2d)
            frames_3d.append(frame3d)

        if max_shift_distance < epsilon:
            print(f"Converged at iteration {iteration + 1}")
            break

    if record_frames:
        return shifted, frames_2d, frames_3d
    else:
        return shifted


def assign_clusters(shifted, threshold=5.0):
    """
    Group the converged colors into clusters.

    Two colors are considered in the same cluster if their Euclidean distance
    is less than the specified threshold.

    :param shifted: (N, 3) ndarray of converged pixel colors.
    :param threshold: float, distance threshold to merge clusters.
    :return: tuple (labels, cluster_centers)
      - labels: (N,) array of integer cluster labels for each pixel.
      - cluster_centers: (K, 3) ndarray of K cluster center RGB values.
    """
    clusters = []
    labels = np.full(len(shifted), -1, dtype=int)
    for i, color in enumerate(shifted):
        found_cluster = False
        for j, center in enumerate(clusters):
            if ColourDist(color, center) < threshold:
                labels[i] = j
                found_cluster = True
                break
        if not found_cluster:
            clusters.append(color)
            labels[i] = len(clusters) - 1
    return labels, np.array(clusters)

from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def visualize_density(shifted, cluster_centers, bandwidth, save_path=None):
    """
    Plot only color‐coded towers whose heights = local density,
    and whose colors = cluster RGB.

    :param shifted: (N,3) ndarray of final pixel colors in RGB.
    :param cluster_centers: (K,3) ndarray of cluster center colors [R,G,B].
    :param bandwidth: float, bandwidth used in Mean Shift (for title).
    :param save_path: optional path to save the figure.
    """
    # Build a KDE on R,G coords
    RG = shifted[:, :2].T
    kde = gaussian_kde(RG, bw_method=bandwidth/np.std(RG, axis=1).mean())

    # Compute density at each cluster center
    dens = kde(cluster_centers[:, :2].T)

    # Plot bars
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # Compute a small bar footprint so bars don't overlap too much
    # Spread them slightly based on data range
    dr = dg = 255.0 / 50

    for (R, G, B), h in zip(cluster_centers, dens):
        color = np.array([R, G, B]) / 255.0
        ax.bar3d(
            R, G, 0,     # x, y, z bottom
            dr, dg, h,   # dx, dy, dz height
            color=color,
            alpha=0.9
        )

    ax.set_xlabel('R', labelpad=10)
    ax.set_ylabel('G', labelpad=10)
    ax.set_zlabel('Density', labelpad=10)
    ax.set_title(f'Cluster Density Towers (bandwidth={bandwidth})', pad=20)

    # Remove numeric ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def main():
    # Example usage with various images/bandwidths
    image_names = ["blue_bird.jpg", "peppers.png", "gas_station.png", "color_monkey.jpg"]
    bandwidths = [30.0, 15.0, 20.0, 25.0, 30.0]

    max_iter = 40  # Maximum Mean Shift iterations
    epsilon = 1  # Convergence threshold
    cluster_threshold = 5.0

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for image_name in image_names:
        input_image_path = os.path.join("img", image_name)
        img_np = cv2.imread(input_image_path)
        if img_np is None:
            print(f"Error: Could not load {input_image_path}")
            continue

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float64)
        h, w, _ = img_np.shape
        flattened_pixels = img_np.reshape(-1, 3)

        for bw in bandwidths:
            print(f"\nProcessing image '{image_name}' with bandwidth {bw}...")
            base_name = os.path.splitext(image_name)[0]
            prefix = f"{base_name}_bw{bw}"
            output_image_path = os.path.join(results_dir, f"{prefix}_clustered_image.png")
            gif_2d_path = os.path.join(results_dir, f"{prefix}_animation_2d.gif")
            gif_3d_path = os.path.join(results_dir, f"{prefix}_animation_3d.gif")

            # Run Mean Shift
            shifted_pixels, frames_2d, frames_3d = MeanShiftColorPixel(
                flattened_pixels, bw, max_iter, epsilon, shape=(h, w), record_frames=True
            )

            # Build 2D GIF
            fig2d = plt.figure()
            ims_2d = []
            for frame in frames_2d:
                im = plt.imshow(frame, animated=True)
                ims_2d.append([im])
            ani2d = animation.ArtistAnimation(fig2d, ims_2d, interval=500)
            ani2d.save(gif_2d_path, writer='pillow')
            plt.close(fig2d)
            print(f"2D animation saved as '{gif_2d_path}'")

            # Build 3D GIF
            fig3d = plt.figure()
            ims_3d = []
            for frame in frames_3d:
                im = plt.imshow(frame, animated=True)
                ims_3d.append([im])
            ani3d = animation.ArtistAnimation(fig3d, ims_3d, interval=500)
            ani3d.save(gif_3d_path, writer='pillow')
            plt.close(fig3d)
            print(f"3D animation saved as '{gif_3d_path}'")

            # Cluster final positions
            labels, cluster_centers = assign_clusters(shifted_pixels, threshold=cluster_threshold)
            num_clusters = len(cluster_centers)
            print("Number of clusters found:", num_clusters)

            # Replace each pixel with its cluster center
            final_pixels = cluster_centers[labels]
            final_img_np = np.clip(final_pixels, 0, 255).reshape(h, w, 3).astype(np.uint8)

            # Save final result
            final_img_bgr = cv2.cvtColor(final_img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, final_img_bgr)
            print(f"Clustered image saved as '{output_image_path}'")

            # Optionally show the final image
            plt.figure(figsize=(8, 6))
            plt.imshow(final_img_np)
            plt.title(f"Final Clustered Image ({num_clusters} clusters) - Bandwidth {bw}")
            plt.axis("off")
            plt.show()

            # Visualize the 3D color density topography
            tower_path = os.path.join(results_dir, f"{prefix}_density.png")
            visualize_density(
                shifted_pixels,
                cluster_centers,
                bandwidth=bw,
                save_path=tower_path
            )
            print(f"3D color density topography saved as '{tower_path}'")


if __name__ == "__main__":
    main()