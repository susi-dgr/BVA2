import numpy as np
from numba import njit, prange

from visualization import create_visualization_frames


def colour_dist(refColor, currColor):
    """
    Calculates the Euclidean distance between two points in 3D RGB space
    """
    return np.sqrt(np.sum((refColor - currColor) ** 2))

@njit
def gaussian_weight(dist, bandwidth):
    """
    Calculates the Gaussian kernel weight given the distance and bandwidth
    """
    return np.exp(-dist / (2.0 * bandwidth * bandwidth))

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
            weight = gaussian_weight(dist_sq, bandwidth)
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


def mean_shift_color_pixel(inPixels, bandwidth, max_iter=40, epsilon=1e-3, shape=None, record_frames=False):
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
            if colour_dist(color, center) < threshold:
                labels[i] = j
                found_cluster = True
                break
        if not found_cluster:
            clusters.append(color)
            labels[i] = len(clusters) - 1
    return labels, np.array(clusters)