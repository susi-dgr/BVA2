import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

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
    ax.set_title(f'Iteration {iteration + 1} - 2D (Bandwidth: {bandwidth})', fontsize=12)

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
    ax.set_title(f'Iteration {iteration + 1} - 3D (Bandwidth: {bandwidth})', fontsize=12)

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