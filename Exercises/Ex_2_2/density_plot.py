import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def color_density_3D(image_path, bw_method=0.05, subsample=2000, grid_size=50, fixed_B=128):
    # Load and prepare image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Image not found: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(float)

    # Optional subsampling
    if len(pixels) > subsample:
        idx = np.random.choice(len(pixels), subsample, replace=False)
        pixels = pixels[idx]

    # Gaussian KDE in RGB space
    kde = gaussian_kde(pixels.T, bw_method=bw_method)

    # Grid for R and G, fixed B
    r = np.linspace(0, 255, grid_size)
    g = np.linspace(0, 255, grid_size)
    R, G = np.meshgrid(r, g)
    B = np.full_like(R, fixed_B)

    grid_points = np.vstack([R.ravel(), G.ravel(), B.ravel()])
    density = kde(grid_points).reshape(R.shape)

    # Normalize RGB for facecolors
    face_colors = np.stack([R / 255.0, G / 255.0, B / 255.0], axis=-1)

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(R, G, density, facecolors=face_colors, rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Density')
    ax.set_title(f'3D Color Density Topography - RGB slice at Blue={fixed_B}')

    plt.tight_layout()

    # Save figure
    output_path = f"results/{os.path.splitext(os.path.basename(image_path))[0]}_density_B{fixed_B}.png"
    plt.savefig(output_path)
    print(f"Saved 3D density plot to '{output_path}'")

    # Show plot after saving
    plt.show()
