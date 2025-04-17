import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from density_plot import color_density_3D
from mean_shift import mean_shift_color_pixel, assign_clusters


def main():
    # Example images
    images = ["peppers.png", "gas_station.png", "color_monkey.jpg", "blue_bird.jpg"]
    bandwidths = [10.0, 20.0, 30.0]

    max_iter = 40  # Maximum Mean Shift iterations
    epsilon = 1  # Convergence threshold
    cluster_threshold = 5.0

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for image_name in images:
        input_image_path = os.path.join("img", image_name)
        img_np = cv2.imread(input_image_path)
        if img_np is None:
            print(f"Error: Could not load {input_image_path}")
            continue

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float64)
        h, w, _ = img_np.shape
        flattened_pixels = img_np.reshape(-1, 3)

        # 3D color density topography
        color_density_3D(input_image_path, bw_method=0.5)

        for bw in bandwidths:
            print(f"\nProcessing image '{image_name}' with Bandwidth {bw}...")
            base_name = os.path.splitext(image_name)[0]
            prefix = f"{base_name}_bw{bw}"
            output_image_path = os.path.join(results_dir, f"{prefix}_clustered_image.png")
            gif_2d_path = os.path.join(results_dir, f"{prefix}_animation_2d.gif")
            gif_3d_path = os.path.join(results_dir, f"{prefix}_animation_3d.gif")

            # Run Mean Shift
            shifted_pixels, frames_2d, frames_3d = mean_shift_color_pixel(
                flattened_pixels, bw, max_iter, epsilon, shape=(h, w), record_frames=True
            )

            # Build 2D GIF
            fig2d = plt.figure()
            ims_2d = []
            for frame in frames_2d:
                im = plt.imshow(frame, animated=True)
                plt.axis("off")
                ims_2d.append([im])
            fig2d.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ani2d = animation.ArtistAnimation(fig2d, ims_2d, interval=500)
            ani2d.save(gif_2d_path, writer='pillow')
            plt.axis("off")
            plt.close(fig2d)
            print(f"2D animation saved as '{gif_2d_path}'")

            # Build 3D GIF
            fig3d = plt.figure()
            ims_3d = []
            for frame in frames_3d:
                im = plt.imshow(frame, animated=True)
                plt.axis("off")
                ims_3d.append([im])
            fig3d.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ani3d = animation.ArtistAnimation(fig3d, ims_3d, interval=500)
            ani3d.save(gif_3d_path, writer='pillow')
            plt.axis("off")
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
            plt.figure(figsize=(8, 6))
            plt.imshow(final_img_np)
            plt.title(f"Final Clustered Image ({num_clusters} clusters) - Bandwidth {bw}")
            plt.axis("off")
            plt.savefig(output_image_path.replace(".png", "_final.png"))

            print(f"Clustered image saved as '{output_image_path}'")

if __name__ == "__main__":
    main()