import numpy as np
import matplotlib.pyplot as plt


def solve_transformation(original, transformed):
    n = len(original)

    # A matrix and B vector
    A = np.zeros((2 * n, 4))
    B = np.zeros(2 * n)

    for i in range(n):
        x, y = original[i]
        x_transformed, y_transformed = transformed[i]

        # x' = s*(x*cos(rot) - y*sin(rot)) + Tx
        A[2 * i, 0] = 1  # coeff for Tx
        A[2 * i, 1] = 0  # coeff for Ty
        A[2 * i, 2] = x  # coeff for s*cos(rot)
        A[2 * i, 3] = -y  # coeff for s*sin(rot)
        B[2 * i] = x_transformed  # x' value

        # y' = s*(x*sin(rot) + y*cos(rot)) + Ty
        A[2 * i + 1, 0] = 0  # coeff for Tx
        A[2 * i + 1, 1] = 1  # coeff for Ty
        A[2 * i + 1, 2] = y  # coeff for s*cos(rot)
        A[2 * i + 1, 3] = x  # coeff for s*sin(rot)
        B[2 * i + 1] = y_transformed  # y' value

    # least squared diff
    # (A^T * A) * X = A^T * B
    ATA = np.dot(A.T, A)
    ATB = np.dot(A.T, B)
    X = np.linalg.solve(ATA, ATB)

    # parameters
    Tx = X[0]
    Ty = X[1]
    s_cos_rot = X[2]
    s_sin_rot = X[3]

    # s and rot
    s = np.sqrt(s_cos_rot ** 2 + s_sin_rot ** 2)
    rot = np.arctan2(s_sin_rot, s_cos_rot)

    return Tx, Ty, s, rot

def transform_points(points, Tx, Ty, s, rot):
    cos_rot = np.cos(rot)
    sin_rot = np.sin(rot)
    rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    transformed = s * np.dot(points, rotation_matrix.T) + np.array([Tx, Ty])
    return transformed

if __name__ == '__main__':
    # original points
    p_original = np.array([[1, 4],
                        [-4, -2],
                        [0.1, 5],
                        [-1, 2],
                        [3, 3],
                        [7, -2],
                        [5, 5],
                        [-6, 3.3]])

    # transformed points
    p_transformed = np.array([[-1.26546, 3.222386],
                            [-4.53286, 0.459128],
                            [-1.64771, 3.831308],
                            [-2.57985, 2.283247],
                            [-0.28072, 2.44692],
                            [1.322025, -0.69344],
                            [1.021729, 3.299737],
                            [-5.10871, 3.523542]])


    # solve transformation to find optimal parameters
    Tx_opt, Ty_opt, s_opt, rot_opt = solve_transformation(p_original, p_transformed)

    # results
    print(f"Optimal Translation (Tx, Ty): {Tx_opt}, {Ty_opt}")
    print(f"Optimal Scale (s): {s_opt}")
    print(f"Optimal Rotation (rot): {rot_opt * 180 / np.pi} degrees ({rot_opt} radians)")

    # apply tranformation to all points
    p_estimated = transform_points(p_original, Tx_opt, Ty_opt, s_opt, rot_opt)

    # calculate noise
    residuals = p_transformed - p_estimated
    residual_distances = np.sqrt(np.sum(residuals ** 2, axis=1))
    noise_std = np.std(residual_distances)
    print(f"Calculated noise: {noise_std}")

    # calculate mean error
    point_errors = np.sqrt(np.sum((p_estimated - p_transformed) ** 2, axis=1))
    mean_error = np.mean(point_errors)
    print(f"Mean error: {mean_error:.4f}")

    # transformation matrix
    transformation_matrix = np.array([
        [s_opt * np.cos(rot_opt), -s_opt * np.sin(rot_opt), Tx_opt],
        [s_opt * np.sin(rot_opt), s_opt * np.cos(rot_opt), Ty_opt],
        [0, 0, 1]
    ])

    print("\nTransformation Matrix:")
    print(transformation_matrix)

    # plotting
    plt.scatter(p_original[:, 0], p_original[:, 1], color='blue', label='Original Points (Pi)')
    plt.scatter(p_transformed[:, 0], p_transformed[:, 1], color='darkorange', label='Transformed Points (P\'i)')
    plt.scatter(p_estimated[:, 0], p_estimated[:, 1], color='green', marker='x',
                label='Transformed Points with Estimated Parameters')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Transformation with Gaussian Noise')

    plt.show()