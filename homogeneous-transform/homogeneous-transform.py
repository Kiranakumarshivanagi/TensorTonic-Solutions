import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """

    # Convert inputs to numpy arrays
    T = np.array(T, dtype=float)
    points = np.array(points, dtype=float)

    single = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single = True

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # Apply transform
    transformed = points_h @ T.T

    # Remove last coordinate
    result = transformed[:, :3]

    if single:
        return result[0]

    return result