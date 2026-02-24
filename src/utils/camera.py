import json
import numpy as np
import cv2

def load_camera_params(path: str):
    """Load camera parameters from JSON file.

    Args:
        path: Path to camera JSON file

    Returns:
        K: 3x3 camera matrix
        dist: 1x5 distortion coefficients
        img_shape: Image shape as tuple (H, W, C)
    """
    with open(path, 'r') as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    img_shape = tuple(data["img_shape"])

    return K, dist, img_shape

def undistort_image(image, K, dist):
    """Undistort image using camera parameters.

    Args:
        image: Input image (H, W, C)
        K: 3x3 camera matrix
        dist: 1x5 distortion coefficients

    Returns:
        Undistorted image
    """
    return cv2.undistort(image, K, dist)
