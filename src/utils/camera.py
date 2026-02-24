import json
import os
import numpy as np
import cv2

# Required keys for camera parameters JSON
_REQUIRED_KEYS = ["camera_matrix", "dist_coeffs", "img_shape"]


def load_camera_params(path: str):
    """Load camera parameters from JSON file.

    Args:
        path: Path to camera JSON file

    Returns:
        K: 3x3 camera matrix
        dist: 1x5 distortion coefficients
        img_shape: Image shape as tuple (H, W, C)

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If required keys are missing or have invalid format
        json.JSONDecodeError: If the file is not valid JSON
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Camera parameters file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    # Validate required keys
    missing_keys = [key for key in _REQUIRED_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in camera parameters: {missing_keys}")

    # Validate camera_matrix is 3x3
    camera_matrix = data["camera_matrix"]
    if (not isinstance(camera_matrix, list) or len(camera_matrix) != 3 or
            any(len(row) != 3 for row in camera_matrix)):
        raise ValueError("camera_matrix must be a 3x3 array")

    # Validate dist_coeffs - can be [[c1, c2, c3, c4, c5]] or [c1, c2, c3, c4, c5]
    dist_coeffs = data["dist_coeffs"]
    if not isinstance(dist_coeffs, (list, tuple)):
        raise ValueError("dist_coeffs must be a list or array")

    # Handle nested array format [[c1, c2, c3, c4, c5]] or flat [c1, c2, c3, c4, c5]
    if len(dist_coeffs) == 1 and isinstance(dist_coeffs[0], (list, tuple)):
        # Nested format [[c1, c2, c3, c4, c5]]
        inner = dist_coeffs[0]
        if len(inner) != 5:
            raise ValueError("dist_coeffs must contain exactly 5 coefficients")
    elif len(dist_coeffs) != 5:
        # Flat format should have exactly 5 elements
        raise ValueError("dist_coeffs must contain exactly 5 coefficients")

    # Validate img_shape is a tuple of 3 elements
    img_shape = data["img_shape"]
    if not isinstance(img_shape, (list, tuple)) or len(img_shape) != 3:
        raise ValueError("img_shape must contain exactly 3 elements (H, W, C)")

    K = np.array(camera_matrix, dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    img_shape = tuple(img_shape)

    return K, dist, img_shape

def undistort_image(image, K, dist):
    """Undistort image using camera parameters.

    Args:
        image: Input image (H, W, C) or (H, W) for grayscale
        K: 3x3 camera matrix
        dist: 1x5 distortion coefficients

    Returns:
        Undistorted image

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If inputs have incorrect shapes
    """
    # Validate image
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")

    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D array, got {image.ndim}D")

    # Validate camera matrix K
    if not isinstance(K, np.ndarray):
        raise TypeError(f"K must be a numpy array, got {type(K).__name__}")

    if K.shape != (3, 3):
        raise ValueError(f"K must be a 3x3 matrix, got shape {K.shape}")

    # Validate distortion coefficients
    if not isinstance(dist, np.ndarray):
        raise TypeError(f"dist must be a numpy array, got {type(dist).__name__}")

    if dist.shape not in ((5,), (1, 5), (5, 1)):
        raise ValueError(f"dist must have 5 elements, got shape {dist.shape}")

    return cv2.undistort(image, K, dist)
