"""Undistort image using camera parameters."""
import numpy as np
import cv2


def undistort(image, camera_matrix, dist_coeffs):
    """Undistort image using camera intrinsic parameters.

    Args:
        image: Input image (H, W, C) or (H, W) for grayscale
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients

    Returns:
        Undistorted image
    """
    return cv2.undistort(image, camera_matrix, dist_coeffs)
