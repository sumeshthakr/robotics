import pytest
import numpy as np
from src.utils.camera import load_camera_params, undistort_image

def test_load_camera_params():
    K, dist, img_shape = load_camera_params("config/camera.json")
    assert K.shape == (3, 3)
    assert dist.shape == (1, 5)
    assert img_shape == (1700, 1200, 3)
    assert K[0, 0] == pytest.approx(10248.145, rel=0.01)

def test_undistort_image_shape():
    import cv2
    K, dist, _ = load_camera_params("config/camera.json")
    test_img = np.zeros((1200, 1700, 3), dtype=np.uint8)
    result = undistort_image(test_img, K, dist)
    assert result.shape == test_img.shape
