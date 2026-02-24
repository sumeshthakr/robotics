import pytest
import numpy as np
import os
from src.utils.camera import load_camera_params, undistort_image

# Get the directory of this test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to config directory from tests/utils/
PROJECT_ROOT = os.path.dirname(os.path.dirname(TEST_DIR))
CAMERA_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "camera.json")


def test_load_camera_params():
    K, dist, img_shape = load_camera_params(CAMERA_CONFIG_PATH)
    assert K.shape == (3, 3)
    assert dist.shape == (1, 5)
    assert img_shape == (1700, 1200, 3)
    assert K[0, 0] == pytest.approx(10248.145, rel=0.01)

def test_undistort_image_shape():
    import cv2
    K, dist, _ = load_camera_params(CAMERA_CONFIG_PATH)
    test_img = np.zeros((1200, 1700, 3), dtype=np.uint8)
    result = undistort_image(test_img, K, dist)
    assert result.shape == test_img.shape
