import pytest
import numpy as np
import cv2
from src.pipeline import BaseballOrientationPipeline
from src.utils.camera import load_camera_params

def test_pipeline_init():
    K, dist, _ = load_camera_params("config/camera.json")
    pipeline = BaseballOrientationPipeline(camera_matrix=K, dist_coeffs=dist)
    assert pipeline is not None

def test_pipeline_process_frame():
    # Create synthetic test image
    img = np.ones((1200, 1700, 3), dtype=np.uint8) * 255
    cv2.circle(img, (850, 600), 100, (200, 200, 200), -1)  # Gray ball
    cv2.circle(img, (850, 600), 100, (0, 0, 200), 2)  # Red seam

    K, dist, _ = load_camera_params("config/camera.json")
    pipeline = BaseballOrientationPipeline(camera_matrix=K, dist_coeffs=dist)

    result = pipeline.process_frame(img, timestamp=0.0)

    assert "ball_detected" in result
    assert "orientation" in result
    assert "frame_number" in result
