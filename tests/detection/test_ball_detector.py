import pytest
import numpy as np
from src.detection.ball_detector import BallDetector

def test_ball_detector_init():
    detector = BallDetector(model_name="yolov8n.pt")
    assert detector.model is not None

def test_detect_ball_shape():
    detector = BallDetector(model_name="yolov8n.pt")
    test_img = np.zeros((1200, 1700, 3), dtype=np.uint8)
    result = detector.detect(test_img)

    assert "bbox" in result
    assert "confidence" in result
    assert "detected" in result
    if result["detected"]:
        x1, y1, x2, y2 = result["bbox"]
        assert x2 > x1 and y2 > y1
