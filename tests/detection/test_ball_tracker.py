"""Tests for the ball tracking module."""
import pytest
import numpy as np
import cv2
from src.detection.ball_detector import BallDetector
from src.detection.ball_tracker import BallTracker


def test_ball_tracker_init():
    """Test tracker initialization."""
    detector = BallDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    tracker = BallTracker(detector=detector, max_lost_frames=5)
    assert tracker.detector is not None
    assert tracker.max_lost_frames == 5
    assert tracker.track_bbox is None


def test_ball_tracker_reset():
    """Test tracker reset functionality."""
    detector = BallDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    tracker = BallTracker(detector=detector)
    tracker.track_bbox = (100, 100, 200, 200)
    tracker.lost_frames = 3
    tracker.reset()
    assert tracker.track_bbox is None
    assert tracker.lost_frames == 0


def test_tracker_iou():
    """Test IoU calculation."""
    detector = BallDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    tracker = BallTracker(detector=detector)

    # Same bbox
    bbox1 = (100, 100, 200, 200)
    iou = tracker._iou(bbox1, bbox1)
    assert iou == pytest.approx(1.0)

    # No overlap
    bbox2 = (300, 300, 400, 400)
    iou = tracker._iou(bbox1, bbox2)
    assert iou == 0.0

    # Partial overlap
    bbox3 = (150, 150, 250, 250)
    iou = tracker._iou(bbox1, bbox3)
    assert 0 < iou < 1


def test_tracker_predict_bbox():
    """Test bbox prediction using velocity."""
    detector = BallDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    tracker = BallTracker(detector=detector)

    # Set initial state
    tracker.track_bbox = (100, 100, 200, 200)
    tracker.velocity = (10, 5)

    predicted = tracker._predict_bbox()
    assert predicted == (110, 105, 210, 205)


def test_tracker_track_frame():
    """Test tracking on a frame."""
    detector = BallDetector(model_name="yolov8n.pt", confidence_threshold=0.25)
    tracker = BallTracker(detector=detector, max_lost_frames=5)

    # Create a test frame with a circle
    frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.circle(frame, (400, 300), 50, (100, 100, 100), -1)

    result = tracker.track(frame)

    # Should have keys
    assert "detected" in result
    assert "bbox" in result
    assert "confidence" in result
    assert "tracking" in result
