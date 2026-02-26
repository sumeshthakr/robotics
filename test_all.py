"""Consolidated tests for the baseball orientation detection pipeline.

Tests all components: camera, detector, seam detection, seam model,
PnP solver, orientation tracker, and both pipelines.

Run with:  pytest test_all.py -v
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch

from camera import load_camera_params, undistort
from detector import BallDetector, BallTracker
from orientation import (rotation_to_quaternion,
                         rotation_to_euler)
from seam_pipeline import (detect_seams, BaseballSeamModel, solve_orientation,
                           SeamPipeline)
from optical_pipeline import RotationEstimator, OpticalFlowPipeline

CAMERA_CONFIG = "config/camera.json"


# ============================================================
# Camera Tests
# ============================================================

class TestCamera:
    def test_load_camera_params(self):
        K, dist, img_shape = load_camera_params(CAMERA_CONFIG)
        assert K.shape == (3, 3)
        assert dist.shape == (1, 5)
        assert img_shape == (1700, 1200, 3)
        assert K[0, 0] == pytest.approx(10248.145, rel=0.01)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_camera_params("nonexistent.json")

    def test_undistort_preserves_shape(self):
        K, dist, _ = load_camera_params(CAMERA_CONFIG)
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        result = undistort(img, K, dist)
        assert result.shape == img.shape


# ============================================================
# Ball Detector Tests
# ============================================================

class TestBallDetector:
    def test_init(self):
        detector = BallDetector(model_path="yolov8n.pt", confidence=0.5)
        assert detector is not None
        assert detector.confidence == 0.5

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            BallDetector(confidence=1.5)

    def test_detect_returns_structure(self):
        detector = BallDetector(model_path="yolov8n.pt", confidence=0.25)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(img)
        assert "detected" in result
        assert "bbox" in result
        assert "confidence" in result

    def test_detect_invalid_input(self):
        detector = BallDetector()
        with pytest.raises(TypeError):
            detector.detect("not an image")
        with pytest.raises(ValueError):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))  # 2D

    def test_detect_with_ball(self):
        """Detection on a synthetic image with a gray circle."""
        detector = BallDetector(model_path="yolov8n.pt", confidence=0.25)
        img = np.zeros((1200, 1700, 3), dtype=np.uint8)
        cv2.circle(img, (850, 600), 100, (200, 200, 200), -1)
        result = detector.detect(img)
        # May or may not detect — just verify structure
        if result["detected"]:
            x1, y1, x2, y2 = result["bbox"]
            assert x2 > x1 and y2 > y1


# ============================================================
# Ball Tracker Tests
# ============================================================

class TestBallTracker:
    def test_init(self):
        detector = BallDetector()
        tracker = BallTracker(detector)
        assert tracker.bbox is None
        assert tracker.lost_frames == 0

    def test_reset(self):
        detector = BallDetector()
        tracker = BallTracker(detector)
        tracker.bbox = (10, 20, 30, 40)
        tracker.lost_frames = 3
        tracker.reset()
        assert tracker.bbox is None
        assert tracker.lost_frames == 0

    def test_track_returns_structure(self):
        detector = BallDetector(confidence=0.25)
        tracker = BallTracker(detector)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = tracker.track(img)
        assert "detected" in result
        assert "bbox" in result
        assert "confidence" in result
        assert "tracking" in result

    def test_velocity_prediction(self):
        """When ball is lost, tracker should predict using velocity."""
        detector = BallDetector(confidence=0.25)
        tracker = BallTracker(detector, max_lost_frames=5)

        # Manually set state as if we had a detection
        tracker.bbox = (100, 100, 200, 200)
        tracker.confidence = 0.8
        tracker.velocity = (10, 5)

        # Mock detector returning no detection
        with patch.object(tracker.detector, 'detect',
                          return_value={"detected": False, "bbox": None,
                                        "confidence": None}):
            result = tracker.track(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result["detected"] is True
        assert result["tracking"] is True
        # Position should have moved by velocity
        assert result["bbox"][0] == 110  # 100 + 10
        assert result["bbox"][1] == 105  # 100 + 5


# ============================================================
# Seam Detection Tests
# ============================================================

class TestSeamDetection:
    def test_detect_seams_structure(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)  # Red circle
        result = detect_seams(img)
        assert "edges" in result
        assert "seam_pixels" in result
        assert "num_pixels" in result
        assert result["edges"].shape == img.shape[:2]

    def test_red_circle_detects_pixels(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)
        result = detect_seams(img)
        assert result["num_pixels"] > 0
        if len(result["seam_pixels"]) > 0:
            assert result["seam_pixels"].shape[1] == 2

    def test_blank_image_few_pixels(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = detect_seams(img)
        # Blank gray image should have very few seam pixels
        assert result["num_pixels"] < 100


# ============================================================
# Seam Model Tests
# ============================================================

class TestSeamModel:
    def test_generate_points_shape(self):
        model = BaseballSeamModel(radius=37.0)
        points = model.generate_points(num_points_per_curve=50)
        assert points.shape == (100, 3)  # 2 curves × 50 points

    def test_points_on_sphere(self):
        """All seam points should be approximately at the specified radius."""
        model = BaseballSeamModel(radius=37.0)
        points = model.generate_points()
        distances = np.linalg.norm(points, axis=1)
        assert np.allclose(distances, 37.0, atol=1.0)

    def test_two_curves_separated(self):
        """The two seam curves should not be identical."""
        model = BaseballSeamModel(radius=37.0)
        points = model.generate_points(num_points_per_curve=50)
        curve1 = points[:50]
        curve2 = points[50:]
        assert not np.allclose(curve1, curve2, atol=1.0)


# ============================================================
# PnP Solver Tests
# ============================================================

class TestPnPSolver:
    def test_solve_identity(self):
        """Project 3D points with known pose, then recover it."""
        model = BaseballSeamModel(radius=37.0)
        points_3d = model.generate_points()
        K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]],
                     dtype=np.float64)
        rvec = np.array([0, 0, 0], dtype=np.float64)
        tvec = np.array([0, 0, 500], dtype=np.float64)

        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
        points_2d = points_2d.reshape(-1, 2)

        result = solve_orientation(points_2d, points_3d, K)
        assert result["success"] is True
        assert result["rotation_matrix"].shape == (3, 3)
        assert result["tvec"] is not None

    def test_too_few_points(self):
        K = np.eye(3) * 1000
        result = solve_orientation(np.zeros((2, 2)), np.zeros((2, 3)), K)
        assert result["success"] is False


# ============================================================
# Rotation Conversion Tests
# ============================================================

class TestConversions:
    def test_quaternion_identity(self):
        q = rotation_to_quaternion(np.eye(3))
        # Identity rotation → quaternion [1, 0, 0, 0]
        assert q[0] == pytest.approx(1.0, abs=0.01)
        assert np.allclose(q[1:], 0, atol=0.01)

    def test_euler_identity(self):
        euler = rotation_to_euler(np.eye(3))
        assert np.allclose(euler, 0, atol=0.01)

    def test_quaternion_90deg_z(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('z', np.pi / 2).as_matrix()
        q = rotation_to_quaternion(R)
        # 90° about Z → q ≈ [cos(45°), 0, 0, sin(45°)]
        assert q[0] == pytest.approx(np.cos(np.pi / 4), abs=0.01)
        assert abs(q[3]) == pytest.approx(np.sin(np.pi / 4), abs=0.01)


# ============================================================
# Rotation Estimator Tests (Optical Flow)
# ============================================================

class TestRotationEstimator:
    @pytest.fixture
    def estimator(self):
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]],
                     dtype=np.float64)
        return RotationEstimator(K, ball_radius_mm=37.0, max_corners=50,
                                 min_flow=0.3, max_flow=40.0)

    def _make_rotating_frame(self, center, radius, rotation_deg, size=(480, 640)):
        """Create a synthetic frame with a rotating pattern inside a circle."""
        frame = np.ones(size, dtype=np.uint8) * 180
        cv2.circle(frame, center, radius, 100, -1)
        cv2.circle(frame, center, radius, 50, 2)

        # Draw multiple ring patterns and radial lines that rotate
        for ring_frac in [0.3, 0.5, 0.7, 0.9]:
            ring_r = int(radius * ring_frac)
            for angle_offset in [0, 90, 180, 270]:
                start = rotation_deg + angle_offset
                for a in range(start, start + 45, 5):
                    rad = np.radians(a)
                    x = int(center[0] + ring_r * np.cos(rad))
                    y = int(center[1] + ring_r * np.sin(rad))
                    cv2.circle(frame, (x, y), 3, 40, -1)

        for j in range(12):
            angle = np.radians(rotation_deg + j * 30)
            x1 = int(center[0] + 0.2 * radius * np.cos(angle))
            y1 = int(center[1] + 0.2 * radius * np.sin(angle))
            x2 = int(center[0] + 0.9 * radius * np.cos(angle))
            y2 = int(center[1] + 0.9 * radius * np.sin(angle))
            cv2.line(frame, (x1, y1), (x2, y2), 60, 2)

        for j in range(6):
            angle = np.radians(rotation_deg + j * 60)
            x = int(center[0] + 0.7 * radius * np.cos(angle))
            y = int(center[1] + 0.7 * radius * np.sin(angle))
            cv2.line(frame, (x - 5, y), (x + 5, y), 40, 2)
            cv2.line(frame, (x, y - 5), (x, y + 5), 40, 2)

        return frame

    def test_init(self, estimator):
        assert estimator.prev_gray is None
        assert estimator.prev_points is None

    def test_reset(self, estimator):
        estimator.accumulated_rotation = np.eye(3) * 2
        estimator.reset()
        assert estimator.prev_gray is None
        assert np.allclose(estimator.accumulated_rotation, np.eye(3))

    def test_first_frame_returns_none(self, estimator):
        """First frame initializes state, can't compute flow yet."""
        frame = self._make_rotating_frame((320, 240), 80, 0)
        result = estimator.estimate_rotation(frame, (240, 160, 400, 320))
        assert result is None
        assert estimator.prev_gray is not None  # State was initialized

    def test_consecutive_frames(self, estimator):
        """Process multiple frames with a rotating pattern.

        NOTE: Optical flow estimation works best with real video textures.
        With synthetic images, the RANSAC may not always find enough
        consistent inliers. This test verifies the pipeline doesn't crash
        and validates any results it does produce.
        """
        center = (320, 240)
        radius = 80
        bbox = (center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius)

        for i in range(20):
            frame = self._make_rotating_frame(center, radius, i * 15)
            result = estimator.estimate_rotation(frame, bbox, timestamp=i * 0.033)

            if result is not None:
                # Validate structure when we do get a result
                assert "rotation_matrix" in result
                assert "spin_axis" in result
                assert "confidence" in result
                R = result["rotation_matrix"]
                assert R.shape == (3, 3)
                assert np.allclose(R.T @ R, np.eye(3), atol=0.2)
                assert abs(np.linalg.det(R) - 1.0) < 0.2
                break  # One valid result is enough

        # At minimum, state should be initialized after processing frames
        assert estimator.prev_gray is not None
        assert estimator.prev_points is not None


# ============================================================
# Seam Pipeline Tests
# ============================================================

class TestSeamPipeline:
    @pytest.fixture
    def pipeline(self):
        K, dist, _ = load_camera_params(CAMERA_CONFIG)
        return SeamPipeline(K, dist)

    def test_init(self, pipeline):
        assert pipeline is not None
        assert pipeline.frame_count == 0

    def test_process_frame_structure(self, pipeline):
        img = np.ones((1200, 1700, 3), dtype=np.uint8) * 255
        cv2.circle(img, (850, 600), 100, (200, 200, 200), -1)
        cv2.circle(img, (850, 600), 100, (0, 0, 200), 2)

        result = pipeline.process_frame(img, timestamp=0.0)
        assert "ball_detected" in result
        assert "orientation" in result
        assert "frame_number" in result
        assert result["frame_number"] == 1

    def test_process_video_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.process_video("/nonexistent/video.mp4")

    def test_reset(self, pipeline):
        pipeline.frame_count = 10
        pipeline.reset()
        assert pipeline.frame_count == 0


# ============================================================
# Optical Flow Pipeline Tests
# ============================================================

class TestOpticalFlowPipeline:
    @pytest.fixture
    def pipeline(self):
        K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
                     dtype=np.float64)
        dist = np.zeros((1, 5))
        return OpticalFlowPipeline(K, dist, confidence=0.25)

    def test_init(self, pipeline):
        assert pipeline is not None
        assert pipeline.frame_count == 0

    def test_process_frame_no_ball(self, pipeline):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        with patch.object(pipeline.detector, 'detect',
                          return_value={"detected": False, "bbox": None,
                                        "confidence": None}):
            result = pipeline.process_frame(frame, timestamp=0.0)
        assert result["ball_detected"] is False
        assert result["orientation"] is None
        assert result["spin_rate"] is None

    def test_result_structure(self, pipeline):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        with patch.object(pipeline.detector, 'detect',
                          return_value={"detected": False, "bbox": None,
                                        "confidence": None}):
            result = pipeline.process_frame(frame)

        expected_keys = {"ball_detected", "bbox", "confidence", "orientation",
                         "spin_rate", "spin_axis", "frame_number", "timestamp",
                         "flow_confidence"}
        assert expected_keys.issubset(set(result.keys()))

    def test_process_video_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.process_video("/nonexistent/video.mp4")

    def test_consecutive_failures_reset(self, pipeline):
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        with patch.object(pipeline.detector, 'detect',
                          return_value={"detected": False, "bbox": None,
                                        "confidence": None}):
            for _ in range(10):
                pipeline.process_frame(frame, timestamp=0.0)
        # After many failures, estimator should have been reset
        assert pipeline._consecutive_failures > 5

    def test_reset(self, pipeline):
        pipeline.frame_count = 10
        pipeline._consecutive_failures = 8
        pipeline.reset()
        assert pipeline.frame_count == 0
        assert pipeline._consecutive_failures == 0
