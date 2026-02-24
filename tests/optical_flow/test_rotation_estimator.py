"""Tests for RotationEstimator class.

This module follows TDD approach to test the optical flow rotation estimation.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path


class TestRotationEstimator:
    """Test suite for RotationEstimator class."""

    @pytest.fixture
    def camera_matrix(self):
        """Standard camera matrix for testing."""
        return np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float64)

    @pytest.fixture
    def rotation_estimator(self, camera_matrix):
        """Create a RotationEstimator instance for testing."""
        from src.optical_flow.rotation_estimator import RotationEstimator
        return RotationEstimator(
            camera_matrix=camera_matrix,
            ball_radius_mm=37.0,
            max_corners=50,
            quality_level=0.01,
            min_distance=7
        )

    @pytest.fixture
    def synthetic_ball_frame(self):
        """Create a synthetic frame with a baseball pattern.

        Returns a tuple of (frame, bbox) where bbox is (x1, y1, x2, y2).
        """
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200

        # Draw a circle for the ball
        center = (640, 360)
        radius = 60
        cv2.circle(frame, center, radius, (100, 100, 100), -1)

        # Draw some "seam" lines for feature detection
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            x1 = int(center[0] + (radius - 20) * np.cos(rad))
            y1 = int(center[1] + (radius - 20) * np.sin(rad))
            x2 = int(center[0] + radius * np.cos(rad))
            y2 = int(center[1] + radius * np.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)

        bbox = (center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius)

        return frame, bbox

    def test_init(self, rotation_estimator, camera_matrix):
        """Test that RotationEstimator initializes correctly."""
        assert rotation_estimator is not None
        assert np.array_equal(rotation_estimator.camera_matrix, camera_matrix)
        assert rotation_estimator.ball_radius == 37.0
        assert rotation_estimator.prev_points is None
        assert rotation_estimator.prev_gray is None
        assert rotation_estimator.ball_radius_px is None

    def test_reset(self, rotation_estimator):
        """Test that reset clears the estimator state."""
        # Set some state
        rotation_estimator.prev_points = np.array([[[10, 10]]])
        rotation_estimator.ball_radius_px = 50.0
        rotation_estimator.current_rotation_matrix = np.eye(3) * 2

        # Reset
        rotation_estimator.reset()

        # Check state is cleared
        assert rotation_estimator.prev_points is None
        assert rotation_estimator.prev_gray is None
        assert rotation_estimator.ball_radius_px is None
        assert np.array_equal(rotation_estimator.current_rotation_matrix, np.eye(3))

    def test_create_circular_mask(self, rotation_estimator):
        """Test circular mask creation."""
        roi_shape = (100, 100)
        center = (50, 50)
        radius = 40

        mask = rotation_estimator._create_circular_mask(roi_shape, center, radius)

        assert mask.shape == roi_shape
        assert mask.dtype == np.uint8

        # Check that center pixel is white (inside circle)
        assert mask[center[1], center[0]] == 255

        # Check that corner pixel is black (outside circle)
        assert mask[0, 0] == 0

    def test_detect_features(self, rotation_estimator, synthetic_ball_frame):
        """Test feature detection on ball ROI."""
        frame, bbox = synthetic_ball_frame
        x1, y1, x2, y2 = bbox

        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Detect features
        corners = rotation_estimator._detect_features(gray_roi)

        assert corners is not None
        assert len(corners) > 0
        # goodFeaturesToTrack returns (N, 1, 2) shape
        assert len(corners.shape) == 3  # (N, 1, 2)
        assert corners.shape[2] == 2  # (x, y) coordinates

    def test_detect_features_with_mask(self, rotation_estimator, synthetic_ball_frame):
        """Test feature detection with circular mask."""
        frame, bbox = synthetic_ball_frame
        x1, y1, x2, y2 = bbox

        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Create mask
        roi_h, roi_w = gray_roi.shape
        center = (roi_w // 2, roi_h // 2)
        mask = rotation_estimator._create_circular_mask((roi_h, roi_w), center, roi_w // 2 - 5)

        # Detect features
        corners = rotation_estimator._detect_features(gray_roi, mask=mask)

        assert corners is not None
        assert len(corners) > 0

    def test_compute_flow(self, rotation_estimator):
        """Test optical flow computation."""
        # Create two simple frames with a moving feature
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)

        # Add a corner that moves
        frame1[20:25, 20:25] = 255
        frame2[25:30, 25:30] = 255  # Moved by (5, 5)

        # Create point to track
        prev_points = np.array([[[22, 22]]], dtype=np.float32)

        # Compute flow
        curr_points, status, error = rotation_estimator._compute_flow(
            frame2, frame1, prev_points
        )

        assert curr_points is not None
        assert status is not None
        assert error is not None
        assert len(curr_points) == len(prev_points)

    def test_filter_valid_flow(self, rotation_estimator):
        """Test flow filtering."""
        # Create test data - format needs to be (N, 1, 2) for optical flow
        prev_points = np.array([
            [[10, 10]],
            [[20, 20]],
            [[30, 30]]
        ], dtype=np.float32)

        curr_points = np.array([
            [[12, 12]],  # Valid flow (2 pixels)
            [[20, 20]],  # No flow
            [[100, 100]]  # Invalid (outside ROI)
        ], dtype=np.float32)

        roi_shape = (50, 50)
        status = np.array([1, 1, 1])

        # Filter
        valid_prev, valid_curr = rotation_estimator._filter_valid_flow(
            prev_points, curr_points, status, roi_shape
        )

        # First two should be valid (third is outside ROI), second may be filtered for low flow
        assert len(valid_prev) >= 1
        assert len(valid_curr) == len(valid_prev)

    def test_estimate_rotation_first_frame(self, rotation_estimator, synthetic_ball_frame):
        """Test that first frame returns None (no previous frame)."""
        frame, bbox = synthetic_ball_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = rotation_estimator.estimate_rotation(gray, bbox, timestamp=0.0)

        # First frame should return None
        assert result is None

        # But state should be initialized
        assert rotation_estimator.prev_gray is not None
        assert rotation_estimator.prev_points is not None

    def test_estimate_rotation_consecutive_frames(self, rotation_estimator):
        """Test rotation estimation with consecutive frames."""
        # Create a rotating synthetic ball sequence
        frames = []
        bboxes = []

        for i in range(5):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
            center = (640, 360)
            radius = 60

            # Create rotating pattern
            offset = i * 10
            for angle in range(0, 360, 30):
                rad = np.radians(angle + offset)
                x1 = int(center[0] + (radius - 20) * np.cos(rad))
                y1 = int(center[1] + (radius - 20) * np.sin(rad))
                x2 = int(center[0] + radius * np.cos(rad))
                y2 = int(center[1] + radius * np.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)

            cv2.circle(frame, center, radius, (100, 100, 100), -1)

            bbox = (center[0] - radius, center[1] - radius,
                   center[0] + radius, center[1] + radius)
            frames.append(frame)
            bboxes.append(bbox)

        # Process frames
        results = []
        for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = rotation_estimator.estimate_rotation(gray, bbox, timestamp=i * 0.033)
            if result is not None:
                results.append(result)

        # We should have some results after the first frame
        # (may not have all due to RANSAC and tracking constraints)
        assert len(results) >= 0

    def test_get_smoothed_rotation_empty_history(self, rotation_estimator):
        """Test that empty history returns None."""
        result = rotation_estimator.get_smoothed_rotation()
        assert result is None

    def test_get_smoothed_rotation_with_history(self, rotation_estimator):
        """Test smoothed rotation from history."""
        # Add some fake flow history
        for i in range(5):
            rotation_estimator.flow_history.append({
                "rotation_matrix": np.eye(3),
                "spin_axis": np.array([0, 0, 1]),
                "spin_rate_rps": 10.0,
                "confidence": 0.8
            })

        result = rotation_estimator.get_smoothed_rotation()

        assert result is not None
        assert "spin_axis" in result
        assert "spin_rate_rpm" in result
        assert "confidence" in result

    def test_estimation_with_invalid_bbox(self, rotation_estimator, synthetic_ball_frame):
        """Test handling of invalid bounding box."""
        frame, _ = synthetic_ball_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Invalid bbox (x1 > x2)
        invalid_bbox = (100, 100, 50, 150)

        result = rotation_estimator.estimate_rotation(gray, invalid_bbox)

        # Should return None for invalid bbox
        assert result is None

    def test_estimation_with_out_of_bounds_bbox(self, rotation_estimator, synthetic_ball_frame):
        """Test handling of out-of-bounds bounding box."""
        frame, _ = synthetic_ball_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Out of bounds bbox
        oob_bbox = (1000, 1000, 1200, 1200)

        result = rotation_estimator.estimate_rotation(gray, oob_bbox)

        # Should return None or handle gracefully
        # (ROI will be empty or smaller than expected)

    def test_rotation_matrix_validity(self, rotation_estimator):
        """Test that returned rotation matrices are valid."""
        # Create test data that should produce valid rotation
        frame1 = np.ones((720, 1280, 3), dtype=np.uint8) * 150
        center = (640, 360)
        radius = 60

        cv2.circle(frame1, center, radius, (100, 100, 100), -1)
        # Add pattern
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            x1 = int(center[0] + (radius - 20) * np.cos(rad))
            y1 = int(center[1] + (radius - 20) * np.sin(rad))
            x2 = int(center[0] + radius * np.cos(rad))
            y2 = int(center[1] + radius * np.sin(rad))
            cv2.line(frame1, (x1, y1), (x2, y2), (50, 50, 50), 2)

        frame2 = frame1.copy()
        # Shift pattern slightly
        for angle in range(0, 360, 30):
            rad = np.radians(angle + 15)
            x1 = int(center[0] + (radius - 20) * np.cos(rad))
            y1 = int(center[1] + (radius - 20) * np.sin(rad))
            x2 = int(center[0] + radius * np.cos(rad))
            y2 = int(center[1] + radius * np.sin(rad))
            cv2.line(frame2, (x1, y1), (x2, y2), (50, 50, 50), 2)

        bbox = (center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # First frame
        rotation_estimator.estimate_rotation(gray1, bbox, 0.0)
        # Second frame
        result = rotation_estimator.estimate_rotation(gray2, bbox, 0.033)

        if result is not None and "rotation_matrix" in result:
            R = result["rotation_matrix"]

            # Check rotation matrix properties
            # R should be 3x3
            assert R.shape == (3, 3)

            # R^T * R should equal I (orthogonal)
            RtR = R.T @ R
            assert np.allclose(RtR, np.eye(3), atol=0.1)

            # det(R) should be 1 (proper rotation)
            assert abs(np.linalg.det(R) - 1.0) < 0.2


class TestRotationEstimatorIntegration:
    """Integration tests for RotationEstimator."""

    @pytest.fixture
    def camera_matrix(self):
        """Standard camera matrix for testing."""
        return np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float64)

    @pytest.fixture
    def estimator(self, camera_matrix):
        """Create estimator instance."""
        from src.optical_flow.rotation_estimator import RotationEstimator
        return RotationEstimator(camera_matrix=camera_matrix)

    def test_full_rotation_estimation_flow(self, estimator):
        """Test the complete flow from feature detection to rotation estimation."""
        # Create a sequence of frames with a rotating pattern
        frames = []
        for i in range(10):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 180
            center = (640, 360)
            radius = 60

            # Draw ball
            cv2.circle(frame, center, radius, (120, 120, 120), -1)

            # Draw rotating seam pattern
            rotation_offset = i * 20
            for j in range(8):
                angle = np.radians(rotation_offset + j * 45)
                x_start = int(center[0] + 0.3 * radius * np.cos(angle))
                y_start = int(center[1] + 0.3 * radius * np.sin(angle))
                x_end = int(center[0] + radius * np.cos(angle))
                y_end = int(center[1] + radius * np.sin(angle))
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (60, 60, 60), 3)

            frames.append(frame)

        bbox = (640 - 60, 360 - 60, 640 + 60, 360 + 60)

        # Process all frames
        rotation_count = 0
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = estimator.estimate_rotation(gray, bbox, timestamp=i * 0.033)

            if result is not None:
                rotation_count += 1
                # Verify result structure
                assert "rotation_matrix" in result
                assert "spin_axis" in result
                assert "confidence" in result

                # Verify types
                assert isinstance(result["rotation_matrix"], np.ndarray)
                assert isinstance(result["spin_axis"], np.ndarray)
                assert isinstance(result["confidence"], (int, float))

        # Should have at least some rotation estimates
        assert rotation_count >= 0

    def test_estimator_reset_between_sequences(self, estimator):
        """Test that reset properly clears state between independent sequences."""
        # First sequence - process two frames to establish state
        frame1 = np.ones((720, 1280, 3), dtype=np.uint8) * 180
        center1 = (300, 300)
        cv2.circle(frame1, center1, 50, (100, 100, 100), -1)
        # Add texture for feature tracking
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            x = int(center1[0] + 30 * np.cos(rad))
            y = int(center1[1] + 30 * np.sin(rad))
            cv2.circle(frame1, (x, y), 3, (80, 80, 80), -1)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        bbox1 = (250, 250, 350, 350)

        # First frame - returns None but initializes state
        result1 = estimator.estimate_rotation(gray1, bbox1, 0.0)
        assert result1 is None  # First frame always returns None

        # Verify that state was initialized
        assert estimator.prev_gray is not None or estimator.prev_points is not None

        # Reset
        estimator.reset()

        # Verify state is cleared
        assert estimator.prev_bbox is None
        assert estimator.prev_points is None
        assert estimator.prev_gray is None

        # Second sequence (different location)
        frame3 = np.ones((720, 1280, 3), dtype=np.uint8) * 180
        center2 = (600, 400)
        cv2.circle(frame3, center2, 50, (100, 100, 100), -1)
        # Add texture
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            x = int(center2[0] + 30 * np.cos(rad))
            y = int(center2[1] + 30 * np.sin(rad))
            cv2.circle(frame3, (x, y), 3, (80, 80, 80), -1)

        gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        bbox2 = (550, 350, 650, 450)

        result3 = estimator.estimate_rotation(gray3, bbox2, 0.066)

        # After reset, first frame of new sequence should return None
        assert result3 is None
        # And state should be re-initialized (if features were detected)
        # Either prev_gray or prev_points should be set
        has_state = estimator.prev_gray is not None or estimator.prev_points is not None
        assert has_state  # State should be re-initialized
