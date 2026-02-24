"""Tests for OpticalFlowPipeline class."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch


class TestOpticalFlowPipeline:
    """Test suite for OpticalFlowPipeline class."""

    @pytest.fixture
    def camera_matrix(self):
        """Standard camera matrix for testing."""
        return np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float64)

    @pytest.fixture
    def dist_coeffs(self):
        """Standard distortion coefficients for testing."""
        return np.array([0.1, -0.2, 0.001, 0.001, 0.0], dtype=np.float64)

    @pytest.fixture
    def pipeline(self, camera_matrix, dist_coeffs):
        """Create an OpticalFlowPipeline instance for testing."""
        from src.pipeline_optical import OpticalFlowPipeline
        return OpticalFlowPipeline(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            ball_radius_mm=37.0,
            confidence_threshold=0.5,
            model_path="yolov8n.pt"
        )

    @pytest.fixture
    def synthetic_frame_with_ball(self):
        """Create a synthetic frame with a baseball."""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200

        # Draw ball
        center = (640, 360)
        radius = 60
        cv2.circle(frame, center, radius, (100, 100, 100), -1)

        # Add texture
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            x1 = int(center[0] + (radius - 20) * np.cos(rad))
            y1 = int(center[1] + (radius - 20) * np.sin(rad))
            x2 = int(center[0] + radius * np.cos(rad))
            y2 = int(center[1] + radius * np.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)

        return frame

    def test_init(self, pipeline, camera_matrix, dist_coeffs):
        """Test that OpticalFlowPipeline initializes correctly."""
        assert pipeline is not None
        assert np.array_equal(pipeline.camera_matrix, camera_matrix)
        assert np.array_equal(pipeline.dist_coeffs, dist_coeffs)
        assert pipeline.ball_radius_mm == 37.0
        assert pipeline._frame_count == 0

    def test_reset(self, pipeline):
        """Test that reset clears pipeline state."""
        pipeline._frame_count = 10
        pipeline._consecutive_failures = 3

        pipeline.reset()

        assert pipeline._frame_count == 0
        assert pipeline._consecutive_failures == 0

    def test_process_frame_no_ball(self, pipeline):
        """Test processing frame without a ball."""
        # Empty frame
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255

        with patch.object(pipeline.detector, 'detect', return_value={
            "detected": False,
            "bbox": None,
            "confidence": None
        }):
            result = pipeline.process_frame(frame, timestamp=0.0)

        assert result["ball_detected"] is False
        assert result["bbox"] is None
        assert result["confidence"] is None
        assert result["orientation"] is None
        assert result["spin_rate"] is None
        assert result["frame_number"] == 1

    def test_process_frame_with_ball(self, pipeline, synthetic_frame_with_ball):
        """Test processing frame with a ball."""
        frame = synthetic_frame_with_ball

        # Mock the detector to return a detection
        with patch.object(pipeline.detector, 'detect', return_value={
            "detected": True,
            "bbox": (580, 300, 700, 420),
            "confidence": 0.9
        }):
            result = pipeline.process_frame(frame, timestamp=0.0)

        # Should detect the ball
        assert result["ball_detected"] is True
        assert result["bbox"] is not None
        assert result["confidence"] is not None
        assert result["frame_number"] == 1

    def test_process_frame_sequence(self, pipeline):
        """Test processing a sequence of frames."""
        # Create a sequence of frames
        frames = []
        for i in range(5):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 180
            center = (640, 360)
            radius = 60

            cv2.circle(frame, center, radius, (120, 120, 120), -1)

            # Rotating pattern
            offset = i * 15
            for angle in range(0, 360, 45):
                rad = np.radians(angle + offset)
                x1 = int(center[0] + (radius - 20) * np.cos(rad))
                y1 = int(center[1] + (radius - 20) * np.sin(rad))
                x2 = int(center[0] + radius * np.cos(rad))
                y2 = int(center[1] + radius * np.sin(rad))
                cv2.line(frame, (x1, y1), (x2, y2), (60, 60, 60), 3)

            frames.append(frame)

        # Mock detector
        with patch.object(pipeline.detector, 'detect', return_value={
            "detected": True,
            "bbox": (580, 300, 700, 420),
            "confidence": 0.9
        }):
            results = []
            for i, frame in enumerate(frames):
                result = pipeline.process_frame(frame, timestamp=i * 0.033)
                results.append(result)

        # Check results
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["ball_detected"] is True
            assert result["frame_number"] == i + 1

    def test_create_visualization(self, pipeline, synthetic_frame_with_ball):
        """Test visualization creation."""
        frame = synthetic_frame_with_ball

        result = {
            "ball_detected": True,
            "bbox": (580, 300, 700, 420),
            "confidence": 0.85,
            "orientation": {
                "rotation_matrix": np.eye(3),
                "quaternion": [1, 0, 0, 0],
                "euler_angles": [0, 0, 0]
            },
            "spin_rate": 1500.0,
            "spin_axis": np.array([0.1, 0.2, 0.975]),
            "flow_confidence": 0.8
        }

        vis_frame = pipeline._create_visualization(frame, result, 10)

        # Check that visualization frame is same size
        assert vis_frame.shape == frame.shape

        # Check that some pixels changed (visualization added)
        assert not np.array_equal(vis_frame, frame)

    def test_create_visualization_no_detection(self, pipeline, synthetic_frame_with_ball):
        """Test visualization when no ball is detected."""
        frame = synthetic_frame_with_ball

        result = {
            "ball_detected": False,
            "bbox": None,
            "confidence": None,
            "orientation": None,
            "spin_rate": None,
            "spin_axis": None,
            "flow_confidence": None
        }

        vis_frame = pipeline._create_visualization(frame, result, 10)

        # Should still return a frame
        assert vis_frame is not None
        assert vis_frame.shape == frame.shape

    def test_consecutive_failures_reset(self, pipeline):
        """Test that consecutive failures trigger estimator reset."""
        empty_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255

        with patch.object(pipeline.detector, 'detect', return_value={
            "detected": False,
            "bbox": None,
            "confidence": None
        }):
            # Process multiple frames without detection
            for i in range(10):
                pipeline.process_frame(empty_frame, timestamp=i * 0.033)

        # After enough consecutive failures, rotation_estimator should be reset
        assert pipeline._consecutive_failures >= pipeline._max_consecutive_failures

    def test_process_video_file_not_found(self, pipeline):
        """Test error handling for non-existent video file."""
        with pytest.raises(FileNotFoundError):
            pipeline.process_video("/nonexistent/path/video.mp4")

    def test_result_structure(self, pipeline, synthetic_frame_with_ball):
        """Test that result has correct structure."""
        frame = synthetic_frame_with_ball

        with patch.object(pipeline.detector, 'detect', return_value={
            "detected": True,
            "bbox": (580, 300, 700, 420),
            "confidence": 0.9
        }):
            result = pipeline.process_frame(frame, timestamp=0.0)

        # Check all required keys are present
        required_keys = [
            "ball_detected", "bbox", "confidence", "orientation",
            "spin_rate", "spin_axis", "frame_number", "timestamp",
            "flow_confidence"
        ]
        for key in required_keys:
            assert key in result

    def test_spin_axis_normalization(self, pipeline, synthetic_frame_with_ball):
        """Test that spin axis is properly normalized."""
        frame = synthetic_frame_with_ball

        with patch.object(pipeline.detector, 'detect', return_value={
            "detected": True,
            "bbox": (580, 300, 700, 420),
            "confidence": 0.9
        }):
            result = pipeline.process_frame(frame, timestamp=0.0)

        if result["spin_axis"] is not None:
            axis = result["spin_axis"]
            norm = np.linalg.norm(axis)
            assert abs(norm - 1.0) < 0.1, f"Spin axis not normalized: norm={norm}"


class TestOpticalFlowPipelineIntegration:
    """Integration tests for OpticalFlowPipeline."""

    @pytest.fixture
    def camera_matrix(self):
        return np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float64)

    @pytest.fixture
    def dist_coeffs(self):
        return np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)

    @pytest.fixture
    def pipeline(self, camera_matrix, dist_coeffs):
        from src.pipeline_optical import OpticalFlowPipeline
        return OpticalFlowPipeline(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            model_path="yolov8n.pt",
            confidence_threshold=0.25
        )

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes all components."""
        assert pipeline.detector is not None
        assert pipeline.ball_tracker is not None
        assert pipeline.rotation_estimator is not None
        assert pipeline.orientation_tracker is not None

    def test_multiple_frames_tracking(self, pipeline):
        """Test tracking across multiple frames."""
        frames = []
        for i in range(20):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 180

            # Moving ball
            x = int(200 + i * 15)
            y = int(300 + i * 5)
            radius = 60

            cv2.circle(frame, (x, y), radius, (120, 120, 120), -1)

            # Add pattern
            for angle in range(0, 360, 30):
                rad = np.radians(angle + i * 10)
                px = int(x + radius * 0.8 * np.cos(rad))
                py = int(y + radius * 0.8 * np.sin(rad))
                cv2.circle(frame, (px, py), 3, (60, 60, 60), -1)

            frames.append(frame)

        # Process sequence
        results = []
        for i, frame in enumerate(frames):
            # We need actual YOLO detection for this, so we'll mock
            # with approximate bounding boxes
            x = int(200 + i * 15)
            y = int(300 + i * 5)
            bbox = (x - radius, y - radius, x + radius, y + radius)

            with patch.object(pipeline.detector, 'detect', return_value={
                "detected": True,
                "bbox": bbox,
                "confidence": 0.85
            }):
                result = pipeline.process_frame(frame, timestamp=i * 0.033)
                results.append(result)

        # All should detect the ball
        detected_count = sum(1 for r in results if r["ball_detected"])
        assert detected_count > 0

    def test_pipeline_resilience_to_lighting_changes(self, pipeline):
        """Test pipeline handles lighting variations."""
        frames = []

        # Create frames with different brightness
        brightness_levels = [100, 150, 200, 180, 220, 140, 160]

        for i, brightness in enumerate(brightness_levels):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * brightness

            # Add ball at same position
            center = (640, 360)
            radius = 60

            cv2.circle(frame, center, radius, (brightness - 80, brightness - 80, brightness - 80), -1)

            frames.append((frame, i * 0.033))

        # Process all frames
        results = []
        for frame, timestamp in frames:
            with patch.object(pipeline.detector, 'detect', return_value={
                "detected": True,
                "bbox": (580, 300, 700, 420),
                "confidence": 0.9
            }):
                result = pipeline.process_frame(frame, timestamp=timestamp)
                results.append(result)

        # Check processing completed
        assert len(results) == len(brightness_levels)
