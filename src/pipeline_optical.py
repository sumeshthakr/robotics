"""Optical flow based pipeline for baseball orientation detection.

This module provides an alternative to the seam-based pipeline that uses
optical flow to track rotation instead of seam detection + PnP.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import cv2

from src.detection.ball_detector import BallDetector
from src.detection.ball_tracker import BallTracker
from src.preprocessing.undistort import undistort
from src.optical_flow.rotation_estimator import RotationEstimator
from src.tracking.orientation_tracker import OrientationTracker


class OpticalFlowPipeline:
    """Pipeline for baseball orientation detection using optical flow.

    This pipeline processes each frame through the following stages:
    1. Undistort the image using camera calibration
    2. Detect the baseball using YOLO
    3. Extract ROI around the detected ball
    4. Track feature points using optical flow
    5. Estimate rotation from optical flow patterns
    6. Track orientation over time for smoothing

    This approach is useful when seam detection is difficult (e.g., poor lighting,
    low contrast, worn ball).

    Example:
        >>> from src.utils.camera import load_camera_params
        >>> K, dist, _ = load_camera_params("config/camera.json")
        >>> pipeline = OpticalFlowPipeline(camera_matrix=K, dist_coeffs=dist)
        >>> result = pipeline.process_frame(frame, timestamp=0.0)
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        ball_radius_mm: float = 37.0,
        confidence_threshold: float = 0.5,
        model_path: str = "yolov8n.pt",
        max_corners: int = 50,
        min_flow_threshold: float = 0.5,
        max_flow_threshold: float = 30.0,
    ):
        """Initialize the optical flow pipeline.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            ball_radius_mm: Radius of baseball in millimeters (default: 37.0)
            confidence_threshold: Detection confidence threshold (0-1)
            model_path: Path to YOLO model weights
            max_corners: Maximum corners for optical flow tracking
            min_flow_threshold: Minimum flow magnitude (pixels)
            max_flow_threshold: Maximum flow magnitude (pixels)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ball_radius_mm = ball_radius_mm
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.detector = BallDetector(
            model_name=model_path,
            confidence_threshold=confidence_threshold
        )
        self.ball_tracker = BallTracker(
            detector=self.detector,
            max_lost_frames=10,
            iou_threshold=0.3
        )
        self.rotation_estimator = RotationEstimator(
            camera_matrix=camera_matrix,
            ball_radius_mm=ball_radius_mm,
            max_corners=max_corners,
            min_flow_threshold=min_flow_threshold,
            max_flow_threshold=max_flow_threshold,
        )
        self.orientation_tracker = OrientationTracker(window_size=10)

        # Frame counter
        self._frame_count = 0
        self._last_timestamp = None

        # State tracking
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    def reset(self):
        """Reset pipeline state."""
        self._frame_count = 0
        self._last_timestamp = None
        self._consecutive_failures = 0
        self.rotation_estimator.reset()

    def process_frame(
        self,
        image: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process a single frame and extract baseball orientation.

        Args:
            image: Input frame (H, W, 3) BGR format
            timestamp: Frame timestamp in seconds (optional)

        Returns:
            Dictionary containing:
                - ball_detected: bool, whether ball was found
                - bbox: (x1, y1, x2, y2) or None
                - confidence: detection confidence or None
                - orientation: dict with rotation_matrix, quaternion, euler_angles or None
                - spin_rate: RPM or None
                - spin_axis: 3D unit vector or None
                - frame_number: int
                - timestamp: float or None
                - flow_confidence: confidence of flow estimate
        """
        self._frame_count += 1

        # Default result structure
        result = {
            "ball_detected": False,
            "bbox": None,
            "confidence": None,
            "orientation": None,
            "spin_rate": None,
            "spin_axis": None,
            "frame_number": self._frame_count,
            "timestamp": timestamp,
            "flow_confidence": None
        }

        # Step 1: Undistort image
        try:
            undistorted = undistort(image, self.camera_matrix, self.dist_coeffs)
        except Exception:
            # If undistortion fails, use original image
            undistorted = image

        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        # Step 2: Track baseball (uses detector + temporal tracking)
        track_result = self.ball_tracker.track(undistorted)
        if not track_result["detected"]:
            self._consecutive_failures += 1
            if self._consecutive_failures > self._max_consecutive_failures:
                self.rotation_estimator.reset()
            return result

        result["ball_detected"] = True
        result["bbox"] = track_result["bbox"]
        result["confidence"] = track_result["confidence"]
        result["tracking"] = track_result["tracking"]  # True if predicted, False if detected

        # Reset consecutive failures counter
        self._consecutive_failures = 0

        # Step 3: Estimate rotation from optical flow
        bbox = track_result["bbox"]
        rotation_result = self.rotation_estimator.estimate_rotation(
            gray, bbox, timestamp
        )

        if rotation_result is None:
            # No rotation estimate yet (first frame or insufficient tracks)
            # Try to return smoothed result from history
            smoothed = self.rotation_estimator.get_smoothed_rotation()
            if smoothed is not None:
                result["spin_rate"] = smoothed.get("spin_rate_rpm")
                result["spin_axis"] = smoothed.get("spin_axis")
                result["flow_confidence"] = smoothed.get("confidence")
            return result

        # Step 4: Extract orientation information
        R = rotation_result["rotation_matrix"]
        spin_axis = rotation_result["spin_axis"]
        spin_rate_rps = rotation_result.get("spin_rate_rps", 0)
        confidence = rotation_result.get("confidence", 0.5)

        # Update orientation tracker with the rotation matrix
        if timestamp is not None:
            self.orientation_tracker.add_orientation(R, timestamp)

        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w] scalar-last
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        # Convert to Euler angles
        euler = rot.as_euler('xyz')  # [roll, pitch, yaw]

        result["orientation"] = {
            "rotation_matrix": R,
            "quaternion": quat_wxyz,
            "euler_angles": euler
        }

        # Use orientation tracker for more accurate spin rate
        tracker_spin_rate = self.orientation_tracker.get_spin_rate()
        tracker_spin_axis = self.orientation_tracker.get_spin_axis()

        result["spin_rate"] = tracker_spin_rate if tracker_spin_rate is not None else (spin_rate_rps * 60)
        result["spin_axis"] = tracker_spin_axis if tracker_spin_axis is not None else spin_axis
        result["flow_confidence"] = confidence

        return result

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """Process a video file and extract baseball orientations.

        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video with visualizations
            visualize: If True, generate visualization overlays

        Returns:
            Dictionary containing:
                - total_frames: int
                - detections: list of per-frame results
                - average_spin_rate: float or None
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output video if needed
        writer = None
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_idx = 0
        timestamps = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else None

            # Process frame
            result = self.process_frame(frame, timestamp=timestamp)
            results.append(result)
            timestamps.append(timestamp)

            # Write visualization if enabled
            if visualize and writer is not None:
                vis_frame = self._create_visualization(frame, result, frame_idx)
                writer.write(vis_frame)

            frame_idx += 1

        # Clean up
        cap.release()
        if writer is not None:
            writer.release()

        # Calculate statistics
        spin_rates = [r["spin_rate"] for r in results if r["spin_rate"] is not None]
        avg_spin_rate = np.mean(spin_rates) if spin_rates else None

        confidences = [r["flow_confidence"] for r in results if r["flow_confidence"] is not None]
        avg_confidence = np.mean(confidences) if confidences else None

        return {
            "total_frames": frame_idx,
            "detections": results,
            "average_spin_rate": avg_spin_rate,
            "average_confidence": avg_confidence,
            "fps": fps
        }

    def _create_visualization(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        frame_idx: int
    ) -> np.ndarray:
        """Create visualization overlay on frame.

        Args:
            frame: Original frame
            result: Processing result dictionary
            frame_idx: Frame number

        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()

        # Draw bounding box if ball detected
        if result["ball_detected"] and result["bbox"] is not None:
            x1, y1, x2, y2 = result["bbox"]

            # Color based on tracking status
            color = (255, 0, 0) if result.get("tracking", False) else (0, 255, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw confidence text
            conf = result.get("confidence", 0)
            cv2.putText(
                vis_frame,
                f"Conf: {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # Draw orientation info if available
        info_y = 30
        info_texts = [f"Frame: {frame_idx}"]

        if result["flow_confidence"] is not None:
            info_texts.append(f"Flow Conf: {result['flow_confidence']:.2f}")

        if result["spin_rate"] is not None:
            info_texts.append(f"Spin: {result['spin_rate']:.1f} RPM")

        if result["spin_axis"] is not None:
            axis = result["spin_axis"]
            info_texts.append(f"Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")

        # Draw method indicator
        cv2.putText(
            vis_frame,
            "Method: Optical Flow",
            (10, info_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        for i, text in enumerate(info_texts):
            cv2.putText(
                vis_frame,
                text,
                (10, info_y + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Draw spin axis visualization if available
        if result["spin_axis"] is not None and result["bbox"] is not None:
            x1, y1, x2, y2 = result["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = min(x2 - x1, y2 - y1) // 2

            axis = result["spin_axis"]

            # Project 3D axis to 2D (simple projection)
            axis_length = radius * 1.5
            axis_end_x = int(cx + axis[0] * axis_length)
            axis_end_y = int(cy + axis[1] * axis_length)

            # Draw axis line
            cv2.arrowedLine(
                vis_frame,
                (cx, cy),
                (axis_end_x, axis_end_y),
                (0, 0, 255),
                3,
                tipLength=0.3
            )

        return vis_frame
