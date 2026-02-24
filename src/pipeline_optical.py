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

        # Store tracked features for visualization
        if rotation_result is not None and "tracked_features" in rotation_result:
            result["tracked_features"] = rotation_result["tracked_features"]

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
        """Create comprehensive visualization overlay for optical flow approach.

        Shows:
        - Ball bounding box
        - Tracked features (corners)
        - Flow vectors
        - Spin axis arrow
        - Orientation info text
        - Real-world coordinates

        Args:
            frame: Original frame
            result: Processing result dictionary
            frame_idx: Frame number

        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_y = 25
        line_height = 22

        # ========== HEADER INFO ==========
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, text_y),
                    font, font_scale, (255, 255, 255), thickness)
        text_y += line_height

        # ========== BALL DETECTION ==========
        if result["ball_detected"] and result["bbox"] is not None:
            x1, y1, x2, y2 = result["bbox"]
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2

            # Draw bounding box
            color = (255, 255, 0) if not result.get("tracking", False) else (0, 255, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw ball center
            cv2.circle(vis_frame, (bbox_cx, bbox_cy), 5, (255, 0, 0), -1)
            cv2.circle(vis_frame, (bbox_cx, bbox_cy), 3, (255, 255, 255), 1)

            # Pixel coordinates
            cv2.putText(vis_frame, f"Ball Center (px): ({bbox_cx}, {bbox_cy})",
                        (10, text_y), font, font_scale * 0.5, (0, 255, 0), 1)
            text_y += line_height

            # ========== FEATURE TRACKING VISUALIZATION ==========
            if result.get("tracked_features") is not None:
                tracked = result["tracked_features"]
                # Draw flow vectors
                for p1, p2 in zip(tracked["prev_points"], tracked["curr_points"]):
                    p1 = tuple(p1.astype(int))
                    p2 = tuple(p2.astype(int))
                    # Draw flow vector
                    cv2.arrowedLine(vis_frame, p1, p2, (255, 255, 0), 1, tipLength=0.3)

                cv2.putText(vis_frame, f"Tracked Features: {len(tracked['prev_points'])}",
                            (10, text_y), font, font_scale * 0.5, (255, 255, 0), 1)
                text_y += line_height

            # ========== ORIENTATION INFO ==========
            if result["orientation"] is not None:
                orient = result["orientation"]
                R = orient["rotation_matrix"]
                quat = orient["quaternion"]
                euler = orient["euler_angles"]

                # Quaternion
                cv2.putText(vis_frame, f"Quat: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]",
                            (10, text_y), font, font_scale * 0.5, (255, 200, 0), 1)
                text_y += line_height

                # Euler angles (in degrees)
                euler_deg = np.degrees(euler)
                cv2.putText(vis_frame, f"Euler: [{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}] deg",
                            (10, text_y), font, font_scale * 0.5, (255, 200, 0), 1)
                text_y += line_height

            # ========== SPIN RATE ==========
            if result["spin_rate"] is not None:
                spin_text = f"Spin Rate: {result['spin_rate']:.1f} RPM"
                cv2.putText(vis_frame, spin_text, (10, text_y),
                            font, font_scale, (0, 255, 255), thickness)
                text_y += line_height

            # ========== SPIN AXIS ==========
            if result["spin_axis"] is not None:
                axis = result["spin_axis"]

                # Draw spin axis arrow from ball center
                axis_length = 60  # pixels
                end_x = int(bbox_cx + axis[0] * axis_length)
                end_y = int(bbox_cy + axis[1] * axis_length)

                cv2.arrowedLine(vis_frame, (bbox_cx, bbox_cy), (end_x, end_y),
                               (255, 0, 255), 4, tipLength=0.3)

                # Axis label
                cv2.putText(vis_frame, f"Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]",
                            (bbox_cx + 10, bbox_cy - 10), font, font_scale * 0.5,
                            (255, 0, 255), 1)

            # ========== REAL-WORLD COORDINATES ==========
            if self.camera_matrix is not None:
                ball_radius_mm = self.ball_radius_mm
                focal_length = self.camera_matrix[0, 0]
                ball_radius_px = min(bbox_w, bbox_h) / 2

                if ball_radius_px > 0:
                    depth_mm = (focal_length * ball_radius_mm) / ball_radius_px

                    cx_cam = self.camera_matrix[0, 2]
                    cy_cam = self.camera_matrix[1, 2]

                    real_x = (bbox_cx - cx_cam) * depth_mm / focal_length
                    real_y = (bbox_cy - cy_cam) * depth_mm / focal_length
                    real_z = depth_mm

                    cv2.putText(vis_frame, f"Position (mm): X={real_x:.0f} Y={real_y:.0f} Z={real_z:.0f}",
                                (10, text_y), font, font_scale * 0.5, (255, 200, 255), 1)
                    text_y += line_height

            # ========== CONFIDENCE ==========
            if result.get("confidence") is not None:
                conf = result["confidence"]
                conf_color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                cv2.putText(vis_frame, f"Conf: {conf:.2f}",
                            (10, text_y), font, font_scale * 0.5, conf_color, 1)
                text_y += line_height

            if result.get("flow_confidence") is not None:
                flow_conf = result["flow_confidence"]
                cv2.putText(vis_frame, f"Flow Conf: {flow_conf:.2f}",
                            (10, text_y), font, font_scale * 0.5, (0, 200, 255), 1)
                text_y += line_height

            # Tracking status
            if result.get("tracking") is not None:
                track_status = "PREDICTED" if result["tracking"] else "DETECTED"
                track_color = (255, 255, 0) if result["tracking"] else (0, 255, 0)
                cv2.putText(vis_frame, f"Status: {track_status}",
                            (10, text_y), font, font_scale * 0.5, track_color, 1)

        else:
            # No ball detected
            cv2.putText(vis_frame, "NO BALL DETECTED", (10, text_y),
                        font, font_scale, (0, 0, 255), thickness)

        # ========== METHOD LABEL ==========
        cv2.putText(vis_frame, "Method: Optical Flow",
                    (10, h - 15), font, font_scale * 0.7, (255, 255, 0), 2)

        # ========== LEGEND ==========
        legend_y = h - 20
        cv2.rectangle(vis_frame, (5, legend_y - 75), (220, legend_y), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (5, legend_y - 75), (220, legend_y), (255, 255, 255), 1)

        legend_y -= 55
        cv2.putText(vis_frame, "LEGEND:", (10, legend_y), font, font_scale * 0.5,
                    (255, 255, 255), 1)
        legend_y += 15

        # Ball box
        cv2.rectangle(vis_frame, (10, legend_y - 5), (30, legend_y + 5), (0, 255, 0), -1)
        cv2.putText(vis_frame, "Ball Box", (40, legend_y), font, font_scale * 0.4,
                    (255, 255, 255), 1)
        legend_y += 15

        # Flow vectors
        cv2.arrowedLine(vis_frame, (15, legend_y), (35, legend_y), (255, 255, 0), 2)
        cv2.putText(vis_frame, "Flow Vector", (40, legend_y), font, font_scale * 0.4,
                    (255, 255, 255), 1)
        legend_y += 15

        # Spin axis
        cv2.arrowedLine(vis_frame, (15, legend_y), (35, legend_y), (255, 0, 255), 2)
        cv2.putText(vis_frame, "Spin Axis", (40, legend_y), font, font_scale * 0.4,
                    (255, 255, 255), 1)

        return vis_frame
