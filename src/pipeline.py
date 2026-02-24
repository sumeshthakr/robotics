"""Main pipeline for baseball orientation detection.

This module integrates all components of the baseball orientation detection system:
- Ball detection using YOLO
- Image undistortion
- Seam detection
- 3D orientation estimation
- Temporal tracking
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import cv2

from src.detection.ball_detector import BallDetector
from src.detection.ball_tracker import BallTracker
from src.preprocessing.undistort import undistort
from src.seams.edge_detector import detect_seams
from src.seams.seam_model import BaseballSeamModel
from src.estimation.pnp_solver import solve_orientation
from src.estimation.sphere_fitter import fit_circle
from src.tracking.orientation_tracker import OrientationTracker


class BaseballOrientationPipeline:
    """Complete pipeline for baseball orientation detection from video frames.

    The pipeline processes each frame through the following stages:
    1. Undistort the image using camera calibration
    2. Detect the baseball using YOLO
    3. Extract ROI around the detected ball
    4. Detect seams in the ROI
    5. Estimate 3D orientation using PnP
    6. Track orientation over time

    Example:
        >>> from src.utils.camera import load_camera_params
        >>> K, dist, _ = load_camera_params("config/camera.json")
        >>> pipeline = BaseballOrientationPipeline(camera_matrix=K, dist_coeffs=dist)
        >>> result = pipeline.process_frame(frame, timestamp=0.0)
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        ball_radius_mm: float = 37.0,
        confidence_threshold: float = 0.5,
        model_path: str = "yolov8n.pt"
    ):
        """Initialize the baseball orientation pipeline.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            ball_radius_mm: Radius of baseball in millimeters (default: 37.0)
            confidence_threshold: Detection confidence threshold (0-1)
            model_path: Path to YOLO model weights
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
        self.seam_model = BaseballSeamModel(radius=ball_radius_mm)
        self.orientation_tracker = OrientationTracker(window_size=10)

        # Frame counter
        self._frame_count = 0
        self._last_timestamp = None

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
            "timestamp": timestamp
        }

        # Step 1: Undistort image
        try:
            undistorted = undistort(image, self.camera_matrix, self.dist_coeffs)
        except Exception:
            # If undistortion fails, use original image
            undistorted = image

        # Step 2: Track baseball (uses detector + temporal tracking)
        track_result = self.ball_tracker.track(undistorted)
        if not track_result["detected"]:
            return result

        result["ball_detected"] = True
        result["bbox"] = track_result["bbox"]
        result["confidence"] = track_result["confidence"]
        result["tracking"] = track_result["tracking"]  # True if predicted, False if detected

        # Step 3: Extract ROI
        x1, y1, x2, y2 = track_result["bbox"]
        roi = undistorted[y1:y2, x1:x2]

        if roi.size == 0:
            return result

        # Step 4: Detect seams
        seam_result = detect_seams(roi)

        # Calculate minimum seam pixels based on ROI size
        roi_area = roi.shape[0] * roi.shape[1]
        min_seam_pixels = max(4, int(roi_area * 0.01))  # At least 1% of ROI area

        if seam_result["num_pixels"] < min_seam_pixels:
            # Not enough seam pixels for reliable orientation
            return result

        # Step 5: Fit circle to ball contour (for center estimation)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (should be the ball)
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5:
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                # Adjust to global coordinates
                global_cx = x1 + int(cx)
                global_cy = y1 + int(cy)

                # Step 6: Solve for orientation
                # Get 2D seam points in global coordinates
                seam_pixels = seam_result["seam_pixels"]
                global_seam_pixels = seam_pixels + np.array([x1, y1])

                # Get 3D seam model points - match number of detected points
                num_detected = len(global_seam_pixels)
                points_3d = self.seam_model.get_all_points()

                # Sample 3D points to match detected points (repeat if needed)
                if num_detected <= len(points_3d):
                    # Use subset of 3D points
                    indices = np.linspace(0, len(points_3d) - 1, num_detected, dtype=int)
                    points_3d_sampled = points_3d[indices]
                else:
                    # Repeat 3D points to match (or could use all with correspondence)
                    repeat_factor = (num_detected // len(points_3d)) + 1
                    points_3d_sampled = np.tile(points_3d, (repeat_factor, 1))[:num_detected]

                # Solve PnP
                if num_detected >= 4:
                    pnp_result = solve_orientation(
                        global_seam_pixels,
                        points_3d_sampled,
                        self.camera_matrix
                    )

                    if pnp_result["success"]:
                        R = pnp_result["rotation_matrix"]
                        tvec = pnp_result["tvec"]

                        # Update tracker
                        if timestamp is not None:
                            self.orientation_tracker.add_orientation(R, timestamp)

                        # Convert to quaternion
                        from src.estimation.pnp_solver import rotation_matrix_to_quaternion
                        quat = rotation_matrix_to_quaternion(R)

                        # Convert to Euler angles
                        from src.estimation.pnp_solver import rotation_matrix_to_euler
                        euler = rotation_matrix_to_euler(R)

                        result["orientation"] = {
                            "rotation_matrix": R,
                            "quaternion": quat,
                            "euler_angles": euler
                        }

                        # Get spin info
                        result["spin_rate"] = self.orientation_tracker.get_spin_rate()
                        result["spin_axis"] = self.orientation_tracker.get_spin_axis()

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
                vis_frame = self._save_visualization(frame, result, frame_idx)
                writer.write(vis_frame)

            frame_idx += 1

        # Clean up
        cap.release()
        if writer is not None:
            writer.release()

        # Calculate statistics
        spin_rates = [r["spin_rate"] for r in results if r["spin_rate"] is not None]
        avg_spin_rate = np.mean(spin_rates) if spin_rates else None

        return {
            "total_frames": frame_idx,
            "detections": results,
            "average_spin_rate": avg_spin_rate,
            "fps": fps
        }

    def _save_visualization(
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
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw orientation info if available
        if result["orientation"] is not None:
            info_text = f"Frame: {frame_idx}"
            if result["spin_rate"] is not None:
                info_text += f" | Spin: {result['spin_rate']:.1f} RPM"

            cv2.putText(
                vis_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return vis_frame

    def reset(self):
        """Reset pipeline state (frame counter and tracker history)."""
        self._frame_count = 0
        self._last_timestamp = None
        self.tracker = OrientationTracker(window_size=10)
