"""Rotation estimation from optical flow for baseball orientation detection.

This module implements rotation estimation by tracking feature points on the
baseball surface between consecutive frames using sparse optical flow.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any


class RotationEstimator:
    """Estimate ball rotation from optical flow.

    This class tracks feature points on the baseball surface across consecutive
    frames and estimates the rotation axis and spin rate from the optical flow field.

    The key idea is that for a rotating sphere:
    - Points near the equator move fastest
    - Points near the poles move slowest
    - Flow vectors form a pattern around the rotation axis

    The rotation axis can be found by analyzing the flow field structure.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        ball_radius_mm: float = 37.0,
        max_corners: int = 50,
        quality_level: float = 0.01,
        min_distance: int = 7,
        win_size: int = 15,
        max_level: int = 3,
        min_flow_threshold: float = 0.5,
        max_flow_threshold: float = 30.0,
    ):
        """Initialize the rotation estimator.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            ball_radius_mm: Radius of baseball in millimeters (default: 37.0)
            max_corners: Maximum number of corners to detect
            quality_level: Quality level for corner detection (0-1)
            min_distance: Minimum distance between corners
            win_size: Window size for optical flow
            max_level: Max pyramid level for optical flow
            min_flow_threshold: Minimum flow magnitude to consider valid (pixels)
            max_flow_threshold: Maximum flow magnitude before rejecting (pixels)
        """
        self.camera_matrix = camera_matrix
        self.ball_radius = ball_radius_mm

        # Feature detection parameters
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        # Optical flow parameters
        self.win_size = (win_size, win_size)
        self.max_level = max_level

        # Flow validation thresholds
        self.min_flow_threshold = min_flow_threshold
        self.max_flow_threshold = max_flow_threshold

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # State tracking
        self.prev_points = None  # Previous frame feature points (in ROI coordinates)
        self.prev_gray = None  # Previous grayscale ROI
        self.prev_bbox = None  # Previous bounding box (x1, y1, x2, y2)
        self.flow_history = []  # History of flow vectors for smoothing

        # Estimated ball radius in pixels (updated per frame)
        self.ball_radius_px = None

        # Rotation state
        self.current_rotation_matrix = np.eye(3)
        self.accumulated_rotation = np.eye(3)

    def reset(self):
        """Reset the estimator state."""
        self.prev_points = None
        self.prev_gray = None
        self.prev_bbox = None
        self.flow_history = []
        self.ball_radius_px = None
        self.current_rotation_matrix = np.eye(3)
        self.accumulated_rotation = np.eye(3)

    def _detect_features(
        self,
        gray_roi: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Detect good features to track in the ball ROI.

        Args:
            gray_roi: Grayscale ROI image
            mask: Optional mask limiting where to detect features

        Returns:
            Array of corner points (N, 1, 2)
        """
        # Use goodFeaturesToTrack for corner detection
        corners = cv2.goodFeaturesToTrack(
            gray_roi,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04
        )

        if corners is None:
            return np.array([])

        return corners

    def _create_circular_mask(
        self,
        roi_shape: Tuple[int, int],
        center: Tuple[int, int],
        radius: int
    ) -> np.ndarray:
        """Create a circular mask for feature detection.

        Args:
            roi_shape: Shape of ROI (height, width)
            center: Center of circle (x, y)
            radius: Radius of circle

        Returns:
            Binary mask
        """
        h, w = roi_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.circle(mask, center, radius, 255, -1)

        return mask

    def _compute_flow(
        self,
        curr_gray: np.ndarray,
        prev_gray: np.ndarray,
        prev_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute sparse optical flow using Lucas-Kanade.

        Args:
            curr_gray: Current grayscale frame
            prev_gray: Previous grayscale frame
            prev_points: Points to track from previous frame (N, 1, 2)

        Returns:
            Tuple of (curr_points, status, error) where:
            - curr_points: Tracked points in current frame (N, 1, 2)
            - status: Track status (1 = tracked, 0 = lost)
            - error: Tracking error for each point
        """
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_points,
            None,
            **self.lk_params
        )

        return curr_points, status, error

    def _filter_valid_flow(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        status: np.ndarray,
        roi_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter valid optical flow tracks.

        Args:
            prev_points: Previous frame points (N, 1, 2)
            curr_points: Current frame points (N, 1, 2)
            status: Track status array
            roi_shape: Shape of ROI (height, width)

        Returns:
            Tuple of (filtered_prev, filtered_curr) arrays (M, 2)
        """
        h, w = roi_shape

        # Initialize with all points
        valid_prev = []
        valid_curr = []

        for i in range(len(prev_points)):
            # Check track status
            if status[i] != 1:
                continue

            p1 = prev_points[i][0]
            p2 = curr_points[i][0]

            # Check if current point is within ROI bounds
            if not (0 <= p2[0] < w and 0 <= p2[1] < h):
                continue

            # Check if previous point is within ROI bounds
            if not (0 <= p1[0] < w and 0 <= p1[1] < h):
                continue

            # Compute flow magnitude
            flow_vec = p2 - p1
            flow_mag = np.linalg.norm(flow_vec)

            # Filter by flow magnitude thresholds
            if flow_mag < self.min_flow_threshold or flow_mag > self.max_flow_threshold:
                continue

            valid_prev.append(p1)
            valid_curr.append(p2)

        if len(valid_prev) == 0:
            return np.array([]), np.array([])

        return np.array(valid_prev), np.array(valid_curr)

    def _estimate_rotation_from_flow(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        roi_center: Tuple[float, float],
        ball_radius_px: float
    ) -> Optional[Dict[str, Any]]:
        """Estimate rotation from optical flow vectors.

        For a rotating sphere observed by a camera:
        - The flow field has a specific structure determined by the rotation axis
        - Points move perpendicular to the axis and their distance from it
        - The angular velocity is related to the tangential velocity and radius

        Args:
            prev_points: Previous frame points (N, 2)
            curr_points: Current frame points (N, 2)
            roi_center: Center of ball ROI (cx, cy)
            ball_radius_px: Ball radius in pixels

        Returns:
            Dictionary with rotation_matrix, spin_axis, spin_rate, confidence
            or None if estimation fails
        """
        if len(prev_points) < 3:
            return None

        # Compute flow vectors
        flow_vectors = curr_points - prev_points  # (N, 2)

        # Compute distances from ROI center
        center = np.array(roi_center)
        prev_from_center = prev_points - center  # (N, 2)
        curr_from_center = curr_points - center  # (N, 2)

        # For each point, estimate the 3D position assuming sphere
        # We need to estimate the rotation axis and angular velocity

        # Method: RANSAC-based approach to find rotation axis
        # The rotation axis is orthogonal to the dominant flow pattern

        best_axis = None
        best_angular_velocity = 0
        best_inliers = 0
        max_iterations = 50
        inlier_threshold = 5.0  # pixels

        for _ in range(max_iterations):
            # Randomly select 2 points to define a candidate rotation axis
            if len(prev_points) < 2:
                break

            idx = np.random.choice(len(prev_points), 2, replace=False)
            p1 = prev_points[idx[0]]
            p2 = prev_points[idx[1]]
            f1 = flow_vectors[idx[0]]
            f2 = flow_vectors[idx[1]]

            # Candidate axis direction (perpendicular to both position and flow)
            # For pure rotation: v = omega x r
            # So omega is parallel to r x v

            r1 = p1 - center
            omega1 = np.cross(np.append(r1, 0), np.append(f1, 0))[:2]

            r2 = p2 - center
            omega2 = np.cross(np.append(r2, 0), np.append(f2, 0))[:2]

            # Use average direction
            if np.linalg.norm(omega1) > 1e-6 and np.linalg.norm(omega2) > 1e-6:
                omega_dir = (omega1 + omega2)
                omega_dir = omega_dir / (np.linalg.norm(omega_dir) + 1e-6)
            else:
                continue

            # Count inliers
            inliers = 0
            angular_vel_sum = 0

            for i in range(len(prev_points)):
                r = prev_points[i] - center
                f = flow_vectors[i]

                # Expected flow for rotation about axis
                # v_expected = omega x r
                omega_vec = np.append(omega_dir * np.linalg.norm(f) / (np.linalg.norm(r) + 1e-6), 0)
                expected_flow = np.cross(omega_vec, np.append(r, 0))[:2]

                # Residual
                residual = np.linalg.norm(f - expected_flow)

                if residual < inlier_threshold:
                    inliers += 1
                    # Estimate angular velocity from this point
                    dist_from_axis = np.linalg.norm(r - omega_dir * np.dot(r, omega_dir))
                    if dist_from_axis > 1e-6:
                        ang_vel = np.linalg.norm(f) / dist_from_axis
                        angular_vel_sum += ang_vel

            if inliers > best_inliers:
                best_inliers = inliers
                best_axis = omega_dir
                if inliers > 0:
                    best_angular_velocity = angular_vel_sum / inliers

        if best_inliers < 3 or best_axis is None:
            return None

        # Normalize axis
        axis_2d = best_axis / (np.linalg.norm(best_axis) + 1e-6)

        # Extend to 3D (assuming rotation axis lies in the image plane)
        # In practice, the axis could have a z-component too
        # For simplicity, we'll use the 2D axis and add a small z component
        axis_3d = np.array([axis_2d[0], axis_2d[1], 0.1])
        axis_3d = axis_3d / np.linalg.norm(axis_3d)

        # Compute rotation angle from angular velocity
        rotation_angle = best_angular_velocity  # radians per frame

        # Create rotation matrix using axis-angle representation
        # Rodrigues' formula: R = I + sin(theta) * [K] + (1-cos(theta)) * [K]^2
        # where [K] is the skew-symmetric matrix of the axis
        K = np.array([
            [0, -axis_3d[2], axis_3d[1]],
            [axis_3d[2], 0, -axis_3d[0]],
            [-axis_3d[1], axis_3d[0], 0]
        ])

        sin_theta = np.sin(rotation_angle)
        cos_theta = np.cos(rotation_angle)

        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

        # Compute confidence based on number of inliers
        confidence = min(1.0, best_inliers / len(prev_points))

        return {
            "rotation_matrix": R,
            "spin_axis": axis_3d,
            "spin_rate_rps": best_angular_velocity / (2 * np.pi),  # revolutions per second
            "angular_velocity": best_angular_velocity,  # radians per second (per frame)
            "confidence": confidence,
            "num_inliers": best_inliers,
            "num_points": len(prev_points)
        }

    def _estimate_rotation_perspective(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        roi_center: Tuple[float, float],
        ball_radius_px: float
    ) -> Optional[Dict[str, Any]]:
        """Alternative method: Estimate rotation using perspective geometry.

        This method models the baseball as a sphere and computes the essential
        matrix from the point correspondences, then extracts the rotation.

        Args:
            prev_points: Previous frame points (N, 2)
            curr_points: Current frame points (N, 2)
            roi_center: Center of ball ROI (cx, cy)
            ball_radius_px: Ball radius in pixels

        Returns:
            Dictionary with rotation_matrix, spin_axis, spin_rate, confidence
            or None if estimation fails
        """
        if len(prev_points) < 8:
            return None

        # Normalize points to camera coordinates
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Convert to normalized image coordinates
        prev_norm = (prev_points - np.array([cx, cy])) / np.array([fx, fy])
        curr_norm = (curr_points - np.array([cx, cy])) / np.array([fx, fy])

        # Compute essential matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            prev_norm,
            curr_norm,
            np.eye(3),  # Use identity since we already normalized
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None

        # Recover rotation from essential matrix
        _, R, _, _ = cv2.recoverPose(E, prev_norm, curr_norm)

        # Convert to axis-angle
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(R)
        rotvec = rot.as_rotvec()

        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            axis = np.array([0, 0, 1])
        else:
            axis = rotvec / angle

        # Compute confidence based on inlier ratio
        inlier_ratio = np.mean(mask) if mask is not None else 0.5

        return {
            "rotation_matrix": R,
            "spin_axis": axis,
            "spin_rate_rps": angle / (2 * np.pi),  # revolutions per frame
            "angular_velocity": angle,  # radians per frame
            "confidence": inlier_ratio,
            "num_inliers": int(np.sum(mask)) if mask is not None else len(prev_points),
            "num_points": len(prev_points)
        }

    def estimate_rotation(
        self,
        frame_gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
        timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Estimate rotation from optical flow.

        Args:
            frame_gray: Full grayscale frame
            bbox: Ball bounding box (x1, y1, x2, y2)
            timestamp: Frame timestamp in seconds

        Returns:
            dict with rotation_matrix, spin_rate, spin_axis, or None
        """
        x1, y1, x2, y2 = bbox
        roi = frame_gray[y1:y2, x1:x2]

        if roi.size == 0:
            self.reset()
            return None

        roi_h, roi_w = roi.shape[:2]

        # Update ball radius in pixels
        self.ball_radius_px = min(roi_w, roi_h) / 2

        # Create circular mask for feature detection
        roi_center = (roi_w // 2, roi_h // 2)
        mask = self._create_circular_mask((roi_h, roi_w), roi_center, int(self.ball_radius_px * 0.9))

        # Detect features in current ROI
        curr_points_roi = self._detect_features(roi, mask=mask)

        if len(curr_points_roi) < 4:
            self.reset()
            return None

        # If we have previous frame, compute optical flow
        if self.prev_gray is not None and self.prev_points is not None:
            # Check if ROI size is similar
            prev_h, prev_w = self.prev_gray.shape[:2]

            # Resize current ROI to match previous if sizes differ
            roi_resized = roi
            scale_factor = 1.0

            if roi_h != prev_h or roi_w != prev_w:
                roi_resized = cv2.resize(roi, (prev_w, prev_h))
                scale_factor = prev_w / roi_w
                roi_h, roi_w = prev_h, prev_w

            # Update current points for scale factor
            curr_points_scaled = curr_points_roi.copy()
            if scale_factor != 1.0:
                curr_points_scaled[:, 0, :] = curr_points_scaled[:, 0, :] * scale_factor

            if abs(roi_h - prev_h) > roi_h * 0.5 or abs(roi_w - prev_w) > roi_w * 0.5:
                # ROI changed too much, reset
                self.prev_gray = roi
                self.prev_points = curr_points_roi
                self.prev_bbox = bbox
                return None

            # Compute optical flow
            curr_points_lk, status, error = self._compute_flow(
                roi_resized, self.prev_gray, self.prev_points
            )

            # Filter valid tracks
            valid_prev, valid_curr = self._filter_valid_flow(
                self.prev_points, curr_points_lk, status, (roi_h, roi_w)
            )

            # Estimate rotation from flow
            result = None

            if len(valid_prev) >= 4:
                # Try both methods and pick the one with higher confidence
                result1 = self._estimate_rotation_from_flow(
                    valid_prev, valid_curr, roi_center, self.ball_radius_px
                )

                result2 = self._estimate_rotation_perspective(
                    valid_prev, valid_curr, roi_center, self.ball_radius_px
                )

                # Choose the result with higher confidence/more inliers
                if result1 is not None and result2 is not None:
                    if result1["num_inliers"] >= result2["num_inliers"]:
                        result = result1
                    else:
                        result = result2
                elif result1 is not None:
                    result = result1
                elif result2 is not None:
                    result = result2

            # Update state for next frame
            self.prev_gray = roi
            self.prev_points = curr_points_roi
            self.prev_bbox = bbox

            if result is not None:
                # Update accumulated rotation
                self.current_rotation_matrix = result["rotation_matrix"]
                self.accumulated_rotation = result["rotation_matrix"] @ self.accumulated_rotation

                # Add to flow history
                self.flow_history.append(result)
                if len(self.flow_history) > 10:
                    self.flow_history.pop(0)

                return result
        else:
            # First frame - initialize
            self.prev_gray = roi
            self.prev_points = curr_points_roi
            self.prev_bbox = bbox

        return None

    def get_smoothed_rotation(self) -> Optional[Dict[str, Any]]:
        """Get smoothed rotation estimate from history.

        Returns:
            Dictionary with smoothed rotation_matrix, spin_axis, spin_rate
        """
        if len(self.flow_history) < 2:
            return None

        # Average the recent rotation matrices
        # Use temporal averaging for spin rate
        spin_rates = [r["spin_rate_rps"] for r in self.flow_history if r.get("spin_rate_rps")]
        if not spin_rates:
            return None

        avg_spin_rate_rps = np.mean(spin_rates)

        # Average the rotation axes
        axes = np.array([r["spin_axis"] for r in self.flow_history if r.get("spin_axis") is not None])
        if len(axes) == 0:
            return None

        # Average axes and normalize
        avg_axis = np.mean(axes, axis=0)
        avg_axis = avg_axis / np.linalg.norm(avg_axis)

        # Get average confidence
        confidences = [r.get("confidence", 0.5) for r in self.flow_history]
        avg_confidence = np.mean(confidences)

        return {
            "rotation_matrix": self.accumulated_rotation,
            "spin_axis": avg_axis,
            "spin_rate_rps": avg_spin_rate_rps,
            "spin_rate_rpm": avg_spin_rate_rps * 60,
            "confidence": avg_confidence
        }
