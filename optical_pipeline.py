"""Optical flow based baseball orientation detection pipeline.

APPROACH:
    Track corner features on the ball surface between consecutive frames
    using Lucas-Kanade optical flow, then estimate the ball's rotation
    matrix from the flow pattern.

KEY PHYSICS:
    For a rotating sphere, the velocity v of a surface point at position r
    from the center, due to angular velocity ω, is:

        v = ω × r    (cross product)

    By observing the flow pattern, we can recover the rotation axis and angle.

PIPELINE (per frame):
    1. Undistort image
    2. Detect baseball with YOLO
    3. Detect corner features on the ball surface
    4. Track features to next frame using Lucas-Kanade optical flow
    5. Solve for rotation from flow using least squares
    6. Accumulate rotation for orientation tracking
"""

import os
import numpy as np
import cv2

from camera import undistort
from detector import BallDetector, BallTracker
from orientation import rotation_to_quaternion, rotation_to_euler


# ============================================================
# Rotation Estimator (from Optical Flow)
# ============================================================

class RotationEstimator:
    """Estimate ball rotation from optical flow between consecutive frames.

    Algorithm:
        1. Detect corners (features) inside the ball ROI
        2. Track them to the next frame with Lucas-Kanade optical flow
        3. Filter out bad/invalid tracks
        4. Lift 2D positions to 3D on a sphere: rz = sqrt(R² - rx² - ry²)
        5. Solve the linear system v = ω × r for angular velocity ω
        6. Convert ω to a rotation matrix via Rodrigues formula
    """

    def __init__(self, camera_matrix, ball_radius_mm=37.0,
                 max_corners=50, min_flow=0.5, max_flow=30.0):
        """
        Args:
            camera_matrix:  3x3 camera intrinsics
            ball_radius_mm: Baseball radius in mm
            max_corners:    Max feature points to track
            min_flow:       Minimum flow magnitude to accept (pixels)
            max_flow:       Maximum flow magnitude to accept (pixels)
        """
        self.camera_matrix = camera_matrix
        self.ball_radius = ball_radius_mm
        self.max_corners = max_corners
        self.min_flow = min_flow
        self.max_flow = max_flow

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # State from previous frame
        self.prev_gray = None
        self.prev_points = None

        # Accumulated rotation for orientation tracking
        self.accumulated_rotation = np.eye(3)

    def reset(self):
        """Clear all tracking state."""
        self.prev_gray = None
        self.prev_points = None
        self.accumulated_rotation = np.eye(3)

    def estimate_rotation(self, frame_gray, bbox, timestamp=None):
        """Estimate rotation from optical flow in the current frame.

        Args:
            frame_gray: Full grayscale frame
            bbox:       Ball bounding box (x1, y1, x2, y2)
            timestamp:  Unused, kept for API compatibility

        Returns:
            dict with rotation_matrix, spin_axis, confidence,
            tracked_features — or None if estimation isn't possible yet
        """
        x1, y1, x2, y2 = bbox
        roi = frame_gray[y1:y2, x1:x2]

        if roi.size == 0:
            self.reset()
            return None

        roi_h, roi_w = roi.shape[:2]
        ball_radius_px = min(roi_w, roi_h) / 2
        roi_center = (roi_w // 2, roi_h // 2)

        # Circular mask — only detect features ON the ball
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(mask, roi_center, int(ball_radius_px * 0.9), 255, -1)

        curr_points = cv2.goodFeaturesToTrack(
            roi, maxCorners=self.max_corners, qualityLevel=0.01,
            minDistance=7, mask=mask, blockSize=7
        )

        if curr_points is None or len(curr_points) < 4:
            self.reset()
            return None

        # First frame: store features, can't compute flow yet
        if self.prev_gray is None or self.prev_points is None:
            self.prev_gray = roi
            self.prev_points = curr_points
            return None

        # Check if ROI size changed drastically
        prev_h, prev_w = self.prev_gray.shape[:2]
        if abs(roi_h - prev_h) > roi_h * 0.5 or abs(roi_w - prev_w) > roi_w * 0.5:
            self.prev_gray = roi
            self.prev_points = curr_points
            return None

        roi_for_flow = roi
        if roi_h != prev_h or roi_w != prev_w:
            roi_for_flow = cv2.resize(roi, (prev_w, prev_h))

        # Compute optical flow
        curr_tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, roi_for_flow, self.prev_points, None, **self.lk_params
        )

        # Filter valid tracks
        valid_prev, valid_curr = [], []
        for i in range(len(self.prev_points)):
            if status[i] != 1:
                continue
            p1 = self.prev_points[i][0]
            p2 = curr_tracked[i][0]
            if not (0 <= p2[0] < prev_w and 0 <= p2[1] < prev_h):
                continue
            flow_mag = np.linalg.norm(p2 - p1)
            if flow_mag < self.min_flow or flow_mag > self.max_flow:
                continue
            valid_prev.append(p1)
            valid_curr.append(p2)

        self.prev_gray = roi
        self.prev_points = curr_points

        if len(valid_prev) < 4:
            return None

        valid_prev = np.array(valid_prev)
        valid_curr = np.array(valid_curr)

        result = self._estimate_rotation(valid_prev, valid_curr,
                                         roi_center, ball_radius_px)

        if result is not None:
            result["tracked_features"] = {
                "prev_points": valid_prev,
                "curr_points": valid_curr
            }
            self.accumulated_rotation = result["rotation_matrix"] @ self.accumulated_rotation

        return result

    def _estimate_rotation(self, prev_pts, curr_pts, center, radius_px):
        """Estimate rotation from optical flow using least squares.

        For a rotating sphere, the velocity v of a surface point at
        3D position r from the center, due to angular velocity ω, is:

            v = ω × r

        We lift 2D points to 3D on the sphere surface:
            r_3d = [rx, ry, sqrt(R² - rx² - ry²)]

        Then solve the linear system A·ω = flow for all points at once.

        Args:
            prev_pts:  (N, 2) previous frame points
            curr_pts:  (N, 2) current frame points
            center:    ROI center (cx, cy)
            radius_px: Ball radius in pixels

        Returns:
            dict with rotation_matrix, spin_axis, confidence — or None
        """
        flow = curr_pts - prev_pts
        center = np.array(center, dtype=np.float64)

        # Lift 2D points to 3D on sphere
        r_2d = prev_pts - center
        r_sq = np.sum(r_2d ** 2, axis=1)
        R_sq = radius_px ** 2
        valid = r_sq < R_sq * 0.95
        if np.sum(valid) < 4:
            return None

        r_2d = r_2d[valid]
        flow_valid = flow[valid]
        r_sq = r_sq[valid]
        rz = np.sqrt(np.maximum(R_sq - r_sq, 0))

        # Build system: A @ [ωx, ωy, ωz] = flow_flat
        # vx =  ωy·rz - ωz·ry
        # vy = -ωx·rz + ωz·rx
        N = len(r_2d)
        rx, ry = r_2d[:, 0], r_2d[:, 1]
        A = np.zeros((2 * N, 3))
        A[0::2, 1] = rz
        A[0::2, 2] = -ry
        A[1::2, 0] = -rz
        A[1::2, 2] = rx
        b = np.empty(2 * N)
        b[0::2] = flow_valid[:, 0]
        b[1::2] = flow_valid[:, 1]

        try:
            omega, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        angular_speed = np.linalg.norm(omega)
        if angular_speed < 1e-8:
            return None

        axis_3d = omega / angular_speed

        R, _ = cv2.Rodrigues(omega.reshape(3, 1))

        # Confidence: fraction of points that agree with the solution
        predicted = A @ omega
        residuals = np.sqrt((predicted[0::2] - b[0::2])**2 +
                            (predicted[1::2] - b[1::2])**2)
        inliers = np.sum(residuals < 5.0)
        confidence = min(1.0, inliers / N)

        return {
            "rotation_matrix": R,
            "spin_axis": axis_3d,
            "confidence": confidence,
        }


# ============================================================
# Complete Optical Flow Pipeline
# ============================================================

class OpticalFlowPipeline:
    """Complete optical flow based orientation detection pipeline.

    Usage:
        from camera import load_camera_params
        K, dist, _ = load_camera_params("config/camera.json")
        pipeline = OpticalFlowPipeline(K, dist)
        result = pipeline.process_frame(frame, timestamp=0.0)
    """

    def __init__(self, camera_matrix, dist_coeffs, ball_radius_mm=37.0,
                 confidence=0.5, model_path="yolov8n.pt",
                 max_corners=50, min_flow=0.5, max_flow=30.0):
        """
        Args:
            camera_matrix:  3x3 camera intrinsic matrix
            dist_coeffs:    Distortion coefficients
            ball_radius_mm: Baseball radius in mm (≈37mm)
            confidence:     YOLO detection confidence threshold
            model_path:     Path to YOLO model weights
            max_corners:    Max feature points for optical flow
            min_flow:       Min flow magnitude threshold (pixels)
            max_flow:       Max flow magnitude threshold (pixels)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ball_radius_mm = ball_radius_mm

        self.detector = BallDetector(model_path, confidence)
        self.tracker = BallTracker(self.detector)
        self.rotation_estimator = RotationEstimator(
            camera_matrix, ball_radius_mm, max_corners, min_flow, max_flow
        )

        self.frame_count = 0
        self._consecutive_failures = 0

    def reset(self):
        """Reset all pipeline state."""
        self.frame_count = 0
        self._consecutive_failures = 0
        self.rotation_estimator.reset()
        self.tracker.reset()

    def process_frame(self, image, timestamp=None):
        """Process a single video frame.

        Args:
            image:     BGR image (H, W, 3)
            timestamp: Unused, kept for API compatibility

        Returns:
            dict with:
                ball_detected, bbox, confidence, orientation,
                spin_rate, spin_axis, frame_number, timestamp,
                flow_confidence, tracked_features
        """
        self.frame_count += 1

        result = {
            "ball_detected": False, "bbox": None, "confidence": None,
            "orientation": None, "spin_rate": None, "spin_axis": None,
            "frame_number": self.frame_count, "timestamp": timestamp,
            "flow_confidence": None, "tracked_features": None
        }

        # Step 1: Undistort
        try:
            image = undistort(image, self.camera_matrix, self.dist_coeffs)
        except Exception:
            pass

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Detect and track ball
        track = self.tracker.track(image)
        if not track["detected"]:
            self._consecutive_failures += 1
            if self._consecutive_failures > 5:
                self.rotation_estimator.reset()
            return result

        result["ball_detected"] = True
        result["bbox"] = track["bbox"]
        result["confidence"] = track["confidence"]
        result["tracking"] = track["tracking"]
        self._consecutive_failures = 0

        # Step 3: Estimate rotation from optical flow
        rot_result = self.rotation_estimator.estimate_rotation(
            gray, track["bbox"], timestamp
        )

        if rot_result and "tracked_features" in rot_result:
            result["tracked_features"] = rot_result["tracked_features"]

        if rot_result is None:
            return result

        # Step 4: Build orientation from accumulated rotation
        R_accumulated = self.rotation_estimator.accumulated_rotation
        result["orientation"] = {
            "rotation_matrix": R_accumulated,
            "quaternion": rotation_to_quaternion(R_accumulated),
            "euler_angles": rotation_to_euler(R_accumulated)
        }
        result["flow_confidence"] = rot_result.get("confidence")

        return result

    def process_video(self, video_path, output_path=None, visualize=False):
        """Process an entire video file.

        Returns:
            dict with total_frames, fps, detections, average_confidence
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path and visualize:
            writer = cv2.VideoWriter(output_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps if fps > 0 else None
            result = self.process_frame(frame, timestamp)
            results.append(result)

            if visualize and writer:
                writer.write(self._visualize(frame, result, frame_idx))

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        confidences = [r["flow_confidence"] for r in results
                       if r["flow_confidence"] is not None]

        return {
            "total_frames": frame_idx,
            "fps": fps,
            "detections": results,
            "average_spin_rate": None,
            "average_confidence": np.mean(confidences) if confidences else None
        }

    def _visualize(self, frame, result, frame_idx):
        """Draw optical flow results on the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 25

        cv2.putText(vis, f"Frame: {frame_idx} | Method: Optical Flow",
                    (10, y), font, 0.6, (255, 255, 255), 2)
        y += 25

        if not result["ball_detected"]:
            cv2.putText(vis, "NO BALL DETECTED",
                        (10, y), font, 0.6, (0, 0, 255), 2)
            return vis

        x1, y1b, x2, y2b = result["bbox"]
        cx, cy = (x1 + x2) // 2, (y1b + y2b) // 2

        color = (255, 255, 0) if not result.get("tracking") else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1b), (x2, y2b), color, 2)
        cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

        # Flow vectors
        if result.get("tracked_features"):
            tracked = result["tracked_features"]
            for p1, p2 in zip(tracked["prev_points"], tracked["curr_points"]):
                cv2.arrowedLine(vis, tuple(p1.astype(int)),
                                tuple(p2.astype(int)),
                                (255, 255, 0), 1, tipLength=0.3)
            cv2.putText(vis, f"Tracked: {len(tracked['prev_points'])} pts",
                        (10, y), font, 0.5, (255, 255, 0), 1)
            y += 20

        # Orientation info
        if result["orientation"]:
            q = result["orientation"]["quaternion"]
            e = np.degrees(result["orientation"]["euler_angles"])
            cv2.putText(vis,
                        f"Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]",
                        (10, y), font, 0.4, (255, 200, 0), 1)
            y += 18
            cv2.putText(vis,
                        f"Euler: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] deg",
                        (10, y), font, 0.4, (255, 200, 0), 1)
            y += 18

        if result.get("flow_confidence"):
            cv2.putText(vis, f"Flow conf: {result['flow_confidence']:.2f}",
                        (10, y), font, 0.5, (0, 200, 255), 1)

        return vis
