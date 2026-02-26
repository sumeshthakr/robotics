"""Seam-based baseball orientation detection pipeline.

APPROACH:
    Detect the baseball's red stitching (seams) in 2D, match them to a known
    3D seam geometry model, and solve for the ball's 3D orientation using the
    Perspective-n-Point (PnP) algorithm.

PIPELINE (per frame):
    1. Undistort image (correct lens distortion)
    2. Detect baseball with YOLO
    3. Extract ROI (region of interest) around the ball
    4. Detect red seam pixels using edge detection + color filtering
    5. Match detected 2D seam points to 3D seam model points
    6. Solve PnP for 3D orientation (rotation matrix)
    7. Track spin rate and axis over time

KEY INSIGHT:
    A baseball has a distinctive seam pattern that follows a known 3D curve.
    If we can detect the seam in 2D and know the 3D shape, we can compute
    the ball's orientation using standard camera geometry (PnP).
"""

import os
import numpy as np
import cv2

from camera import undistort
from detector import BallDetector, BallTracker
from orientation import OrientationTracker, rotation_to_quaternion, rotation_to_euler


# ============================================================
# Seam Detection
# ============================================================

def detect_seams(roi, canny_low=30, canny_high=100):
    """Detect baseball seam pixels in a ball ROI.

    Baseball seams are red stitching on a white ball. We detect them by:
        1. Boosting color saturation (seams can be pale in video)
        2. Running Canny edge detection to find all edges
        3. Filtering edges to keep only those in red color regions
        4. Cleaning up with morphological operations

    Args:
        roi:        Cropped ball image (H, W, 3) BGR format
        canny_low:  Lower Canny edge threshold
        canny_high: Upper Canny edge threshold

    Returns:
        dict with:
            seam_pixels: Nx2 array of (x, y) coordinates
            num_pixels:  Count of detected seam pixels
            edges:       Binary edge mask
    """
    h, w = roi.shape[:2]

    # Use more sensitive thresholds for small ROIs
    if h < 60 or w < 60:
        canny_low = max(20, canny_low - 20)
        canny_high = max(50, canny_high - 50)
        sat_low, val_low = 15, 40
    else:
        sat_low, val_low = 50, 80

    # --- Step 1: Boost saturation to make pale seams more visible ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = np.clip(s_ch.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    boosted = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch]), cv2.COLOR_HSV2BGR)

    # --- Step 2: Canny edge detection ---
    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    ksize = min(5, max(3, (h + w) // 20))
    if ksize % 2 == 0:
        ksize += 1  # Gaussian kernel must be odd
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # --- Step 3: Filter for red color (seam color) ---
    # Red wraps around 0°/180° in HSV hue, so we need TWO ranges:
    #   Range 1: hue 0-20   (red-orange)
    #   Range 2: hue 160-180 (red-magenta)
    hsv_boosted = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv_boosted,
                          np.array([0, sat_low, val_low]),
                          np.array([20, 255, 255]))
    red_high = cv2.inRange(hsv_boosted,
                           np.array([160, sat_low, val_low]),
                           np.array([180, 255, 255]))
    red_mask = red_low | red_high
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Keep only edges that are also in red regions
    combined = edges & red_mask

    # Fall back to all edges if color filter removes too much (>70%)
    if np.sum(combined) < np.sum(edges) * 0.3:
        combined = edges

    # --- Step 4: Dilate to connect nearby edge fragments ---
    combined = cv2.dilate(combined,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Extract (x, y) pixel coordinates
    yx = np.column_stack(np.where(combined > 0))
    seam_pixels = yx[:, [1, 0]] if len(yx) > 0 else np.zeros((0, 2), dtype=int)

    return {"seam_pixels": seam_pixels, "num_pixels": len(seam_pixels), "edges": combined}


# ============================================================
# 3D Baseball Seam Model
# ============================================================

class BaseballSeamModel:
    """Parametric 3D model of baseball seam geometry.

    A real baseball has two continuous seam curves that spiral around the
    sphere. Each curve makes approximately 2.5 revolutions with a
    sinusoidally varying inclination angle.

    We generate 3D points along these curves. These serve as the "known 3D
    positions" needed for PnP solving — we match detected 2D seam pixels
    against these 3D model points.

    The seam is parameterized in spherical coordinates:
        phi(t)   = 2.5 * t + phase    (azimuthal — spirals around)
        theta(t) = π/2 + 0.4*sin(2.5*t) (polar — wobbles around equator)
    """

    def __init__(self, radius=37.0):
        """
        Args:
            radius: Ball radius in mm (standard baseball ≈ 37mm radius)
        """
        self.radius = radius

    def generate_points(self, num_points_per_curve=100):
        """Generate 3D points along both seam curves.

        Returns:
            (2*N, 3) numpy array of 3D points on the sphere surface
        """
        t = np.linspace(0, 2 * np.pi, num_points_per_curve)
        curve1 = self._make_curve(t, phase=0)
        curve2 = self._make_curve(t, phase=np.pi)  # Second curve offset by 180°
        return np.vstack([curve1, curve2])

    def _make_curve(self, t, phase=0):
        """Generate one seam curve.

        Args:
            t:     Parameter array (0 to 2π)
            phase: Phase offset (0 for curve 1, π for curve 2)

        Returns:
            (N, 3) array of 3D points
        """
        phi = t * 2.5 + phase                     # Azimuthal angle (spiral)
        theta = np.pi / 2 + 0.4 * np.sin(2.5 * t)  # Polar angle (wobble)

        # Spherical → Cartesian
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        return np.column_stack([x, y, z])


# ============================================================
# PnP Orientation Solver
# ============================================================

def solve_orientation(points_2d, points_3d, camera_matrix,
                      rvec_init=None, tvec_init=None):
    """Solve for 3D orientation using PnP (Perspective-n-Point).

    Given matched 2D image points and their corresponding 3D world positions,
    find the rotation and translation that best explains how the 3D points
    project to the observed 2D locations.

    This uses OpenCV's solvePnP which minimizes the reprojection error.

    NOTE: In this pipeline, the 2D-3D matching is approximate — we don't
    have true correspondences. We sample 3D model points to match the count
    of detected 2D seam pixels. This gives rough orientation estimates.
    Better correspondence matching would improve accuracy significantly.

    Args:
        points_2d:     Nx2 detected seam pixel coordinates
        points_3d:     Nx3 corresponding 3D seam model points
        camera_matrix: 3x3 camera intrinsics
        rvec_init:     Initial rotation vector for refinement (3x1), or None
        tvec_init:     Initial translation vector for refinement (3x1), or None

    Returns:
        dict with:
            success:         bool
            rotation_matrix: 3x3 numpy array, or None
            rvec:            3x1 rotation vector, or None
            tvec:            3x1 translation vector, or None
            num_inliers:     int
    """
    if len(points_2d) < 4 or len(points_3d) < 4:
        return {"success": False, "rotation_matrix": None,
                "rvec": None, "tvec": None, "num_inliers": 0}

    pts2d = np.array(points_2d, dtype=np.float64)
    pts3d = np.array(points_3d, dtype=np.float64)

    use_guess = rvec_init is not None and tvec_init is not None

    # Use RANSAC PnP for robustness against wrong correspondences
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, camera_matrix.astype(np.float64), None,
            rvec=rvec_init.copy().astype(np.float64) if use_guess else None,
            tvec=tvec_init.copy().astype(np.float64) if use_guess else None,
            useExtrinsicGuess=use_guess,
            iterationsCount=200,
            reprojectionError=15.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return {"success": False, "rotation_matrix": None,
                "rvec": None, "tvec": None, "num_inliers": 0}

    if success and inliers is not None and len(inliers) >= 4:
        R, _ = cv2.Rodrigues(rvec)
        return {"success": True, "rotation_matrix": R,
                "rvec": rvec, "tvec": tvec, "num_inliers": len(inliers)}

    return {"success": False, "rotation_matrix": None,
            "rvec": None, "tvec": None, "num_inliers": 0}


def estimate_tvec_from_bbox(bbox, camera_matrix, ball_radius_mm=37.0):
    """Estimate ball center translation from bounding box using pinhole model.

    Uses: distance = focal_length × real_diameter / pixel_diameter

    Args:
        bbox:           (x1, y1, x2, y2) bounding box
        camera_matrix:  3x3 intrinsic matrix
        ball_radius_mm: Ball radius in mm

    Returns:
        tvec (3x1 numpy array) or None
    """
    x1, y1, x2, y2 = bbox
    cx_img = (x1 + x2) / 2.0
    cy_img = (y1 + y2) / 2.0
    diameter_px = ((x2 - x1) + (y2 - y1)) / 2.0  # average of width and height

    if diameter_px < 5:
        return None

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]

    # Pinhole model: z = f * D_real / D_pixel
    z = fx * (2 * ball_radius_mm) / diameter_px

    # Back-project image center to 3D
    tx = (cx_img - cx0) * z / fx
    ty = (cy_img - cy0) * z / fy

    return np.array([[tx], [ty], [z]], dtype=np.float64)


# ============================================================
# Complete Seam Pipeline
# ============================================================

class SeamPipeline:
    """Complete seam-based orientation detection pipeline.

    Combines all components: detection, seam finding, 3D model matching,
    PnP solving, and temporal tracking into a single easy-to-use class.

    Usage:
        from camera import load_camera_params
        K, dist, _ = load_camera_params("config/camera.json")
        pipeline = SeamPipeline(K, dist)
        result = pipeline.process_frame(frame, timestamp=0.0)
    """

    def __init__(self, camera_matrix, dist_coeffs, ball_radius_mm=37.0,
                 confidence=0.5, model_path="yolov8n.pt"):
        """
        Args:
            camera_matrix:  3x3 camera intrinsic matrix
            dist_coeffs:    Distortion coefficients
            ball_radius_mm: Baseball radius in mm (≈37mm)
            confidence:     YOLO detection confidence threshold
            model_path:     Path to YOLO model weights
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ball_radius_mm = ball_radius_mm

        # Initialize components
        self.detector = BallDetector(model_path, confidence)
        self.tracker = BallTracker(self.detector)
        self.seam_model = BaseballSeamModel(radius=ball_radius_mm)
        self.orientation_tracker = OrientationTracker()
        self.frame_count = 0

        # Pose tracking for PnP
        self.prev_rvec = None
        self.prev_tvec = None

        # Seam-based optical flow tracking for RPM
        # PnP with approximate correspondences gives noisy orientations.
        # Instead, we track seam pixels between frames with optical flow
        # and estimate rotation from the flow pattern (same physics as
        # optical_pipeline.py). This gives much more reliable RPM.
        self.prev_gray = None          # Previous full-frame grayscale
        self.prev_seam_pts = None      # Previous seam points (full-frame coords)
        self.prev_bbox = None          # Previous bounding box
        self.flow_accumulated_R = np.eye(3)  # Accumulated rotation from flow

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def process_frame(self, image, timestamp=None):
        """Process a single video frame.

        Args:
            image:     BGR image (H, W, 3)
            timestamp: Frame time in seconds (needed for spin rate)

        Returns:
            dict with:
                ball_detected:  bool
                bbox:           (x1, y1, x2, y2) or None
                confidence:     float or None
                orientation:    dict with rotation_matrix, quaternion, euler_angles — or None
                spin_rate:      RPM or None
                spin_axis:      3D unit vector or None
                frame_number:   int
                timestamp:      float or None
                seam_pixels:    Nx2 array or None
                num_seam_pixels: int
        """
        self.frame_count += 1

        result = {
            "ball_detected": False, "bbox": None, "confidence": None,
            "orientation": None, "spin_rate": None, "spin_axis": None,
            "frame_number": self.frame_count, "timestamp": timestamp,
            "seam_pixels": None, "num_seam_pixels": 0
        }

        # Step 1: Undistort
        try:
            image = undistort(image, self.camera_matrix, self.dist_coeffs)
        except Exception:
            pass  # Use original if undistortion fails

        # Step 2: Detect and track ball
        track = self.tracker.track(image)
        if not track["detected"]:
            return result

        result["ball_detected"] = True
        result["bbox"] = track["bbox"]
        result["confidence"] = track["confidence"]
        result["tracking"] = track["tracking"]

        # Step 3: Extract ROI
        x1, y1, x2, y2 = track["bbox"]
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return result

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 4: Detect seam pixels
        seam = detect_seams(roi)
        result["seam_pixels"] = seam["seam_pixels"]
        result["num_seam_pixels"] = seam["num_pixels"]

        # Need minimum seam pixels for reliable results
        min_pixels = max(4, int(roi.shape[0] * roi.shape[1] * 0.01))
        if seam["num_pixels"] < min_pixels:
            self.prev_gray = gray
            self.prev_seam_pts = None
            self.prev_bbox = track["bbox"]
            return result

        # Convert ROI coordinates → global image coordinates
        seam_2d = seam["seam_pixels"] + np.array([x1, y1])

        # ============================================================
        # Step 5: Spin rate from optical flow on seam features
        # ============================================================
        # PnP with approximate correspondences gives noisy orientations (±120°)
        # which produces ~640 RPM of pure noise. Instead, we track seam pixels
        # between frames using optical flow and compute rotation from the flow
        # pattern. This measures actual surface motion, not correspondence noise.
        flow_R = self._estimate_rotation_from_seam_flow(
            gray, seam_2d, track["bbox"])
        if flow_R is not None:
            self.flow_accumulated_R = flow_R @ self.flow_accumulated_R
            if timestamp is not None:
                self.orientation_tracker.add(self.flow_accumulated_R, timestamp)
            result["spin_rate"] = self.orientation_tracker.get_spin_rate()
            result["spin_axis"] = self.orientation_tracker.get_spin_axis()

        # Store seam points for next frame's flow tracking
        # Subsample to ~50 points for efficient optical flow
        if len(seam_2d) > 50:
            idx = np.random.choice(len(seam_2d), 50, replace=False)
            self.prev_seam_pts = seam_2d[idx].astype(np.float32).reshape(-1, 1, 2)
        else:
            self.prev_seam_pts = seam_2d.astype(np.float32).reshape(-1, 1, 2)
        self.prev_gray = gray
        self.prev_bbox = track["bbox"]

        # ============================================================
        # Step 6: Absolute orientation from PnP (for display only)
        # ============================================================
        # NOTE: This is still noisy due to approximate correspondences,
        # but provides a rough absolute orientation estimate.
        model_3d = self.seam_model.generate_points(num_points_per_curve=200)
        bbox_tvec = estimate_tvec_from_bbox(
            track["bbox"], self.camera_matrix, self.ball_radius_mm)
        init_rvec = self.prev_rvec
        init_tvec = self.prev_tvec if self.prev_tvec is not None else bbox_tvec

        n = len(seam_2d)
        if n <= len(model_3d):
            indices = np.linspace(0, len(model_3d) - 1, n, dtype=int)
            matched_3d = model_3d[indices]
        else:
            repeats = (n // len(model_3d)) + 1
            matched_3d = np.tile(model_3d, (repeats, 1))[:n]

        if n >= 4:
            pnp = solve_orientation(
                seam_2d, matched_3d, self.camera_matrix,
                rvec_init=init_rvec, tvec_init=init_tvec
            )
            if pnp["success"]:
                self.prev_rvec = pnp["rvec"]
                self.prev_tvec = pnp["tvec"]

                # Use flow-accumulated R for orientation if available,
                # fall back to PnP R
                R_display = self.flow_accumulated_R if flow_R is not None \
                    else pnp["rotation_matrix"]
                result["orientation"] = {
                    "rotation_matrix": R_display,
                    "quaternion": rotation_to_quaternion(R_display),
                    "euler_angles": rotation_to_euler(R_display)
                }

        return result

    def _estimate_rotation_from_seam_flow(self, gray, seam_2d_curr, bbox):
        """Estimate rotation by tracking seam pixels between frames.

        Uses Lucas-Kanade optical flow on the previous frame's seam pixels,
        then RANSAC to fit a rotation model to the flow.

        Same physics as optical_pipeline.py:
            v = ω × r  →  solve linear system for ω

        Args:
            gray:          Current frame grayscale
            seam_2d_curr:  Current frame seam pixels (full-frame coords)
            bbox:          Current bounding box (x1, y1, x2, y2)

        Returns:
            3x3 rotation matrix, or None
        """
        if self.prev_gray is None or self.prev_seam_pts is None:
            return None
        if self.prev_bbox is None:
            return None

        # Track previous seam points into current frame
        try:
            curr_tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_seam_pts, None,
                **self.lk_params
            )
        except cv2.error:
            return None

        # Filter valid tracks
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius_px = min(x2 - x1, y2 - y1) / 2.0

        valid_prev, valid_curr = [], []
        for i in range(len(self.prev_seam_pts)):
            if status[i] != 1:
                continue
            p1 = self.prev_seam_pts[i][0]
            p2 = curr_tracked[i][0]

            # Check bounds
            if not (x1 <= p2[0] <= x2 and y1 <= p2[1] <= y2):
                continue

            flow_mag = np.linalg.norm(p2 - p1)
            if flow_mag < 0.5 or flow_mag > 30.0:
                continue

            valid_prev.append(p1)
            valid_curr.append(p2)

        if len(valid_prev) < 4:
            return None

        prev_pts = np.array(valid_prev)
        curr_pts = np.array(valid_curr)
        flow = curr_pts - prev_pts

        # Center relative to ball center
        center = np.array([cx, cy], dtype=np.float64)
        r_2d = prev_pts - center
        r_sq = np.sum(r_2d ** 2, axis=1)
        R_sq = radius_px ** 2

        # Only use points on the sphere (inside ball circle)
        valid = r_sq < R_sq * 0.95
        if np.sum(valid) < 4:
            return None

        r_2d = r_2d[valid]
        flow_v = flow[valid]
        rz = np.sqrt(np.maximum(R_sq - r_sq[valid], 0))

        # Build linear system: A @ omega = flow_flat (vectorized)
        # For point i with position (rx, ry, rz):
        #   vx =  ωy·rz - ωz·ry  →  row [0,  rz, -ry]
        #   vy = -ωx·rz + ωz·rx  →  row [-rz, 0,  rx]
        N = len(r_2d)
        rx, ry = r_2d[:, 0], r_2d[:, 1]
        A = np.zeros((2 * N, 3))
        A[0::2, 1] = rz
        A[0::2, 2] = -ry
        A[1::2, 0] = -rz
        A[1::2, 2] = rx
        b = np.empty(2 * N)
        b[0::2] = flow_v[:, 0]
        b[1::2] = flow_v[:, 1]

        # RANSAC
        best_omega = None
        best_inliers = 0
        best_residuals = None

        for _ in range(50):
            idx = np.random.choice(N, min(3, N), replace=False)
            rows = np.concatenate([[2*j, 2*j+1] for j in idx])
            try:
                omega, _, _, _ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
            except np.linalg.LinAlgError:
                continue

            predicted = A @ omega
            residuals = np.sqrt((predicted[0::2] - b[0::2])**2 +
                                (predicted[1::2] - b[1::2])**2)
            inlier_mask = residuals < 5.0
            inliers = np.sum(inlier_mask)

            if inliers > best_inliers:
                best_inliers = inliers
                best_omega = omega
                best_residuals = residuals

        if best_inliers < 3 or best_omega is None:
            return None

        # Refit on inliers
        inlier_mask = best_residuals < 5.0
        inlier_rows = np.concatenate([[2*j, 2*j+1]
                                       for j in range(N) if inlier_mask[j]])
        try:
            best_omega, _, _, _ = np.linalg.lstsq(
                A[inlier_rows], b[inlier_rows], rcond=None)
        except np.linalg.LinAlgError:
            pass

        angular_speed = np.linalg.norm(best_omega)
        if angular_speed < 1e-8:
            return None

        axis = best_omega / angular_speed
        angle = angular_speed

        # Convert axis-angle to rotation matrix using OpenCV's Rodrigues
        R, _ = cv2.Rodrigues((axis * angle).reshape(3, 1))
        return R

    def process_video(self, video_path, output_path=None, visualize=False):
        """Process an entire video file.

        Args:
            video_path:  Path to input video
            output_path: Path to save annotated output video (optional)
            visualize:   If True, write visualization overlay to output_path

        Returns:
            dict with:
                total_frames:      int
                fps:               float
                detections:        list of per-frame results
                average_spin_rate: float or None
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

        spin_rates = [r["spin_rate"] for r in results if r["spin_rate"] is not None]

        return {
            "total_frames": frame_idx,
            "fps": fps,
            "detections": results,
            "average_spin_rate": np.mean(spin_rates) if spin_rates else None
        }

    def _visualize(self, frame, result, frame_idx):
        """Draw detection and orientation results on the frame.

        Shows: bounding box, seam pixels, orientation info, spin rate/axis.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 25

        cv2.putText(vis, f"Frame: {frame_idx} | Method: Seam",
                    (10, y), font, 0.6, (255, 255, 255), 2)
        y += 25

        if not result["ball_detected"]:
            cv2.putText(vis, "NO BALL DETECTED",
                        (10, y), font, 0.6, (0, 0, 255), 2)
            return vis

        x1, y1b, x2, y2b = result["bbox"]
        cx, cy = (x1 + x2) // 2, (y1b + y2b) // 2

        # Bounding box (green = detected, yellow = predicted)
        color = (0, 255, 0) if not result.get("tracking") else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1b), (x2, y2b), color, 2)
        cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

        # Seam pixels (red dots)
        if result.get("seam_pixels") is not None and len(result["seam_pixels"]) > 0:
            for px, py in result["seam_pixels"]:
                gx, gy = int(px + x1), int(py + y1b)
                if 0 <= gx < w and 0 <= gy < h:
                    cv2.circle(vis, (gx, gy), 1, (0, 0, 255), -1)
            cv2.putText(vis, f"Seam pixels: {result['num_seam_pixels']}",
                        (10, y), font, 0.5, (0, 0, 255), 1)
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

        # Spin rate
        if result["spin_rate"] is not None:
            cv2.putText(vis, f"Spin: {result['spin_rate']:.1f} RPM",
                        (10, y), font, 0.6, (0, 255, 255), 2)
            y += 22

        # Spin axis arrow
        if result["spin_axis"] is not None:
            axis = result["spin_axis"]
            end_x = int(cx + axis[0] * 60)
            end_y = int(cy + axis[1] * 60)
            cv2.arrowedLine(vis, (cx, cy), (end_x, end_y),
                            (255, 0, 255), 3, tipLength=0.3)

        # Detection confidence
        if result["confidence"] is not None:
            cv2.putText(vis, f"Conf: {result['confidence']:.2f}",
                        (10, y), font, 0.5, (0, 255, 0), 1)

        return vis

    def reset(self):
        """Reset all pipeline state."""
        self.frame_count = 0
        self.tracker.reset()
        self.orientation_tracker = OrientationTracker()
        self.prev_rvec = None
        self.prev_tvec = None
        self.prev_gray = None
        self.prev_seam_pts = None
        self.prev_bbox = None
        self.flow_accumulated_R = np.eye(3)
