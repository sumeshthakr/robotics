# Baseball Orientation Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a computer vision pipeline that detects baseball orientation from high-speed video using seam patterns, outputting rotation matrices, quaternions, spin axis, and spin rate.

**Architecture:** Hybrid approach using pretrained YOLOv8 for ball detection, classical CV (Canny edge detection, circle fitting) for seam localization, and PnP solver with 3D seam model for orientation estimation. Temporal tracking computes spin rate and axis.

**Tech Stack:** Python, YOLOv8 (ultralytics), OpenCV, NumPy, SciPy, Matplotlib

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `config/camera.json`
- Create: `main.py`

**Step 1: Create requirements.txt**

```bash
cat > requirements.txt << 'EOF'
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
EOF
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Create project structure**

```bash
mkdir -p src/{detection,preprocessing,seams,estimation,tracking,utils}
mkdir -p tests/{detection,preprocessing,seams,estimation,tracking}
mkdir -p data outputs/{viz,results}
touch src/__init__.py
touch src/detection/__init__.py
touch src/preprocessing/__init__.py
touch src/seams/__init__.py
touch src/estimation/__init__.py
touch src/tracking/__init__.py
touch src/utils/__init__.py
```

**Step 4: Copy camera parameters**

```bash
cp spin_camera_matrix.json config/camera.json
```

**Step 5: Create basic main.py skeleton**

```python
# main.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Baseball Orientation Detection")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()

    print(f"Processing video: {args.video_path}")
    # Pipeline will be added here

if __name__ == "__main__":
    main()
```

**Step 6: Test basic setup**

Run: `python main.py --help`
Expected: Help message displays

**Step 7: Commit**

```bash
git add requirements.txt src/ config/ main.py
git commit -m "feat: initialize project structure and dependencies"
```

---

## Task 2: Camera Utilities Module

**Files:**
- Create: `src/utils/camera.py`
- Create: `tests/utils/test_camera.py`

**Step 1: Write failing test**

```python
# tests/utils/test_camera.py
import pytest
import numpy as np
from src.utils.camera import load_camera_params, undistort_image

def test_load_camera_params():
    K, dist, img_shape = load_camera_params("config/camera.json")
    assert K.shape == (3, 3)
    assert dist.shape == (1, 5)
    assert img_shape == (1700, 1200, 3)
    assert K[0, 0] == pytest.approx(10248.145, rel=0.01)

def test_undistort_image_shape():
    import cv2
    K, dist, _ = load_camera_params("config/camera.json")
    test_img = np.zeros((1200, 1700, 3), dtype=np.uint8)
    result = undistort_image(test_img, K, dist)
    assert result.shape == test_img.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_camera.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'src.utils.camera'"

**Step 3: Implement camera.py**

```python
# src/utils/camera.py
import json
import numpy as np
import cv2

def load_camera_params(path: str):
    """Load camera parameters from JSON file.

    Args:
        path: Path to camera JSON file

    Returns:
        K: 3x3 camera matrix
        dist: 1x5 distortion coefficients
        img_shape: Image shape as tuple (H, W, C)
    """
    with open(path, 'r') as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    img_shape = tuple(data["img_shape"])

    return K, dist, img_shape

def undistort_image(image, K, dist):
    """Undistort image using camera parameters.

    Args:
        image: Input image (H, W, C)
        K: 3x3 camera matrix
        dist: 1x5 distortion coefficients

    Returns:
        Undistorted image
    """
    return cv2.undistort(image, K, dist)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/utils/test_camera.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/utils/camera.py tests/utils/test_camera.py
git commit -m "feat: add camera utilities for loading params and undistortion"
```

---

## Task 3: Ball Detection Module

**Files:**
- Create: `src/detection/ball_detector.py`
- Create: `tests/detection/test_ball_detector.py`

**Step 1: Write failing test**

```python
# tests/detection/test_ball_detector.py
import pytest
import numpy as np
from src.detection.ball_detector import BallDetector

def test_ball_detector_init():
    detector = BallDetector(model_name="yolov8n.pt")
    assert detector.model is not None

def test_detect_ball_shape():
    detector = BallDetector(model_name="yolov8n.pt")
    test_img = np.zeros((1200, 1700, 3), dtype=np.uint8)
    result = detector.detect(test_img)

    assert "bbox" in result
    assert "confidence" in result
    assert "detected" in result
    if result["detected"]:
        x1, y1, x2, y2 = result["bbox"]
        assert x2 > x1 and y2 > y1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/detection/test_ball_detector.py -v`
Expected: FAIL - Module not found

**Step 3: Implement ball_detector.py**

```python
# src/detection/ball_detector.py
import numpy as np
from ultralytics import YOLO

class BallDetector:
    """Baseball detector using YOLOv8."""

    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """Initialize detector.

        Args:
            model_name: YOLO model name or path
            confidence_threshold: Minimum confidence for detection
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # Sports ball class index in COCO is 32
        self.sport_ball_class = 32

    def detect(self, image: np.ndarray) -> dict:
        """Detect baseball in image.

        Args:
            image: Input image (H, W, 3)

        Returns:
            dict with keys:
                - detected: bool
                - bbox: (x1, y1, x2, y2) or None
                - confidence: float or None
        """
        results = self.model(image, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"detected": False, "bbox": None, "confidence": None}

        # Get the most confident detection for sports ball
        best_box = None
        best_conf = 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == self.sport_ball_class and conf > self.confidence_threshold:
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            return {
                "detected": True,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": best_conf
            }

        return {"detected": False, "bbox": None, "confidence": None}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/detection/test_ball_detector.py -v`
Expected: PASS (Note: first run will download YOLO model)

**Step 5: Commit**

```bash
git add src/detection/ball_detector.py tests/detection/test_ball_detector.py
git commit -m "feat: add YOLOv8 ball detector"
```

---

## Task 4: Sphere Fitting Module

**Files:**
- Create: `src/estimation/sphere_fitter.py`
- Create: `tests/estimation/test_sphere_fitter.py`

**Step 1: Write failing test**

```python
# tests/estimation/test_sphere_fitter.py
import pytest
import numpy as np
from src.estimation.sphere_fitter import fit_circle

def test_fit_circle_perfect():
    # Generate perfect circle points
    angles = np.linspace(0, 2*np.pi, 50)
    cx, cy, r = 100, 100, 50
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    points = np.column_stack([x, y])

    result = fit_circle(points)

    assert result["cx"] == pytest.approx(cx, abs=1)
    assert result["cy"] == pytest.approx(cy, abs=1)
    assert result["radius"] == pytest.approx(r, abs=1)
    assert result["success"] is True

def test_fit_circle_noisy():
    # Generate noisy circle points
    angles = np.linspace(0, 2*np.pi, 30)
    cx, cy, r = 150, 200, 75
    np.random.seed(42)
    x = cx + r * np.cos(angles) + np.random.randn(len(angles)) * 2
    y = cy + r * np.sin(angles) + np.random.randn(len(angles)) * 2
    points = np.column_stack([x, y])

    result = fit_circle(points)

    assert result["cx"] == pytest.approx(cx, abs=5)
    assert result["cy"] == pytest.approx(cy, abs=5)
    assert result["radius"] == pytest.approx(r, abs=5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/estimation/test_sphere_fitter.py -v`
Expected: FAIL - Module not found

**Step 3: Implement sphere_fitter.py**

```python
# src/estimation/sphere_fitter.py
import numpy as np
from scipy.optimize import least_squares

def fit_circle(points):
    """Fit circle to 2D points using least squares.

    Args:
        points: Nx2 array of (x, y) points

    Returns:
        dict with keys:
            - cx: center x coordinate
            - cy: center y coordinate
            - radius: circle radius
            - success: bool
    """
    if len(points) < 3:
        return {"cx": 0, "cy": 0, "radius": 0, "success": False}

    # Initial guess: centroid + average distance
    x_mean, y_mean = np.mean(points, axis=0)
    r_guess = np.mean(np.sqrt(np.sum((points - [x_mean, y_mean])**2, axis=1)))

    def residuals(params, points):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        return distances - r

    result = least_squares(
        residuals,
        x0=[x_mean, y_mean, r_guess],
        args=(points,),
        method='lm'
    )

    if result.success:
        cx, cy, r = result.x
        return {"cx": float(cx), "cy": float(cy), "radius": float(r), "success": True}
    else:
        return {"cx": 0, "cy": 0, "radius": 0, "success": False}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/estimation/test_sphere_fitter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/estimation/sphere_fitter.py tests/estimation/test_sphere_fitter.py
git commit -m "feat: add circle fitting for sphere detection"
```

---

## Task 5: Seam Detection Module

**Files:**
- Create: `src/seams/edge_detector.py`
- Create: `tests/seams/test_edge_detector.py`

**Step 1: Write failing test**

```python
# tests/seams/test_edge_detector.py
import pytest
import numpy as np
from src.seams.edge_detector import detect_seams

def test_detect_seams_shape():
    # Create synthetic image with red curves on white background
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Draw a red circle
    import cv2
    cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)

    result = detect_seams(img)

    assert "edges" in result
    assert "seam_pixels" in result
    assert result["edges"].shape == img.shape[:2]
    assert len(result["seam_pixels"]) > 0 or result["seam_pixels"].shape[1] == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/seams/test_edge_detector.py -v`
Expected: FAIL - Module not found

**Step 3: Implement edge_detector.py**

```python
# src/seams/edge_detector.py
import numpy as np
import cv2

def detect_seams(image: np.ndarray,
                 canny_low=50,
                 canny_high=150,
                 use_color_filter=True) -> dict:
    """Detect baseball seams using edge detection and optional color filtering.

    Args:
        image: Input image (H, W, 3), ROI of ball only
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        use_color_filter: If True, filter for red seam color

    Returns:
        dict with keys:
            - edges: binary edge map (H, W)
            - seam_pixels: Nx2 array of (x, y) seam pixel coordinates
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Optional: color filter for red seams
    if use_color_filter:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range (wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Combine edges with red mask
        edges = cv2.bitwise_and(edges, red_mask)

    # Find seam pixel coordinates
    seam_pixels = np.column_stack(np.where(edges > 0))

    # Swap to (x, y) format for OpenCV compatibility
    if len(seam_pixels) > 0:
        seam_pixels = seam_pixels[:, [1, 0]]

    return {
        "edges": edges,
        "seam_pixels": seam_pixels,
        "num_pixels": len(seam_pixels)
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/seams/test_edge_detector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/seams/edge_detector.py tests/seams/test_edge_detector.py
git commit -m "feat: add seam detection using Canny edge detection"
```

---

## Task 6: 3D Seam Model

**Files:**
- Create: `src/seams/seam_model.py`
- Create: `tests/seams/test_seam_model.py`

**Step 1: Write failing test**

```python
# tests/seams/test_seam_model.py
import pytest
import numpy as np
from src.seams.seam_model import BaseballSeamModel

def test_seam_model_shape():
    model = BaseballSeamModel(radius=1.0)
    points = model.get_3d_points(num_points_per_curve=50)

    # Should have 2 curves
    assert len(points) == 2

    # Each curve should have requested points
    assert points[0].shape[0] == 50
    assert points[1].shape[0] == 50

    # Points should be 3D
    assert points[0].shape[1] == 3

def test_seam_model_radius():
    model = BaseballSeamModel(radius=37.0)  # ~37mm baseball radius
    points = model.get_3d_points()

    # All points should be approximately at radius distance from origin
    for curve in points:
        distances = np.linalg.norm(curve, axis=1)
        assert np.allclose(distances, 37.0, atol=1.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/seams/test_seam_model.py -v`
Expected: FAIL - Module not found

**Step 3: Implement seam_model.py**

```python
# src/seams/seam_model.py
import numpy as np

class BaseballSeamModel:
    """3D model of baseball seam geometry.

    Baseball has two curved seams that follow a specific pattern on the sphere.
    This model generates 3D points along both seam curves.
    """

    def __init__(self, radius=1.0):
        """Initialize seam model.

        Args:
            radius: Ball radius in desired units (default: 1.0 for normalized)
        """
        self.radius = radius

    def get_3d_points(self, num_points_per_curve=100) -> list:
        """Generate 3D points along both seam curves.

        The baseball seam can be approximated as two curves that are
        rotations of a single curve pattern.

        Args:
            num_points_per_curve: Number of 3D points per seam curve

        Returns:
            List of two Nx3 arrays, one for each seam curve
        """
        t = np.linspace(0, 2*np.pi, num_points_per_curve)

        # Baseball seam approximation: curve that goes around the sphere
        # One common parameterization:
        curve1 = self._generate_seam_curve(t, phase_offset=0)
        curve2 = self._generate_seam_curve(t, phase_offset=np.pi)

        return [curve1, curve2]

    def _generate_seam_curve(self, t, phase_offset=0) -> np.ndarray:
        """Generate one seam curve.

        Uses a mathematical approximation of the baseball seam shape.
        The seam is a curve on sphere that makes approximately 2.5 revolutions.

        Args:
            t: Parameter values (0 to 2*pi)
            phase_offset: Phase shift for this curve

        Returns:
            Nx3 array of 3D points
        """
        # Baseball seam makes approximately 2.5 turns around the ball
        num_revolutions = 2.5

        # Parameterize curve on sphere
        # Using spherical coordinates with varying inclination
        phi = t * num_revolutions + phase_offset  # Azimuthal angle
        theta = np.pi/2 + 0.4 * np.sin(2.5 * t)   # Polar angle (varies)

        # Convert to Cartesian coordinates
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        return np.column_stack([x, y, z])

    def get_all_points(self) -> np.ndarray:
        """Get all seam points as a single array.

        Returns:
            (2*N)x3 array of all seam 3D points
        """
        curves = self.get_3d_points()
        return np.vstack(curves)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/seams/test_seam_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/seams/seam_model.py tests/seams/test_seam_model.py
git commit -m "feat: add 3D baseball seam model"
```

---

## Task 7: PnP Solver Module

**Files:**
- Create: `src/estimation/pnp_solver.py`
- Create: `tests/estimation/test_pnp_solver.py`

**Step 1: Write failing test**

```python
# tests/estimation/test_pnp_solver.py
import pytest
import numpy as np
from src.estimation.pnp_solver import solve_orientation
from src.seams.seam_model import BaseballSeamModel

def test_solve_orientation_identity():
    # Generate known 3D points and project with identity rotation
    model = BaseballSeamModel(radius=37.0)
    points_3d = model.get_all_points()

    # Simple camera: identity rotation, ball at center
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float64)

    # Project 3D points (simplified - no translation)
    rvec = np.array([0, 0, 0], dtype=np.float64)
    tvec = np.array([0, 0, 500], dtype=np.float64)  # 500mm in front of camera

    import cv2
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    points_2d = points_2d.reshape(-1, 2)

    result = solve_orientation(points_2d, points_3d, K)

    assert result["success"] is True
    assert "rotation_matrix" in result
    assert "translation" in result
    assert result["rotation_matrix"].shape == (3, 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/estimation/test_pnp_solver.py -v`
Expected: FAIL - Module not found

**Step 3: Implement pnp_solver.py**

```python
# src/estimation/pnp_solver.py
import numpy as np
import cv2

def solve_orientation(points_2d, points_3d, camera_matrix,
                      method=cv2.SOLVEPNP_ITERATIVE) -> dict:
    """Solve for orientation using PnP with 3D seam model.

    Args:
        points_2d: Nx2 array of detected 2D seam pixel coordinates
        points_3d: Nx3 array of corresponding 3D seam model points
        camera_matrix: 3x3 camera intrinsics matrix
        method: PnP solving method

    Returns:
        dict with keys:
            - success: bool
            - rotation_matrix: 3x3 rotation matrix or None
            - translation: 3x1 translation vector or None
            - rvec: 3x1 rotation vector (Rodrigues) or None
            - tvec: 3x1 translation vector or None
    """
    if len(points_2d) < 4 or len(points_3d) < 4:
        return {
            "success": False,
            "rotation_matrix": None,
            "translation": None,
            "rvec": None,
            "tvec": None
        }

    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        distCoeffs=None,
        flags=method
    )

    if success:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        return {
            "success": True,
            "rotation_matrix": R,
            "translation": tvec,
            "rvec": rvec,
            "tvec": tvec
        }
    else:
        return {
            "success": False,
            "rotation_matrix": None,
            "translation": None,
            "rvec": None,
            "tvec": None
        }

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        [w, x, y, z] quaternion (scalar-first)
    """
    # Handle potential numerical issues
    R = np.array(R, dtype=np.float64)

    # Compute quaternion using trace method
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (ZYX convention).

    Args:
        R: 3x3 rotation matrix

    Returns:
        [roll, pitch, yaw] in radians
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/estimation/test_pnp_solver.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/estimation/pnp_solver.py tests/estimation/test_pnp_solver.py
git commit -m "feat: add PnP solver for orientation estimation"
```

---

## Task 8: Orientation Tracking Module

**Files:**
- Create: `src/tracking/orientation_tracker.py`
- Create: `tests/tracking/test_orientation_tracker.py`

**Step 1: Write failing test**

```python
# tests/tracking/test_orientation_tracker.py
import pytest
import numpy as np
from src.tracking.orientation_tracker import OrientationTracker

def test_tracker_init():
    tracker = OrientationTracker()
    assert len(tracker.history) == 0

def test_tracker_add_orientation():
    tracker = OrientationTracker()

    R1 = np.eye(3)
    tracker.add_orientation(R1, timestamp=0.0)

    assert len(tracker.history) == 1
    assert tracker.history[0]["timestamp"] == 0.0

def test_compute_spin_rate():
    tracker = OrientationTracker()

    # Add two orientations with known rotation
    from scipy.spatial.transform import Rotation

    R1 = Rotation.from_euler('z', 0).as_matrix()
    R2 = Rotation.from_euler('z', np.pi/2).as_matrix()  # 90 degree rotation

    tracker.add_orientation(R1, timestamp=0.0)
    tracker.add_orientation(R2, timestamp=0.1)  # 100ms

    spin_rate = tracker.get_spin_rate()

    # 90 degrees in 0.1 seconds = 900 deg/s = 150 RPM
    expected_rpm = 150
    assert spin_rate == pytest.approx(expected_rpm, rel=0.1)

def test_compute_spin_axis():
    tracker = OrientationTracker()

    from scipy.spatial.transform import Rotation

    R1 = Rotation.from_euler('z', 0).as_matrix()
    R2 = Rotation.from_euler('z', np.pi/4).as_matrix()  # Rotation around Z

    tracker.add_orientation(R1, timestamp=0.0)
    tracker.add_orientation(R2, timestamp=0.1)

    axis = tracker.get_spin_axis()

    # Should be approximately [0, 0, 1] (Z-axis rotation)
    assert np.allclose(axis[:2], [0, 0], atol=0.1)
    assert abs(axis[2]) > 0.9
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/tracking/test_orientation_tracker.py -v`
Expected: FAIL - Module not found

**Step 3: Implement orientation_tracker.py**

```python
# src/tracking/orientation_tracker.py
import numpy as np
from scipy.spatial.transform import Rotation

class OrientationTracker:
    """Track ball orientation over time to compute spin rate and axis."""

    def __init__(self, window_size=10):
        """Initialize tracker.

        Args:
            window_size: Number of frames to keep in history
        """
        self.history = []
        self.window_size = window_size

    def add_orientation(self, rotation_matrix, timestamp):
        """Add a new orientation measurement.

        Args:
            rotation_matrix: 3x3 rotation matrix
            timestamp: Frame timestamp in seconds
        """
        self.history.append({
            "rotation_matrix": np.array(rotation_matrix),
            "timestamp": timestamp
        })

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_spin_rate(self):
        """Compute current spin rate in RPM.

        Returns:
            Spin rate in RPM, or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        # Use last two orientations
        R1 = self.history[-2]["rotation_matrix"]
        R2 = self.history[-1]["rotation_matrix"]
        t1 = self.history[-2]["timestamp"]
        t2 = self.history[-1]["timestamp"]

        # Compute relative rotation: R_rel = R1^T * R2
        R_rel = R1.T @ R2

        # Convert to axis-angle
        rot = Rotation.from_matrix(R_rel)
        angle = rot.as_rotvec()  # Rotation vector (axis * angle)

        # Magnitude is the rotation angle in radians
        rotation_angle = np.linalg.norm(angle)
        if rotation_angle < 1e-6:
            return 0.0

        # Time difference
        dt = t2 - t1
        if dt < 1e-6:
            return None

        # Angular velocity in rad/s
        omega = rotation_angle / dt

        # Convert to RPM
        rpm = omega * 60 / (2 * np.pi)

        return rpm

    def get_spin_axis(self):
        """Compute current spin axis in camera coordinates.

        Returns:
            3D unit vector representing spin axis, or None
        """
        if len(self.history) < 2:
            return None

        R1 = self.history[-2]["rotation_matrix"]
        R2 = self.history[-1]["rotation_matrix"]

        # Compute relative rotation
        R_rel = R1.T @ R2

        # Convert to rotation vector
        rot = Rotation.from_matrix(R_rel)
        rotvec = rot.as_rotvec()

        magnitude = np.linalg.norm(rotvec)
        if magnitude < 1e-6:
            return np.array([0, 0, 1])  # Default axis if no rotation

        # Normalize to get axis
        axis = rotvec / magnitude

        return axis

    def get_current_orientation(self):
        """Get the most recent orientation.

        Returns:
            dict with rotation_matrix, quaternion, euler_angles or None
        """
        if len(self.history) == 0:
            return None

        R = self.history[-1]["rotation_matrix"]

        # Compute quaternion
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w] scalar-last
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        # Compute Euler angles
        euler = rot.as_euler('xyz')  # [roll, pitch, yaw]

        return {
            "rotation_matrix": R,
            "quaternion": quat_wxyz,
            "euler_angles": euler
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/tracking/test_orientation_tracker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tracking/orientation_tracker.py tests/tracking/test_orientation_tracker.py
git commit -m "feat: add orientation tracker for spin rate and axis computation"
```

---

## Task 9: Main Pipeline Integration

**Files:**
- Create: `src/pipeline.py`
- Modify: `main.py`

**Step 1: Write integration test**

```python
# tests/test_pipeline.py
import pytest
import numpy as np
import cv2
from src.pipeline import BaseballOrientationPipeline
from src.utils.camera import load_camera_params

def test_pipeline_init():
    K, dist, _ = load_camera_params("config/camera.json")
    pipeline = BaseballOrientationPipeline(camera_matrix=K, dist_coeffs=dist)
    assert pipeline is not None

def test_pipeline_process_frame():
    # Create synthetic test image
    img = np.ones((1200, 1700, 3), dtype=np.uint8) * 255
    cv2.circle(img, (850, 600), 100, (200, 200, 200), -1)  # Gray ball
    cv2.circle(img, (850, 600), 100, (0, 0, 200), 2)  # Red seam

    K, dist, _ = load_camera_params("config/camera.json")
    pipeline = BaseballOrientationPipeline(camera_matrix=K, dist_coeffs=dist)

    result = pipeline.process_frame(img, timestamp=0.0)

    assert "ball_detected" in result
    assert "orientation" in result
    assert "frame_number" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL - Module not found

**Step 3: Implement pipeline.py**

```python
# src/pipeline.py
import numpy as np
import cv2
from src.detection.ball_detector import BallDetector
from src.preprocessing.undistort import undistort_roi
from src.seams.edge_detector import detect_seams
from src.seams.seam_model import BaseballSeamModel
from src.estimation.sphere_fitter import fit_circle
from src.estimation.pnp_solver import solve_orientation, rotation_matrix_to_quaternion, rotation_matrix_to_euler
from src.tracking.orientation_tracker import OrientationTracker

class BaseballOrientationPipeline:
    """Complete pipeline for baseball orientation detection."""

    def __init__(self, camera_matrix, dist_coeffs,
                 ball_radius_mm=37.0,
                 confidence_threshold=0.5):
        """Initialize pipeline.

        Args:
            camera_matrix: 3x3 camera intrinsics
            dist_coeffs: 1x5 distortion coefficients
            ball_radius_mm: Baseball radius in mm (~37mm)
            confidence_threshold: Detection confidence threshold
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ball_radius = ball_radius_mm

        # Initialize components
        self.detector = BallDetector(confidence_threshold=confidence_threshold)
        self.seam_model = BaseballSeamModel(radius=ball_radius_mm)
        self.tracker = OrientationTracker()

        self.frame_count = 0

    def process_frame(self, image, timestamp=None):
        """Process a single frame.

        Args:
            image: Input image (H, W, 3)
            timestamp: Frame timestamp in seconds

        Returns:
            dict with processing results
        """
        if timestamp is None:
            timestamp = self.frame_count / 30.0  # Assume 30 FPS default

        result = {
            "frame_number": self.frame_count,
            "timestamp": timestamp,
            "ball_detected": False
        }

        # Step 1: Detect ball
        detection = self.detector.detect(image)

        if not detection["detected"]:
            self.frame_count += 1
            return result

        result["ball_detected"] = True
        result["bbox"] = detection["bbox"]
        result["confidence"] = detection["confidence"]

        # Step 2: Extract and undistort ROI
        x1, y1, x2, y2 = detection["bbox"]
        roi = image[y1:y2, x1:x2]

        # Undistort ROI
        roi_undistorted = cv2.undistort(roi, self.camera_matrix, self.dist_coeffs)

        # Step 3: Detect seams
        seam_result = detect_seams(roi_undistorted)
        result["num_seam_pixels"] = seam_result["num_pixels"]

        if seam_result["num_pixels"] < 10:
            self.frame_count += 1
            result["orientation_estimated"] = False
            return result

        # Step 4: Fit circle to ball outline (for center validation)
        # Use edge map for ball outline
        gray = cv2.cvtColor(roi_undistorted, cv2.COLOR_BGR2GRAY)
        ball_edges = cv2.Canny(gray, 50, 150)

        # Get contours and fit circle
        contours, _ = cv2.findContours(ball_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_center = None
        ball_radius_px = None

        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5:
                (cx, cy), radius = cv2.minEnclosingCircle(largest)
                ball_center = (cx, cy)
                ball_radius_px = radius

        result["ball_center"] = ball_center
        result["ball_radius_px"] = ball_radius_px

        # Step 5: Solve for orientation using PnP
        seam_pixels_2d = seam_result["seam_pixels"]
        seam_points_3d = self.seam_model.get_all_points()

        # Adjust 3D points to match camera view (simplified)
        # In production, you'd want better initial guess handling

        pnp_result = solve_orientation(
            seam_pixels_2d,
            seam_points_3d,
            self.camera_matrix
        )

        result["orientation_estimated"] = pnp_result["success"]

        if pnp_result["success"]:
            R = pnp_result["rotation_matrix"]

            # Add to tracker
            self.tracker.add_orientation(R, timestamp)

            # Get orientation representations
            quat = rotation_matrix_to_quaternion(R)
            euler = rotation_matrix_to_euler(R)

            result["rotation_matrix"] = R.tolist()
            result["quaternion"] = quat.tolist()
            result["euler_angles"] = euler.tolist()

            # Get spin info
            spin_rate = self.tracker.get_spin_rate()
            spin_axis = self.tracker.get_spin_axis()

            result["spin_rate_rpm"] = float(spin_rate) if spin_rate else None
            result["spin_axis"] = spin_axis.tolist() if spin_axis is not None else None

        self.frame_count += 1
        return result

    def process_video(self, video_path, output_path=None, visualize=False):
        """Process entire video.

        Args:
            video_path: Path to input video
            output_path: Optional path to save results JSON
            visualize: If True, save visualization frames

        Returns:
            List of per-frame results
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            result = self.process_frame(frame, timestamp)
            results.append(result)

            if visualize and result.get("orientation_estimated"):
                self._save_visualization(frame, result, frame_idx)

            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()

        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

        return results

    def _save_visualization(self, frame, result, frame_idx):
        """Save visualization frame."""
        import os
        os.makedirs("outputs/viz", exist_ok=True)

        vis_frame = frame.copy()

        if result.get("ball_detected"):
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result.get("spin_axis"):
            axis = result["spin_axis"]
            # Draw axis arrow at ball center
            if result.get("ball_center"):
                cx, cy = result["ball_center"]
                x1, y1, _, _ = result["bbox"]
                # Adjust to full image coordinates
                cx_full = int(cx + x1)
                cy_full = int(cy + y1)

                # Scale axis for visualization
                scale = 50
                end_x = int(cx_full + axis[0] * scale)
                end_y = int(cy_full + axis[1] * scale)

                cv2.arrowedLine(vis_frame, (cx_full, cy_full), (end_x, end_y), (255, 0, 0), 3)

        cv2.imwrite(f"outputs/viz/frame_{frame_idx:04d}.jpg", vis_frame)
```

**Step 4: Update main.py**

```python
# main.py (updated)
import argparse
from pathlib import Path
from src.pipeline import BaseballOrientationPipeline
from src.utils.camera import load_camera_params
import json

def main():
    parser = argparse.ArgumentParser(description="Baseball Orientation Detection")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/results/results.json",
                        help="Output JSON path")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization frames")
    args = parser.parse_args()

    print(f"Loading camera parameters...")
    K, dist, _ = load_camera_params("config/camera.json")

    print(f"Initializing pipeline...")
    pipeline = BaseballOrientationPipeline(camera_matrix=K, dist_coeffs=dist)

    print(f"Processing video: {args.video_path}")
    results = pipeline.process_video(
        args.video_path,
        output_path=args.output,
        visualize=args.visualize
    )

    # Summary statistics
    detected_frames = sum(1 for r in results if r.get("ball_detected"))
    orientation_frames = sum(1 for r in results if r.get("orientation_estimated"))

    print(f"\n=== Summary ===")
    print(f"Total frames: {len(results)}")
    print(f"Ball detected: {detected_frames} ({detected_frames/len(results)*100:.1f}%)")
    print(f"Orientation estimated: {orientation_frames}")

    spin_rates = [r["spin_rate_rpm"] for r in results if r.get("spin_rate_rpm")]
    if spin_rates:
        avg_spin = sum(spin_rates) / len(spin_rates)
        print(f"Average spin rate: {avg_spin:.1f} RPM")

    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

**Step 6: Test on actual video**

Run: `python main.py spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4 --visualize`
Expected: Processes video and generates outputs

**Step 7: Commit**

```bash
git add src/pipeline.py main.py tests/test_pipeline.py
git commit -m "feat: integrate complete orientation detection pipeline"
```

---

## Task 10: Documentation and Cleanup

**Files:**
- Create: `README.md`
- Create: `docs/plans/2025-02-24-baseball-orientation-design.md`

**Step 1: Create README.md**

```bash
cat > README.md << 'EOF'
# Baseball Orientation Detection

Computer vision pipeline for detecting baseball orientation from high-speed video using seam patterns.

## Features

- Ball detection using pretrained YOLOv8
- Seam detection via edge detection and color filtering
- 3D orientation estimation using PnP with seam model
- Spin rate and spin axis computation
- Multiple output formats: rotation matrix, quaternion, Euler angles

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Process a video
python main.py path/to/video.mp4 --output results.json --visualize

# Process your spin dataset
python main.py spin_dataset/raw_spin_video_*.mp4 --visualize
```

## Output

The pipeline generates:
- `results.json`: Per-frame orientation data
- `outputs/viz/`: Visualization frames with overlays

### Output Format

```json
{
  "frame_number": 0,
  "timestamp": 0.0,
  "ball_detected": true,
  "orientation_estimated": true,
  "rotation_matrix": [[...]],
  "quaternion": [w, x, y, z],
  "euler_angles": [roll, pitch, yaw],
  "spin_rate_rpm": 1500.5,
  "spin_axis": [0.1, 0.2, 0.97]
}
```

## Project Structure

```
├── src/
│   ├── detection/       # Ball detection
│   ├── preprocessing/   # Image preprocessing
│   ├── seams/          # Seam detection and 3D model
│   ├── estimation/     # Orientation estimation
│   ├── tracking/       # Temporal tracking
│   └── utils/          # Camera utilities
├── config/             # Camera parameters
├── tests/              # Unit tests
└── main.py            # Entry point
```

## Running Tests

```bash
pytest tests/ -v
```

## Next Steps

- [ ] Implement Approach 3 (Optical Flow)
- [ ] Compare results between approaches
- [ ] Fine-tune YOLO on baseball dataset
EOF
```

**Step 2: Create design document**

```bash
cat > docs/plans/2025-02-24-baseball-orientation-design.md << 'EOF'
# Baseball Orientation Detection - Design Document

## Overview

This document describes the design of a computer vision pipeline for detecting baseball orientation from high-speed video footage.

## Approach 1: Sphere Fitting + Seam Template Matching

### Architecture

```
Input Video → Ball Detection → ROI Extraction → Preprocessing → Seam Detection →
Orientation Estimation → Tracking → Output
```

### Components

1. **Ball Detection Module** (`src/detection/`)
   - Uses pretrained YOLOv8
   - Outputs bounding box and confidence

2. **Preprocessing** (`src/preprocessing/`)
   - Camera undistortion using calibration data
   - ROI extraction

3. **Seam Detection** (`src/seams/`)
   - Canny edge detection
   - Red color filtering (HSV)
   - Outputs seam pixel coordinates

4. **3D Seam Model** (`src/seams/seam_model.py`)
   - Parametric model of baseball seam geometry
   - Generates 3D points for PnP

5. **Orientation Estimation** (`src/estimation/`)
   - Circle fitting for ball center/radius
   - PnP solver with camera intrinsics
   - Multiple output formats (matrix, quaternion, Euler)

6. **Tracking** (`src/tracking/`)
   - Temporal orientation tracking
   - Spin rate computation (RPM)
   - Spin axis extraction

### Data Flow

```
Frame → YOLO → bbox → undistort(I, bbox) → edges → seam_pixels →
fit_circle → PnP(seam_pixels, seam_3d_model, K) → R,t →
track(R sequence) → {axis, rpm, quaternion}
```

## Approach 3: Optical Flow (Future)

To be implemented for comparison with Approach 1.

## References

- Camera calibration: `config/camera.json`
- Spin dataset: `data/spin_dataset/`
EOF
```

**Step 3: Run all tests to verify**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 4: Clean up temporary frame directories**

Run: `rm -rf frames_video1 frames_video2`

**Step 5: Commit**

```bash
git add README.md docs/plans/2025-02-24-baseball-orientation-design.md
git commit -m "docs: add README and design documentation"
```

---

## Summary

This implementation plan builds a complete baseball orientation detection pipeline in 10 tasks:

1. ✅ Project setup and dependencies
2. ✅ Camera utilities (load params, undistort)
3. ✅ Ball detection (YOLOv8)
4. ✅ Sphere/circle fitting
5. ✅ Seam detection (Canny + color filter)
6. ✅ 3D seam model
7. ✅ PnP orientation solver
8. ✅ Temporal tracking (spin rate, axis)
9. ✅ Pipeline integration
10. ✅ Documentation

Each task follows TDD with:
- Failing test → Implementation → Passing test → Commit

Total estimated time: 4-8 hours for implementation.
