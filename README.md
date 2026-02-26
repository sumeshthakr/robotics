# ⚾ Baseball Orientation Detection

> **Detect 3D orientation of a baseball from monocular video — using two computer-vision pipelines.**

[![CI](https://github.com/sumeshthakr/robotics/actions/workflows/ci.yml/badge.svg)](https://github.com/sumeshthakr/robotics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange)
![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)

---

## 🎯 What This Does

Given a 30 fps monocular video of a hand-tossed baseball, the system outputs per-frame:
- **Bounding box** — ball location detected by YOLOv8
- **Absolute orientation** — rotation matrix / quaternion / Euler angles

Two independent algorithms tackle the problem:

| | Seam-Based Pipeline | Optical Flow Pipeline |
|---|---|---|
| **Core idea** | Detect red stitching → match 3D seam model → PnP solve | Track surface corners → Lucas-Kanade flow → least-squares rotation |
| **Orientation** | Perspective-n-Point (PnP) | Accumulated rotation matrix |
| **Best for** | High-contrast seams, close-up balls | Any surface texture, small balls |

---

## 🖼️ Live Detection Results

### Seam-Based Pipeline — Video 1

![Seam Pipeline Video 1](https://github.com/user-attachments/assets/144175a4-1728-4b91-89d0-35ecf92d84b5)

*Red dots = detected seam pixels (1135 px). Yellow box = YOLO bounding box.*

### Seam-Based Pipeline — Video 2

![Seam Pipeline Video 2](https://github.com/user-attachments/assets/46c34922-5db5-4ba6-9bde-6d141c85a26a)

*1009 seam pixels detected. Euler angles show the ball's current 3D orientation.*

### Optical Flow Pipeline — Video 2

![Optical Flow Video 2](https://github.com/user-attachments/assets/d7d48087-e060-46f0-9c3c-aa21c9776679)

*Yellow arrows = Lucas-Kanade optical flow vectors on 49 tracked corner features. Circle = ball boundary.*

### Side-by-Side Comparison — Video 1

![Comparison Video 1](https://github.com/user-attachments/assets/b6e2dc42-63bd-442f-964f-20e930bdef63)

*Left: Seam pipeline (1082 px detected). Right: Optical flow (42 tracked points).*

---

## 🏗️ System Architecture

```
  Video Frame (1700×1200 BGR)
         │
         ▼
  ┌──────────────┐
  │  camera.py   │  ← Load K, dist from config/camera.json
  │  undistort() │    OpenCV undistort (k1..k3, p1, p2)
  └──────┬───────┘
         │ Undistorted frame
         ▼
  ┌──────────────┐
  │ detector.py  │  ← YOLOv8n (COCO "sports ball" class 32)
  │ BallDetector │    Confidence threshold: 0.25
  │ BallTracker  │    Velocity-based EMA prediction (up to 5 lost frames)
  └──────┬───────┘
         │ bbox (x1,y1,x2,y2)  +  confidence
         │
    ┌────┴────────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌──────────────────────────┐   ┌─────────────────────────────┐
│     SEAM PIPELINE        │   │    OPTICAL FLOW PIPELINE    │
│   seam_pipeline.py       │   │    optical_pipeline.py      │
│                          │   │                             │
│ 1. Crop ROI from bbox    │   │ 1. Crop grayscale ROI       │
│ 2. Boost HSV saturation  │   │ 2. Detect corners           │
│    (1.5×)                │   │    goodFeaturesToTrack()    │
│ 3. Canny edge detection  │   │ 3. Track with LK optical    │
│    (adaptive thresholds) │   │    flow (pyramid, 3 levels) │
│ 4. HSV red filter        │   │ 4. Filter by flow magnitude │
│    hue [0-20]∪[160-180]  │   │    (0.5–30 px)              │
│ 5. seam_pixels: Nx2      │   │ 5. Lift 2D→3D on sphere:    │
│                          │   │    rz = √(R²−rx²−ry²)       │
│ 6. PnP for orientation:  │   │ 6. Build linear system:     │
│    BaseballSeamModel     │   │    A·ω = v  (v=ω×r)         │
│    (200 pts, 2 curves)   │   │ 7. lstsq solve → ω          │
│    solvePnPRansac()      │   │                             │
│                          │   │ 8. Accumulate rotation:     │
│                          │   │    R_acc = R_new @ R_acc    │
└──────────┬───────────────┘   └──────────┬──────────────────┘
           │                              │
           ▼                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                   orientation.py                       │
  │   rotation_to_quaternion()  → [w, x, y, z]             │
  │   rotation_to_euler()       → [roll, pitch, yaw]       │
  └────────────────────────────────────────────────────────┘
           │
           ▼
  Per-frame result dict:
    ball_detected, bbox, confidence
    orientation { rotation_matrix, quaternion, euler_angles }
    seam_pixels (seam pipeline) / tracked_features (optical pipeline)
```

---

## 📦 Module Reference

### `camera.py` — Camera Calibration

| Function | Purpose |
|---|---|
| `load_camera_params(path)` | Load K (3×3), dist (1×5), img_shape from JSON |
| `undistort(image, K, dist)` | Remove lens distortion via `cv2.undistort` |

**Camera intrinsics** (from `config/camera.json`):
- Focal length: **fx = fy = 10,248 px** (very long telephoto)
- Principal point: (362, 836) px
- Distortion: k1=0.388, k2=−32.6, p1=0.005, p2=−0.012, k3=3.27

---

### `detector.py` — Ball Detection & Tracking

#### `BallDetector`
- Runs **YOLOv8n** (6M params, COCO pre-trained)
- Filters detections to **class 32** ("sports ball")
- Returns highest-confidence detection per frame

#### `BallTracker`
Wraps `BallDetector` with velocity-based prediction:

```
Detected? ─Yes─► Update EMA velocity  →  return bbox
         └─No──► Lost frames < max?
                    Yes ─► Predict: bbox += velocity  (confidence × 0.9)
                    No  ─► Reset (ball lost)
```

- EMA velocity: `v_new = 0.7 × v_old + 0.3 × v_measured`
- Max lost frames: 5 (configurable)

---

### `seam_pipeline.py` — Seam Detection + PnP Orientation

#### `detect_seams(roi)`
Extracts red seam pixels from a ball ROI:

```
ROI (BGR)
 └─ Boost saturation ×1.5   (pale seams under strobe lighting)
 └─ Canny edge detection     (adaptive thresholds for small ROIs)
 └─ HSV red filter           hue ∈ [0°,20°] ∪ [160°,180°]
 └─ Combine: edges ∩ red     (fallback to all edges if <30% remain)
 └─ Morphological dilation   (connect nearby seam fragments)
 └─ Returns: Nx2 pixel coords
```

#### `BaseballSeamModel`
Parametric 3D model of baseball seam geometry:
- Two sinusoidal curves spiraling 2.5 revolutions around a sphere
- Parameterization: φ(t) = 2.5t + phase, θ(t) = π/2 + 0.4·sin(2.5t)
- Generates 400 3D points (200 per curve), all at radius ≈ 37 mm

#### `solve_orientation(pts2d, pts3d, K)`
Solves for 3D pose using **RANSAC PnP**:
- `cv2.solvePnPRansac` with 200 iterations, 15 px reprojection threshold
- Returns: R (3×3), rvec, tvec, inlier count

#### `SeamPipeline`
Full processing chain per frame:
1. Undistort → YOLO detect → ROI crop
2. Detect seam pixels
3. **Absolute orientation** via PnP

---

### `optical_pipeline.py` — Optical Flow Orientation

#### `RotationEstimator`
Estimates rotation from surface feature flow:

**Physics:**  For a rotating sphere, each surface point at 3D position **r** moves with velocity **v = ω × r**.

**Algorithm:**
1. Detect Shi-Tomasi corners inside ball circle (masked)
2. Track with Lucas-Kanade pyramid (3 levels, 15×15 window)
3. Filter tracks: `0.5 px < |flow| < 30 px` and inside ball circle
4. Lift 2D positions to 3D: `rz = √(R² − rx² − ry²)`
5. Build linear system: `A · [ωx, ωy, ωz]ᵀ = [vx₁, vy₁, …]ᵀ`
6. Solve with `numpy.linalg.lstsq` → rotation matrix via Rodrigues

#### `OpticalFlowPipeline`
Full processing chain per frame:
1. Undistort → YOLO detect
2. `RotationEstimator.estimate_rotation()` on ball ROI
3. Accumulate: `R_acc = R_incremental @ R_acc`
4. Return orientation as quaternion / Euler angles

---

### `orientation.py` — Rotation Format Conversions

| Function | Convention |
|---|---|
| `rotation_to_quaternion(R)` | Scalar-first: [w, x, y, z] |
| `rotation_to_euler(R)` | Intrinsic XYZ: [roll, pitch, yaw] in radians |

---

### `main.py` — CLI Entry Point

```bash
# Seam-based approach (default)
python main.py spin_dataset/video.mp4 --visualize

# Optical flow approach
python main.py spin_dataset/video.mp4 --approach optical --visualize

# Custom model / confidence / output
python main.py video.mp4 --model yolov8s.pt --confidence 0.3 --output results/
```

---

### `compare.py` — Side-by-Side Comparison Video

Processes each video through both pipelines simultaneously, writing a split-screen MP4:
- LEFT: seam pipeline visualization
- RIGHT: optical flow visualization
- BOTTOM: detection %, orientation %, frame time (ms)
- Saves `comparison_results.json` with full numeric breakdown

---

## 📊 Performance Metrics

*Measured on the two provided 30 fps spin_dataset videos.*

| Metric | Video 1 (98 frames) | Video 2 (85 frames) |
|--------|:-------------------:|:-------------------:|
| **Ball Detection Rate** | 45.9% | 48.2% |
| **Seam Orientation Rate** | 43.9% | 48.2% |
| **Optical Orientation Rate** | 39.8% | 45.9% |
| **Optical Avg Flow Confidence** | 0.549 | 0.634 |

---

## 🔁 CI/CD Pipeline

The repository uses **GitHub Actions** for continuous integration on every push and pull request.

```
.github/workflows/ci.yml
├── Job: lint-and-test  (Python 3.10 + 3.11 matrix)
│   ├── pip install -r requirements.txt + flake8 + pytest
│   ├── flake8 (syntax errors & undefined names → fail; style → warn)
│   └── pytest test_all.py -v  (37 unit tests)
│
└── Job: quick-verify  (runs after lint-and-test)
    └── python verify.py --quick  (math/model sanity checks, no video needed)
```

**What's tested in CI:**
- Camera parameter loading & undistortion
- YOLOv8 detector & velocity tracker
- Seam detection on synthetic images
- 3D seam model geometry (sphere distance, curve separation)
- PnP solver with known ground-truth pose
- Rotation format conversions (quaternion, Euler)
- Optical flow estimator (init, reset, frame processing)
- Both pipeline process_video error handling

---

## 🚀 Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run seam-based approach with visualization
python main.py spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4 \
    --visualize --output outputs/video1_seam

# Run optical flow approach
python main.py spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4 \
    --approach optical --visualize --output outputs/video1_optical

# Generate side-by-side comparison videos for both datasets
python compare.py

# Extract best detection frames for documentation
python extract_frames.py

# Run all 37 unit tests
pytest test_all.py -v

# Quick math/model verification (no video required)
python verify.py --quick
```

---

## 🗂️ Project Structure

```
robotics/
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD: lint → test → verify (matrix 3.10/3.11)
│
├── camera.py                  # Camera calibration loading + undistortion
├── detector.py                # YOLOv8 ball detection + EMA velocity tracking
├── orientation.py             # Quaternion/Euler conversion utilities
├── seam_pipeline.py           # Seam-based pipeline (Canny+HSV → PnP orientation)
├── optical_pipeline.py        # Optical flow pipeline (LK corners → lstsq rotation)
├── main.py                    # CLI entry point (--approach seam|optical)
├── compare.py                 # Side-by-side comparison video generator
├── extract_frames.py          # Best-frame extractor for documentation
├── test_all.py                # 37 unit tests (pytest)
├── verify.py                  # Physical-constraint verification
│
├── config/
│   └── camera.json            # fx=10248, dist coeffs, img_shape
│
├── spin_dataset/              # Two 30 fps spin demo videos (~98 and ~85 frames)
│   ├── raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4
│   └── raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4
│
├── docs/
│   └── frames/                # Best detection frames (generated by extract_frames.py)
│
├── REPORT.md                  # Assignment report (system design + math)
├── AI_COLLABORATION_LOG.md    # AI usage documentation
├── requirements.txt           # ultralytics, opencv-python, numpy, scipy, matplotlib
└── yolov8n.pt                 # YOLOv8 nano weights (COCO pre-trained)
```

---

## 🧪 Testing

```bash
# All 37 unit tests
pytest test_all.py -v

# Individual test classes
pytest test_all.py::TestCamera -v
pytest test_all.py::TestSeamDetection -v
pytest test_all.py::TestConversions -v
pytest test_all.py::TestRotationEstimator -v

# Physical constraint verification (no videos needed)
python verify.py --quick
```

**Test coverage by module:**

| Module | Tests | What's Validated |
|---|:---:|---|
| `camera.py` | 3 | JSON loading, missing file error, undistort shape |
| `detector.py` | 5 | Init, invalid confidence, output structure, invalid input |
| `BallTracker` | 4 | Init, reset, output structure, velocity prediction |
| `seam_pipeline.py` | 7 | Seam detection, seam model geometry, PnP solver |
| `orientation.py` | 3 | Quaternion and Euler conversions |
| `optical_pipeline.py` | 9 | Init, flow estimator, consecutive frames, pipeline reset |

---

## 📋 Requirements

```
ultralytics>=8.0.0     # YOLOv8 ball detection
opencv-python>=4.8.0   # Image processing, optical flow, PnP
numpy>=1.24.0          # Array math
scipy>=1.10.0          # Rotation math (Rotation class)
matplotlib>=3.7.0      # Optional: 3D visualization (plot_3d.py)
```

Python 3.10+ required.

---

## 📄 Deliverables

1. **System Design Document** → [`REPORT.md`](REPORT.md) (Parts 1–3: exposure time, focal length, Hough vs YOLO, bullet spin)
2. **Prototype Code** → This repository (modular 6-module pipeline)
3. **AI Usage Report** → [`AI_COLLABORATION_LOG.md`](AI_COLLABORATION_LOG.md)
4. **CI/CD Pipeline** → [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
