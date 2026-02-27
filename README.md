# ⚾ Baseball Orientation Detection

> **Detect 3D orientation of a baseball from monocular video using seam detection and ellipse-based orientation estimation.**

[![CI](https://github.com/sumeshthakr/robotics/actions/workflows/ci.yml/badge.svg)](https://github.com/sumeshthakr/robotics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-orange)
![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen)

---

## 🎯 What This Does

Given a 30 fps monocular video of a hand-tossed baseball, the system outputs per-frame:
- **Bounding box** — ball location detected by YOLOv8
- **Seam pixels** — detected red stitching in the ball ROI
- **Orientation** — rotation matrix / quaternion / Euler angles from seam distribution
- **3D trajectory** — reconstructed ball path from pinhole camera geometry

### How Orientation is Estimated

| Step | Description |
|------|-------------|
| 1. Detect ball | YOLOv8 nano finds the baseball bounding box |
| 2. Extract seam pixels | Canny edge detection + HSV red color filtering (with circular mask) |
| 3. Fit ellipse | OpenCV `fitEllipse()` on the seam pixel distribution |
| 4. Compute orientation | Ellipse angle → seam direction, axis ratio → seam tilt |
| 5. Build rotation matrix | R = Rz(seam_angle) × Rx(tilt) |

This gives us 2 of 3 rotation degrees of freedom from a single frame. The seam direction tells us which way the stitching runs across the ball, and the tilt tells us how much the seam plane is angled toward or away from the camera.

---

## 🖼️ Detection Results

### Seam-Based Pipeline — Best Frames

| Video 1 | Video 2 |
|:---:|:---:|
| ![Seam Video 1](docs/frames/video1_seam_best.jpg) | ![Seam Video 2](docs/frames/video2_seam_best.jpg) |
| *Red dots = detected seam pixels. Green box = YOLO detection. Euler angles show orientation.* | *Seam pixels detected with ellipse-fitted orientation.* |

---

## 📐 3D Trajectory Reconstruction

Ball 3D position is recovered from the bounding box using the pinhole camera model:
- **Depth:** `Z = fx × D_real / D_pixel` (ball diameter = 74 mm)
- **Lateral:** `X = (cx_img − cx0) × Z / fx`
- **Vertical:** `Y = (cy_img − cy0) × Z / fy`

### Detected Ball Path (from bounding box geometry)

| Video 1 — 3D Trajectory | Video 2 — 3D Trajectory |
|:---:|:---:|
| ![Video 1 Path](docs/frames/video1_detected_path.png) | ![Video 2 Path](docs/frames/video2_detected_path.png) |

### Seam-Based Orientation Arrows

| Video 1 — Seam Orientation | Video 2 — Seam Orientation |
|:---:|:---:|
| ![Video 1 Seam 3D](docs/frames/video1_seam_orientation.png) | ![Video 2 Seam 3D](docs/frames/video2_seam_orientation.png) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BASEBALL ORIENTATION DETECTION                       │
│                     Monocular Video → 3D Orientation + Trajectory           │
└─────────────────────────────────────────────────────────────────────────────┘

  Input: Video Frame (1700×1200 BGR, 30 fps)
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   PREPROCESSING STAGE                     │
  │  camera.py                                                │
  │  ┌─────────────────────────────────────────────────────┐  │
  │  │ load_camera_params() → K (3×3), dist (1×5)         │  │
  │  │ undistort()          → Remove lens distortion       │  │
  │  └─────────────────────────────────────────────────────┘  │
  └──────────────────────┬───────────────────────────────────┘
                         │ Undistorted frame
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    DETECTION STAGE                        │
  │  detector.py                                              │
  │  ┌─────────────────────────────────────────────────────┐  │
  │  │ BallDetector                                        │  │
  │  │  • YOLOv8n (6M params, COCO pre-trained)            │  │
  │  │  • Filter: class 32 ("sports ball"), conf ≥ 0.25    │  │
  │  │  • Output: bbox (x1,y1,x2,y2) + confidence         │  │
  │  ├─────────────────────────────────────────────────────┤  │
  │  │ BallTracker                                         │  │
  │  │  • EMA velocity smoothing (α=0.3)                   │  │
  │  │  • Predict position during lost frames (≤5 frames)  │  │
  │  └─────────────────────────────────────────────────────┘  │
  └──────────────────────┬───────────────────────────────────┘
                         │ bbox + confidence
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                SEAM-BASED PIPELINE                        │
  │  seam_pipeline.py                                         │
  │                                                           │
  │  1. Crop ROI from bbox                                    │
  │  2. Create circular mask (inner 85% of ball)              │
  │  3. Boost HSV saturation (×1.5 for pale seams)            │
  │  4. Canny edge detection (adaptive thresholds)            │
  │  5. HSV dual-range red filter:                            │
  │     hue ∈ [0,25] ∪ [155,180]                              │
  │  6. Combine: edges ∩ red mask                             │
  │  7. Morphological cleanup + dilate                        │
  │                                                           │
  │  ORIENTATION:                                             │
  │  8. Fit ellipse to seam pixel distribution                │
  │  9. Seam angle = ellipse rotation angle                   │
  │  10. Seam tilt = arccos(minor/major axis ratio)           │
  │  11. R = Rz(angle) × Rx(tilt)                             │
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼
  ┌────────────────────────────────────────────────────────────┐
  │                    OUTPUT STAGE                             │
  │  orientation.py                                            │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │ rotation_to_quaternion(R) → [w, x, y, z]            │  │
  │  │ rotation_to_euler(R)      → [roll, pitch, yaw] rad  │  │
  │  └──────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────┘
                         │
                         ▼
  Per-frame result dict:
    ball_detected, bbox, confidence, tracking
    orientation { rotation_matrix, quaternion, euler_angles,
                  seam_angle_deg, seam_tilt_deg }
    seam_pixels (Nx2), num_seam_pixels
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/sumeshthakr/robotics.git
cd robotics

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Process a video with visualization
python main.py spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4 \
    --visualize --output outputs/video1

# Custom confidence threshold
python main.py video.mp4 --confidence 0.3 --output results/
```

### Generating Documentation Outputs

```bash
# Extract best detection frames for documentation
python extract_frames.py

# Generate 3D trajectory and orientation plots
python plot_3d.py
```

### Running Tests & Verification

```bash
# All 29 unit tests
pytest test_all.py -v

# Physical constraint verification (no videos needed)
python verify.py --quick
```

---

## 🗂️ Project Structure

```
robotics/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: lint → test → verify (Python 3.10/3.11)
│
├── camera.py                   # Camera calibration loading + undistortion
├── detector.py                 # YOLOv8 ball detection + EMA velocity tracking
├── orientation.py              # Quaternion/Euler conversion utilities
├── seam_pipeline.py            # Seam detection + ellipse-based orientation
├── main.py                     # CLI entry point
├── extract_frames.py           # Best-frame extractor for documentation
├── plot_3d.py                  # 3D trajectory + orientation visualization
├── test_all.py                 # 29 unit tests (pytest)
├── verify.py                   # Physical-constraint verification
│
├── config/
│   └── camera.json             # fx=10248, dist coeffs, img_shape=(1700,1200,3)
│
├── spin_dataset/               # Two 30 fps spin demo videos
│   ├── raw_spin_video_…_1767711685.mp4   (98 frames — Video 1)
│   └── raw_spin_video_…_1767742221.mp4   (85 frames — Video 2)
│
├── docs/
│   └── frames/                 # Generated detection frames + 3D plots
│       ├── video1_seam_best.jpg
│       ├── video1_detected_path.png
│       ├── video1_seam_orientation.png
│       ├── video2_seam_best.jpg
│       ├── video2_detected_path.png
│       ├── video2_seam_orientation.png
│       └── metrics.json
│
├── REPORT.md                   # Assignment report (system design + math)
├── AI_COLLABORATION_LOG.md     # AI usage documentation
├── requirements.txt            # ultralytics, opencv-python, numpy, scipy, matplotlib
└── yolov8n.pt                  # YOLOv8 nano weights (COCO pre-trained)
```

---

## 🔁 CI/CD Pipeline

The repository uses **GitHub Actions** for continuous integration on every push and pull request.

```
.github/workflows/ci.yml
├── Job: lint-and-test  (Python 3.10 + 3.11 matrix)
│   ├── pip install -r requirements.txt + flake8 + pytest
│   ├── flake8 (syntax errors & undefined names → fail; style → warn)
│   └── pytest test_all.py -v  (29 unit tests)
│
└── Job: quick-verify  (runs after lint-and-test)
    └── python verify.py --quick  (math/model sanity checks, no video needed)
```

**What's tested in CI (29 tests across 5 modules):**

| Module | Tests | What's Validated |
|---|:---:|---|
| `camera.py` | 3 | JSON loading, missing file error, undistort shape preservation |
| `detector.py` | 9 | Init, confidence, output structure, input validation, tracking |
| `seam_pipeline.py` | 11 | Seam detection, 3D seam model geometry, orientation estimation (PCA/ellipse), pipeline init/reset |
| `orientation.py` | 3 | Quaternion identity, Euler identity, 90° rotation conversion |
| Integration | 3 | Pipeline frame processing, video not found, reset |

---

## 📋 Requirements

```
ultralytics>=8.0.0     # YOLOv8 ball detection (COCO pre-trained)
opencv-python>=4.8.0   # Image processing, edge detection, ellipse fitting
numpy>=1.24.0          # Array math, linear algebra
scipy>=1.10.0          # Rotation math (scipy.spatial.transform.Rotation)
matplotlib>=3.7.0      # 3D visualization (plot_3d.py)
```

Python 3.10+ required. Tested on Python 3.10 and 3.11.

---

## 📄 Deliverables

1. **System Design Document** → [`REPORT.md`](REPORT.md) (exposure time, focal length, Hough vs YOLO, bullet spin)
2. **Prototype Code** → This repository (seam-based orientation pipeline)
3. **3D Visualizations** → [`docs/frames/`](docs/frames/) (trajectory plots, orientation arrows)
4. **AI Usage Report** → [`AI_COLLABORATION_LOG.md`](AI_COLLABORATION_LOG.md)
5. **CI/CD Pipeline** → [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
