# Baseball Orientation Detection

Detect baseball 3D orientation from high-speed monocular camera video by analyzing seam patterns and surface motion.

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Run on a video (seam-based approach)
python main.py spin_dataset/video1.mp4 --visualize
# Run optical flow approach
python main.py spin_dataset/video1.mp4 --approach optical --visualize
# Side-by-side comparison of both approaches
python compare.py
```

## Project Structure

```
├── camera.py              # Camera calibration loading + undistortion
├── detector.py            # YOLOv8 ball detection + velocity-based tracking
├── orientation.py         # Rotation tracking, spin rate/axis, format conversions
├── seam_pipeline.py       # Seam-based pipeline (detect seams → PnP → flow RPM)
├── optical_pipeline.py    # Optical flow pipeline (feature tracking → RANSAC rotation)
├── main.py                # CLI entry point
├── compare.py             # Side-by-side comparison video generator
├── test_all.py            # 44 unit tests (pytest)
├── verify.py              # Verification against physical constraints
├── REPORT.md              # Assignment report (Parts 1-3 answers)
├── AI_COLLABORATION_LOG.md # AI usage documentation (Part 4)
├── config/camera.json     # Camera intrinsic parameters
├── spin_dataset/          # Input videos (2 spin demos, 30fps)
├── outputs/               # Generated comparison videos, metrics, frames
├── requirements.txt       # Python dependencies
└── yolov8n.pt             # YOLOv8 nano model weights
```

## Two Approaches

### 1. Seam-Based (default)
Detects red seam stitching via Canny edge detection + HSV color filtering, matches to a parametric 3D seam model, and solves PnP for absolute orientation. Spin rate is computed from optical flow tracking of seam pixels between frames.

### 2. Optical Flow
Tracks corner features on the ball surface using Lucas-Kanade optical flow, then estimates rotation from the flow pattern using the rigid-body equation **v = omega x r** with RANSAC.

## Key Results

| Metric | Video 1 | Video 2 |
|--------|---------|---------|
| Frames | 98 | 85 |
| Ball Detection Rate | 45% | 48% |
| Seam Avg Spin (RPM) | 67 | 170 |
| Optical Avg Spin (RPM) | 54 | 82 |

Both approaches produce physically plausible spin rates for hand-tossed baseballs (expected range: 50-800 RPM), well below the Nyquist limit of 900 RPM at 30fps.

## Testing

```bash
# Unit tests (44 tests)
pytest test_all.py -v

# Verification against physical constraints
python verify.py              # Full verification (includes video processing)
python verify.py --quick      # Math/model checks only (fast)
```

## Requirements

- Python 3.10+
- OpenCV 4.8+
- ultralytics (YOLOv8)
- NumPy, SciPy
- See `requirements.txt` for complete list

## Deliverables

1. **System Design Document**: See `REPORT.md` Part 1
2. **Prototype Code**: This repository (modular 3-part pipeline)
3. **AI Usage Report**: See `AI_COLLABORATION_LOG.md`
