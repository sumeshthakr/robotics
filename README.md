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
# Seam-based approach (default)
python main.py video.mp4 --visualize

# Optical flow approach
python main.py video.mp4 --approach optical --visualize

# Process spin dataset
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
│   ├── detection/       # Ball detection and tracking
│   ├── preprocessing/   # Image preprocessing
│   ├── seams/          # Seam detection and 3D model
│   ├── optical_flow/   # Optical flow rotation estimation
│   ├── estimation/     # Orientation estimation (PnP)
│   ├── tracking/       # Temporal orientation tracking
│   └── utils/          # Camera utilities
├── config/             # Camera parameters
├── tests/              # Unit tests (80+ tests, all passing)
└── main.py            # Entry point (supports both approaches)
```

## Running Tests

```bash
pytest tests/ -v
```

## Results Summary

### Approach Comparison

| Approach | Orientation Rate | Spin Rate | Winner |
|----------|-----------------|-----------|--------|
| **Seam (PnP)** | **50.5%** | **615 RPM** | ✅ |
| **Optical Flow** | 45.6% | 470 RPM | - |

**Seam Detection + PnP is recommended** for production use.

**Detailed comparison:** `docs/analysis/approach-comparison.md`

### Reusable Components

Both approaches share these core modules:
- `BallDetector` + `BallTracker` - Ball detection and temporal tracking
- `load_camera_params()` - Camera calibration utilities
- `OrientationTracker` - Spin rate/axis from rotation sequence

### Code Snippets

```python
# Ball tracking (both approaches)
from src.detection.ball_tracker import BallTracker
tracker = BallTracker(detector, max_lost_frames=10)
result = tracker.track(frame)  # Returns bbox with velocity prediction

# Optical flow estimation
from src.optical_flow.rotation_estimator import RotationEstimator
estimator = RotationEstimator(camera_matrix=K)
result = estimator.estimate_rotation(frame_gray, bbox, timestamp)
```

## Next Steps

- [x] ~~Implement Approach 1~~ - Completed and refined
- [x] ~~Implement Approach 3~~ - Completed
- [x] ~~Compare results~~ - Seam approach wins
- [ ] Fine-tune YOLO on baseball dataset
- [ ] Implement hybrid approach (seam + optical flow fusion)
