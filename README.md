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

## Results Summary

### Approach 1: Sphere Fitting + Seam Template Matching (REFINED)

**Status:** ✅ WORKING after refinements

| Video | Detection Rate | Orientation Rate | Avg Spin Rate |
|-------|---------------|------------------|---------------|
| video1 | 50.0% (49/98) | 46.9% (46/98) | 593 RPM |
| video2 | 54.1% (46/85) | 54.1% (46/85) | 637 RPM |

**Improvements from baseline:**
- Lowered confidence threshold: 0.5 → 0.25
- Added temporal ball tracking (velocity prediction)
- Improved seam detection with adaptive thresholds + color boosting

**See:** `docs/analysis/refined-results.md` for detailed analysis

### Approach 3: Optical Flow

**Status:** PENDING

## Next Steps

- [x] ~~Implement Approach 1~~ - Completed and refined
- [ ] Implement Approach 3 (Optical Flow)
- [ ] Compare results between approaches
- [ ] Fine-tune YOLO on baseball dataset
