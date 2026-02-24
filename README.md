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

### Approach 1: Sphere Fitting + Seam Template Matching

**Status:** NOT WORKING with current setup

| Video | Detection Rate | Orientation Success |
|-------|---------------|-------------------|
| video1 | 4.1% (4/98 frames) | 0% |
| video2 | 7.1% (6/85 frames) | 0% |

**Issues:**
- YOLOv8n (nano) model cannot reliably detect small baseballs
- Ball is 75-130 pixels in 1700x1200 frame (~5% of image)
- Pretrained COCO model not optimized for baseball
- Without ball detection, orientation estimation is impossible

**See:** `docs/analysis/approach1-results.md` for detailed analysis

### Approach 3: Optical Flow

**Status:** IN PROGRESS

## Next Steps

- [x] ~~Implement Approach 1~~ - Completed but ineffective
- [ ] Implement Approach 3 (Optical Flow)
- [ ] Compare results between approaches
- [ ] Fine-tune YOLO on baseball dataset (if results promising)
