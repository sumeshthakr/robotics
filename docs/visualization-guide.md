# Enhanced Visualization Documentation

**Date:** 2025-02-24

## Overview

The enhanced visualization displays comprehensive baseball orientation analysis results directly on video frames, including ball detection, seams, orientation, and real-world coordinates.

## Visualization Elements

### 1. Header Information

- **Frame number** - Current frame being processed

### 2. Ball Detection

| Element | Description |
|---------|-------------|
| **Green Box** | Ball bounding box (solid green = detected, yellow = tracked) |
| **Blue Dot** | Ball center point |
| **Green Text** | Pixel coordinates of ball center |

### 3. Seam Detection (Seam Approach Only)

| Element | Description |
|---------|-------------|
| **Red Dots** | Detected seam pixel locations |
| **Seam Pixels count** | Number of seam pixels found |

### 4. Orientation Information

| Element | Description |
|---------|-------------|
| **Quaternion** | [w, x, y, z] rotation quaternion (scalar-first) |
| **Euler Angles** | [roll, pitch, yaw] in degrees |
| **Spin Rate** | Rotation speed in RPM |
| **Spin Axis** | Purple arrow showing rotation axis, with vector values |

### 5. Real-World Coordinates

```
Position (mm): X={real_x} Y={real_y} Z={real_z}
```

Calculated using camera calibration and baseball radius (37mm):
- **X, Y**: Horizontal position relative to camera optical center
- **Z**: Depth distance from camera (derived from ball size in pixels)

### 6. Additional Information

| Element | Description |
|---------|-------------|
| **Confidence** | Detection confidence (green >0.5, yellow <0.5) |
| **Flow Confidence** | Optical flow tracking confidence (optical approach only) |
| **Status** | "DETECTED" or "PREDICTED" (tracking) |
| **Method** | "Seam Detection + PnP" or "Optical Flow" |

### 7. Legend (Bottom Left)

Shows visual legend for all displayed elements.

## Sample Output Frame

![Sample visualization](https://maas-log-prod.cn-wlcb.ufileos.com/anthropic/76c6a5db-e09e-4b17-b901-5e16a856f0d2/sample_enhanced_viz.jpg)

## Real-World Coordinate Calculation

The 3D position is calculated using pinhole camera model:

```python
# Depth from ball size
focal_length = K[0,0]  # from camera matrix
ball_radius_mm = 37  # standard baseball radius
ball_radius_px = (bbox_width / 2)

depth_mm = (focal_length * ball_radius_mm) / ball_radius_px

# Position relative to camera center
real_x = (ball_center_x - K[0,2]) * depth_mm / focal_length
real_y = (ball_center_y - K[1,2]) * depth_mm / focal_length
real_z = depth_mm
```

## Output Format for Analysis

The visualization is meant for visual inspection, but all numerical data is also available in JSON format for detailed analysis:

```json
{
  "frame_number": 30,
  "timestamp": 1.0,
  "ball_detected": true,
  "bbox": [868, 757, 912, 806],
  "confidence": 0.67,
  "seam_pixels": 567,
  "orientation": {
    "rotation_matrix": [[3x3 array]],
    "quaternion": [w, x, y, z],
    "euler_angles": [roll, pitch, yaw]
  },
  "spin_rate": 593.1,
  "spin_axis": [0.12, -0.34, 0.93],
  "position_mm": {
    "x": -123.4,
    "y": 45.6,
    "z": 1245.2
  }
}
```

## Usage Examples

```bash
# Seam approach with enhanced visualization
python main.py video.mp4 --approach seam --visualize

# Optical flow approach with enhanced visualization
python main.py video.mp4 --approach optical --visualize --max-corners 100

# Both approaches produce the same output format for comparison
```

## Reusability

The visualization components can be used independently:

```python
from src.pipeline import BaseballOrientationPipeline
from src.pipeline_optical import OpticalFlowPipeline

# Both return structured data with:
# - bbox: (x1, y1, x2, y2) - pixel coordinates
# - orientation: {rotation_matrix, quaternion, euler_angles}
# - spin_rate: RPM
# - spin_axis: 3D unit vector
# - confidence: detection/tracking confidence

# Real-world position calculation is done in both pipelines
# using the same camera calibration data
```

## Notes

- Real-world coordinates are **approximate** - they assume the baseball has the standard 37mm radius
- Z coordinate represents depth from camera optical center
- For accurate 3D reconstruction, use stereo cameras or known ball trajectory
