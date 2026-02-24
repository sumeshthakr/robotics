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
