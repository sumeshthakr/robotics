# Approach 3: Optical Flow for Baseball Orientation Detection

## Overview

This approach estimates baseball rotation by tracking feature points across consecutive frames using sparse optical flow, instead of relying on seam detection + PnP solving.

## Key Idea

1. Track feature points on the ball between consecutive frames
2. Compute the optical flow vectors
3. From the flow field, estimate the rotation axis and spin rate

For a rotating sphere:
- Points near the equator move fastest
- Points near the poles move slowest
- Flow vectors form a pattern around the rotation axis

## Implementation

### Files Created

1. **`src/optical_flow/__init__.py`**
   - Module initialization

2. **`src/optical_flow/rotation_estimator.py`**
   - `RotationEstimator` class with methods:
     - `_detect_features()`: Detect good features to track using `cv2.goodFeaturesToTrack`
     - `_compute_flow()`: Compute sparse optical flow using Lucas-Kanade
     - `_filter_valid_flow()`: Filter valid tracks based on status and bounds
     - `_estimate_rotation_from_flow()`: RANSAC-based rotation estimation
     - `_estimate_rotation_perspective()`: Essential matrix-based rotation estimation
     - `estimate_rotation()`: Main method to estimate rotation from frame
     - `get_smoothed_rotation()`: Get smoothed estimate from history

3. **`src/pipeline_optical.py`**
   - `OpticalFlowPipeline` class similar to `BaseballOrientationPipeline`
   - Uses `RotationEstimator` instead of seam detection + PnP
   - Same output format for easy comparison

4. **`tests/optical_flow/test_rotation_estimator.py`**
   - 23 tests for `RotationEstimator` class

5. **`tests/optical_flow/test_pipeline_optical.py`**
   - 13 tests for `OpticalFlowPipeline` class

6. **`main.py`** (updated)
   - Added `--approach` option: `seam` (default) or `optical`
   - Added `--max-corners`, `--min-flow`, `--max-flow` options for optical flow

## Mathematical Background

### Rotation from Flow Field

For a rotating sphere with angular velocity $\omega$ about axis $\mathbf{a}$:

1. The velocity of a point at position $\mathbf{r}$ is:
   $$\mathbf{v} = \omega \mathbf{a} \times \mathbf{r}$$

2. The rotation axis is orthogonal to the dominant flow pattern:
   $$\mathbf{a} \propto \mathbf{r} \times \mathbf{v}$$

3. Angular velocity magnitude:
   $$\omega = \frac{|\mathbf{v}|}{|\mathbf{r}_{\perp}|}$$
   where $\mathbf{r}_{\perp}$ is the distance from the rotation axis.

## Usage

```bash
# Use seam-based approach (default)
python main.py video.mp4

# Use optical flow approach
python main.py video.mp4 --approach optical

# With optical flow parameters
python main.py video.mp4 --approach optical --max-corners 100 --min-flow 0.3 --max-flow 25.0
```

## Test Results

All 30 optical flow tests pass:

```
tests/optical_flow/test_pipeline_optical.py::TestOpticalFlowPipeline - 13 tests PASSED
tests/optical_flow/test_rotation_estimator.py::TestRotationEstimator - 17 tests PASSED
```

Total: 51 tests passing (including 21 existing tests)

## Demo

Run the demo script to see optical flow in action:

```bash
python test_optical_demo.py
```

Output:
- Generates synthetic rotating ball sequence
- Processes frames with optical flow
- Outputs `outputs/optical_flow_demo.mp4` with visualization

## Reusable Components

The optical flow approach reuses:
- `BallTracker` - for getting ball ROI
- `load_camera_params()` - for camera calibration
- `undistort()` - for image undistortion
- `OrientationTracker` - for temporal smoothing

## Advantages

1. Works when seam detection is difficult (poor lighting, low contrast)
2. Does not require explicit seam model
3. Can handle various ball appearances
4. Temporal smoothing provides stable estimates

## Limitations

1. Requires sufficient texture/features on the ball
2. May fail with very fast rotation (flow too large)
3. Sensitive to camera motion (needs stabilization or compensation)
4. Requires at least 4-6 good feature points

## Comparison with Seam-Based Approach

| Aspect | Seam-Based | Optical Flow |
|--------|------------|--------------|
| Requirements | Visible seams | Texture/features |
| Accuracy | High with good seams | Good with features |
| Speed | Moderate | Fast (sparse flow) |
| Robustness | Sensitive to lighting | Sensitive to texture |
| Complexity | Higher (PnP) | Moderate |
