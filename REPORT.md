# Baseball Orientation Detection — Assignment Report

## Part 1: System Design

### 1.1 Maximum Allowable Exposure Time (Shutter Speed)

**Goal:** Keep motion blur of the red seams under **2 pixels** when the ball travels at **100 MPH**.

**Calculation:**

First, determine the ball's image-plane velocity in pixels/second.

Given parameters:
- Ball speed: 100 MPH = **44.70 m/s**
- Distance from camera: 3–5 m (use worst-case 3 m for max blur)
- Camera focal length: fx = **10,248 px** (from our calibration)

The image-plane velocity of the ball is:

$$v_{\text{image}} = \frac{f_x \times v_{\text{ball}}}{d} = \frac{10{,}248 \times 44.70}{3.0} \approx 153{,}035 \ \text{px/s}$$

For blur ≤ 2 pixels:

$$t_{\text{exposure}} \leq \frac{2 \ \text{px}}{v_{\text{image}}} = \frac{2}{153{,}035} \approx 13.1 \ \mu\text{s}$$

**Answer:** The maximum exposure time is approximately **13 microseconds** (1/76,500 s). This is achievable with high-speed cameras (e.g., Phantom or Photron series) paired with high-intensity strobe lighting to compensate for the extremely short exposure.

At 5 meters distance:

$$v_{\text{image}} = \frac{10{,}248 \times 44.70}{5.0} \approx 91{,}617 \ \text{px/s}$$

$$t_{\text{exposure}} \leq \frac{2}{91{,}617} \approx 21.8 \ \mu\text{s}$$

So the exposure should be **13–22 μs** depending on distance.

---

### 1.2 Lens Focal Length and Sensor Resolution for 200px Ball at 5m

**Goal:** Ball occupies at least **200 pixels in diameter** at **5 meters** distance.

Using the pinhole camera model:

$$D_{\text{pixel}} = \frac{f \times D_{\text{real}}}{d}$$

Where:
- $D_{\text{real}}$ = baseball diameter = 74.0 mm = 0.074 m
- $d$ = 5 m
- $D_{\text{pixel}}$ = 200 px (minimum requirement)

Solving for focal length:

$$f = \frac{D_{\text{pixel}} \times d}{D_{\text{real}}} = \frac{200 \times 5.0}{0.074} \approx 13{,}514 \ \text{px}$$

Converting to physical units: If the sensor has a pixel pitch of $p$ μm, the physical focal length is:

$$f_{\text{mm}} = f_{\text{px}} \times p$$

**Sensor resolution requirement:** The ball must fit within the frame while occupying 200 px. The ball's image at 5m with $f$ = 13,514 px has angular extent:

$$\theta_{\text{ball}} = 2 \times \arctan\left(\frac{0.037}{5.0}\right) \approx 0.85°$$

Minimum field of view needs to cover the pitch corridor (≈1–2m wide at 5m):

$$\theta_{\text{FOV}} = 2 \times \arctan\left(\frac{1.0}{5.0}\right) \approx 22.6°$$

Required sensor width in pixels:

$$W_{\text{sensor}} = \frac{200 \times 1.0}{0.074} \approx 2{,}703 \ \text{px}$$

**Answer:** A focal length of approximately **13,500 pixels** (roughly a **100mm lens** on a sensor with ~7.5 μm pixel pitch) with a sensor resolution of at least **2,700 × 2,000 pixels** would meet this requirement. Our actual camera (fx = 10,248 px) yields ~153 px ball diameter at 5m, slightly under the 200 px target — a slightly longer lens or closer placement would be needed.

---

### 1.3 Hough Circle Transform vs. Deep Learning (YOLO/Segmentation)

| Criterion | Hough Circle Transform | YOLO / Deep Learning |
|---|---|---|
| **Speed** | Very fast (~1ms), no GPU needed | Needs GPU for real-time; ~5-15ms with optimization |
| **Accuracy** | Assumes perfectly circular shape; fails with partial occlusion, elliptical perspective | Handles occlusion, non-circular views, varied backgrounds |
| **Robustness** | Sensitive to edge detection thresholds, noise, motion blur | Trained on diverse conditions; naturally robust |
| **Edge Deployment** | Trivially deployable on ARM/embedded (pure OpenCV) | Requires model optimization (TensorRT, ONNX, quantization) |
| **Calibration** | Needs manual tuning of (min_radius, max_radius, dp, minDist) per setup | Single model works across setups after training |
| **Multi-ball** | Returns all circles; needs post-filtering | Natively detects multiple objects with confidence scores |
| **Latency** | Deterministic, predictable timing | Inference time varies by model size and hardware |

**Recommendation for this system:** We chose **YOLO** because:
1. The ball may not appear perfectly circular (perspective distortion, motion blur)
2. YOLO's pretrained COCO model already includes "sports ball" (class 32)
3. The bounding box naturally provides the ROI for seam extraction
4. For production, YOLOv8n (6M params) can be quantized to run at >100fps on edge GPUs (Jetson)

**When Hough is better:** If deploying on an FPGA or microcontroller with no ML accelerator, and the ball is always well-framed (controlled lab setup), Hough Circle Transform is the simpler, more deterministic choice.

---

### 1.4 Bullet Spin Detection Challenge

**Problem:** When the rotation axis is parallel to the optical axis (ball spinning like a rifled bullet, coming straight at the camera), the seam pattern appears **stationary** in the image. The seams don't move across the visible hemisphere — they only rotate about the center point.

**Why this is hard:**
- The seam pattern on the visible face appears identical from frame to frame
- Optical flow detects zero motion (or only sub-pixel tangential motion)
- PnP-based orientation sees the same 2D seam projection each frame
- Standard spin rate estimation yields ~0 RPM despite the ball spinning rapidly

**Detection logic changes:**

1. **Radial seam symmetry analysis**: For bullet spin, seam curvature near the pole viewed head-on has a characteristic rotational symmetry. Track how the seam's angular orientation (measured from center of ball) changes over frames — even when the seam pattern looks "static", its rotational registration will shift.

2. **Sub-pixel angular tracking**: Use phase correlation or Fourier-based rotation estimation on the ball ROI:
   - Convert ball patch to polar coordinates (r, θ)
   - Track the θ-shift between frames using cross-correlation
   - This detects rotation even when the whole pattern just rotates in place

3. **Multi-camera solution**: In practice, a single monocular camera fundamentally cannot measure bullet spin from the front. The standard solution at MLB stadiums (Hawk-Eye, Trackman) uses either:
   - Multiple cameras at different angles (triangulation of spin axis)
   - Doppler radar (measures spin via frequency shift of seam returns)

4. **Texture-based rotation**: If the ball has any asymmetric markings (logos, dirt), track those features across frames. With pure bullet spin, the ball face doesn't change but the markings rotate around the center.

**In our pipeline:** Our optical flow approach would measure ~0 RPM for pure bullet spin because the surface features barely translate in image space. The seam pipeline would similarly fail because seam positions don't shift. Detection of bullet spin requires either multi-camera geometry or the Fourier polar approach described above.

---

## Part 2: Basic Implementation

### 2.1 Circular Boundary Detection (Ball Detection)

**Implementation:** `detector.py` — `BallDetector` class

We use **YOLOv8** (nano variant, 6M parameters) pretrained on COCO to detect the "sports ball" class. This provides a bounding box around the baseball.

```python
# From detector.py
class BallDetector:
    SPORTS_BALL_CLASS = 32  # COCO class index
    
    def detect(self, image):
        results = self.model(image, verbose=False)
        # Find highest-confidence sports ball detection
        # Returns: {"detected": bool, "bbox": (x1,y1,x2,y2), "confidence": float}
```

**Robustness to noise and motion blur:**
- YOLO is trained on diverse images including blurry/noisy conditions
- We wrap detection in `BallTracker` which uses **velocity-based prediction** (exponential moving average) during brief detection failures (up to 5 frames)
- Confidence threshold (default 0.25) is low enough to catch blurred balls

**Why not Hough:** As discussed in Part 1.3, Hough requires careful threshold tuning and assumes a clean circular edge, which motion blur destroys.

### 2.2 Seam Pixel Extraction from Cropped Ball Region

**Implementation:** `seam_pipeline.py` — `detect_seams()` function

```python
def detect_seams(roi, canny_low=30, canny_high=100):
    # 1. Boost saturation (pale seams in video)
    # 2. Canny edge detection with adaptive thresholds
    # 3. HSV red color filtering (hue 0-20 | 160-180)
    # 4. Combine: keep edges that are also red
    # 5. Morphological cleanup
    # Returns: {"seam_pixels": Nx2, "num_pixels": int, "edges": mask}
```

**Key design choices:**
- **Saturation boost** (1.5×): Baseball seams appear pale under strobe lighting
- **Dual hue range**: Red wraps around 0°/180° in HSV, so we check hue [0,20] ∪ [160,180]
- **Adaptive thresholds**: For small ROIs (<60px), lower Canny thresholds and saturation requirements
- **Fallback**: If color filter removes >70% of edges, we fall back to all edges (handles unusual lighting)

### 2.3 Full Pipeline: Raw Image → Seam Coordinates

**Implementation:** `seam_pipeline.py` — `SeamPipeline` class

The complete pipeline chains all steps:

1. **Undistort** image using camera calibration (`camera.py`)
2. **Detect ball** with YOLO + tracker (`detector.py`)
3. **Crop ROI** around the detected bounding box
4. **Extract seams** using color-filtered Canny edges
5. **Convert** ROI-local seam coordinates to full-frame coordinates

```python
pipeline = SeamPipeline(K, dist)
result = pipeline.process_frame(frame, timestamp=0.0)
# result["seam_pixels"]  → Nx2 array of seam pixel locations
# result["bbox"]         → ball bounding box
```

**Robustness:**
- Handles motion blur via YOLO's robustness + tracker prediction
- Handles lighting variation via saturation boosting + adaptive thresholds
- Handles small balls via adaptive Canny/HSV parameters

---

## Part 3: Bonus Implementation

### 3.4 3D Orientation from Seam Segments

**Implementation:** `seam_pipeline.py` — PCA + ellipse-based orientation estimation

**Approach — Ellipse fitting on seam pixel distribution:**

Instead of trying to match 2D seam pixels to a 3D model (which requires solving the correspondence problem), we use the statistical distribution of the detected seam pixels to determine orientation:

1. **Detect seam pixels** using Canny + HSV red filtering (with circular mask)
2. **Fit an ellipse** to the seam pixel distribution using `cv2.fitEllipse()`
3. **Extract orientation** from the ellipse parameters:
   - **Seam angle** = ellipse rotation angle (in-plane direction of seam)
   - **Seam tilt** = arccos(minor_axis / major_axis) (out-of-plane tilt)
4. **Build rotation matrix** R = Rz(seam_angle) × Rx(tilt)

```python
# seam_pipeline.py — estimate_orientation_from_seams()
ellipse = cv2.fitEllipse(seam_pixels)
(cx, cy), (minor, major), angle = ellipse
tilt = arccos(minor / major)     # How tilted the seam plane is
R = Rz(angle) @ Rx(tilt)        # Construct rotation matrix
```

This recovers 2 of 3 rotation degrees of freedom from a single frame. The third (spin around the seam plane normal) would require frame-to-frame tracking.

**Why this approach:**
- Simple and geometrically meaningful
- No correspondence problem to solve
- Uses standard OpenCV functions (`fitEllipse`, PCA)
- Produces valid rotation matrices by construction (product of two rotation matrices)
- Easy for a student to understand and implement

### Physical Consistency Check (5 Consecutive Frames)

The verification script (`verify.py`) checks that frame-to-frame orientation changes are physically plausible:

```python
R_relative = R_prev.T @ R_curr     # How much did it rotate?
angle = ||rotvec(R_relative)||      # Total rotation angle
# At 30fps, max ~90 deg/frame is physical limit
```

### 3.5 Test Case: Rotation Matrix Validity

**Implementation:** `test_all.py` — `TestOrientationEstimation`

```python
def test_returns_valid_rotation(self):
    """Rotation matrix should be proper (det=1, R^T R = I)."""
    result = estimate_orientation_from_seams(seam_pixels, roi_shape)
    R = result["rotation_matrix"]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)  # Orthogonal
    assert abs(np.linalg.det(R) - 1.0) < 1e-10          # Proper rotation
```

Additionally, `verify.py` contains `check_rotation_math()` that validates:
1. Rotation matrices from 24 axis-angle combinations match OpenCV and SciPy (max error < 1e-10)
2. Orthogonality ($R^T R = I$) and proper rotation ($\det(R) = 1$)
3. Quaternion/Euler roundtrip conversions (20 random rotations)

---

## Part 4: AI Collaboration Log

See **AI_COLLABORATION_LOG.md** for the complete log.

---

## Summary of Results

| Metric | Video 1 | Video 2 |
|--------|---------|---------|
| Frames | 98 | 85 |
| Ball Detection Rate | ~46% | ~48% |
| Seam Orientation Rate | ~44% | ~48% |

The seam pipeline successfully detects ball orientation (rotation matrix / quaternion / Euler angles) on detected frames using ellipse-based seam distribution analysis.
