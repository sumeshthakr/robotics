# Approach 1 Results Analysis

**Date:** 2025-02-24
**Approach:** Sphere Fitting + Seam Template Matching
**Model:** YOLOv8n (pretrained on COCO)

## Video Processing Results

### Video 1: `raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4`

| Metric | Value |
|--------|-------|
| Duration | 3.27s |
| Total Frames | 98 |
| FPS | 30.0 |
| Ball Detections | 4/98 (4.1%) |
| Orientation Estimates | 0/98 (0%) |
| Avg Bbox Size | 17,061 pixels² (~130x130 px) |
| Confidence Range | 0.501 - 0.671 |

**Detected Frames:** 29, 31, 34, 38

### Video 2: `raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4`

| Metric | Value |
|--------|-------|
| Duration | 2.83s |
| Total Frames | 85 |
| FPS | 30.0 |
| Ball Detections | 6/85 (7.1%) |
| Orientation Estimates | 0/85 (0%) |
| Avg Bbox Size | 5,804 pixels² (~75x75 px) |
| Confidence Range | 0.507 - 0.598 |

**Detected Frames:** 35, 39, 40, 41, 45, 48

## Key Findings

### 1. Ball Detection Performance
- **Detection Rate:** 4-7% (very low)
- **Confidence:** 0.5-0.67 (barely above threshold)
- **Issue:** Ball is detected in only a few consecutive frames, then lost

### 2. Orientation Estimation
- **Success Rate:** 0%
- **Reason:** No seam orientation could be estimated because ball detection was too sparse

### 3. Visual Analysis
From extracted frames:
- Frame 0: No ball visible (scene setup)
- Frame 29: Ball detected in upper right, small size
- Frame 50: Ball present but not detected

The baseball appears as a small object (75-130 pixels wide) in the 1700x1200 frame.

## Root Causes

### Primary Issue: YOLOv8n Limitations
1. **Training Domain Mismatch:**
   - YOLOv8n trained on COCO dataset
   - "Sports ball" class (32) includes various balls (soccer, tennis, basketball)
   - Not specifically tuned for baseball seams/spin

2. **Small Object Detection:**
   - Nano model has limited capacity for small objects
   - Baseball is ~4-8% of image dimensions
   - YOLOv8n struggles with objects <5% of frame size

3. **Motion Blur:**
   - High-speed video (30 fps but actual capture may be faster)
   - Ball rotation causes seam blur
   - Red seams become less distinct

### Secondary Issues
1. **Seam Detection Challenges:**
   - Even when ball is detected, seam pixels are sparse
   - Canny edge detection picks up many non-seam edges
   - Red color filtering is sensitive to lighting

2. **No Temporal Smoothing:**
   - Each frame processed independently
   - No tracking between detections
   - Cannot interpolate orientation when ball is lost

## Recommendations for Approach 1

### Immediate Improvements
1. **Use Larger YOLO Model:**
   - YOLOv8s or YOLOv8m for better small object detection
   - Trade-off: slower inference but better accuracy

2. **Lower Confidence Threshold:**
   - Current: 0.5
   - Try: 0.3-0.4 with non-maximum suppression

3. **Add Temporal Tracking:**
   - Kalman filter to predict ball position
   - Optical flow to track between detections
   - Interpolate orientation during gaps

### Medium-Term Improvements
1. **Fine-tune YOLO on Baseball Dataset:**
   - Collect/buy labeled baseball images
   - Fine-tune YOLOv8 specifically for baseball detection
   - Annotate seam locations for orientation

2. **Improve Seam Detection:**
   - Use seam-specific color profiles
   - Implement learned edge detection (CNN-based)
   - Add morphological operations to clean edges

3. **Background Subtraction:**
   - Use moving ball as key differentiator
   - Subtract static background to isolate ball

## Conclusion

**Approach 1 Status:** NOT WORKING for current setup

The pretrained YOLOv8n model is not suitable for baseball detection in this video format. The pipeline architecture is sound, but the ball detection component fails too frequently to enable orientation estimation.

**Next Steps:**
1. Implement Approach 3 (Optical Flow) as alternative
2. Compare results between approaches
3. Consider training custom baseball detector if either approach shows promise

---

## Appendix: Frame Samples

| Frame | Detected | Notes |
|-------|----------|-------|
| 0 | No | Scene setup, no ball visible |
| 29 | Yes | Ball in upper right, ~130px bbox |
| 50 | No | Ball present but not detected (small/blur) |
