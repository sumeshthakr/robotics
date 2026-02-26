#!/usr/bin/env python3
"""Extract best detection frames from both pipelines for README documentation.

Processes both spin_dataset videos with both approaches and saves:
  - Best seam detection frames (highest seam pixel count + ball detected)
  - Best optical flow frames (highest tracked features + ball detected)
  - Composite comparison frames
  - Per-video JSON metrics

Output is saved to docs/frames/ for use in the README.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from camera import load_camera_params
from seam_pipeline import SeamPipeline
from optical_pipeline import OpticalFlowPipeline


OUTPUT_DIR = Path("docs/frames")
CAMERA_JSON = "config/camera.json"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.25


def score_seam_frame(result):
    """Score a seam-pipeline frame — higher = better visual for README."""
    if not result["ball_detected"]:
        return -1
    score = result["num_seam_pixels"]
    if result["orientation"] is not None:
        score += 500
    if result["spin_rate"] is not None:
        score += 200
    return score


def score_optical_frame(result):
    """Score an optical-flow frame — higher = better visual for README."""
    if not result["ball_detected"]:
        return -1
    tf = result.get("tracked_features")
    tracked = len(tf["prev_points"]) if tf else 0
    score = tracked * 10
    if result["orientation"] is not None:
        score += 500
    if result["spin_rate"] is not None:
        score += 200
    conf = result.get("flow_confidence") or 0
    score += int(conf * 100)
    return score


def annotate_seam_frame(frame, result, frame_idx, video_label):
    """Draw rich seam-pipeline annotations on a frame for README use."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header bar
    cv2.rectangle(vis, (0, 0), (w, 40), (20, 20, 60), -1)
    cv2.putText(vis, f"SEAM-BASED PIPELINE  |  {video_label}  |  Frame {frame_idx}",
                (10, 28), font, 0.65, (0, 200, 255), 2)

    if not result["ball_detected"]:
        return vis

    x1, y1, x2, y2 = result["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rad = max((x2 - x1), (y2 - y1)) // 2

    # Ball circle overlay
    color = (0, 255, 0) if not result.get("tracking") else (0, 255, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.circle(vis, (cx, cy), rad, color, 1)
    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

    # Seam pixels (red dots)
    if result.get("seam_pixels") is not None and len(result["seam_pixels"]) > 0:
        for px, py in result["seam_pixels"]:
            gx, gy = int(px + x1), int(py + y1)
            if 0 <= gx < w and 0 <= gy < h:
                cv2.circle(vis, (gx, gy), 2, (0, 0, 255), -1)

    # Info panel
    y_txt = 60
    info_lines = [
        (f"Seam pixels: {result['num_seam_pixels']}", (100, 180, 255)),
    ]
    if result.get("confidence"):
        info_lines.append((f"YOLO conf: {result['confidence']:.2f}", (180, 255, 180)))
    if result["spin_rate"] is not None:
        info_lines.append((f"Spin rate: {result['spin_rate']:.1f} RPM", (0, 255, 255)))
    if result["spin_axis"] is not None:
        ax = result["spin_axis"]
        info_lines.append((f"Spin axis: ({ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f})", (200, 200, 255)))
    if result["orientation"]:
        e = np.degrees(result["orientation"]["euler_angles"])
        info_lines.append((f"Euler: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] deg",
                           (255, 200, 100)))
        q = result["orientation"]["quaternion"]
        info_lines.append((f"Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]",
                           (255, 200, 100)))

    for text, col in info_lines:
        cv2.putText(vis, text, (10, y_txt), font, 0.55, col, 1)
        y_txt += 22

    # Spin-axis arrow
    if result["spin_axis"] is not None:
        axis = result["spin_axis"]
        end_x = int(cx + axis[0] * 70)
        end_y = int(cy + axis[1] * 70)
        cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (255, 0, 255), 3, tipLength=0.3)
        cv2.putText(vis, "spin axis", (end_x + 5, end_y), font, 0.4, (255, 0, 255), 1)

    return vis


def annotate_optical_frame(frame, result, frame_idx, video_label):
    """Draw rich optical-flow annotations on a frame for README use."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header bar
    cv2.rectangle(vis, (0, 0), (w, 40), (40, 20, 20), -1)
    cv2.putText(vis, f"OPTICAL FLOW PIPELINE  |  {video_label}  |  Frame {frame_idx}",
                (10, 28), font, 0.65, (255, 200, 0), 2)

    if not result["ball_detected"]:
        return vis

    x1, y1, x2, y2 = result["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rad = max((x2 - x1), (y2 - y1)) // 2

    color = (255, 255, 0) if not result.get("tracking") else (0, 255, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.circle(vis, (cx, cy), rad, color, 1)
    cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)

    # Flow vectors
    if result.get("tracked_features"):
        tracked = result["tracked_features"]
        for p1, p2 in zip(tracked["prev_points"], tracked["curr_points"]):
            pt1 = (int(p1[0] + x1), int(p1[1] + y1))
            pt2 = (int(p2[0] + x1), int(p2[1] + y1))
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.arrowedLine(vis, pt1, pt2, (0, 255, 255), 1, tipLength=0.4)

    y_txt = 60
    info_lines = []
    tf = result.get("tracked_features")
    if tf:
        info_lines.append((f"Tracked features: {len(tf['prev_points'])}", (100, 255, 255)))
    if result.get("confidence"):
        info_lines.append((f"YOLO conf: {result['confidence']:.2f}", (180, 255, 180)))
    if result["spin_rate"] is not None:
        info_lines.append((f"Spin rate: {result['spin_rate']:.1f} RPM", (0, 255, 255)))
    if result["spin_axis"] is not None:
        ax = result["spin_axis"]
        info_lines.append((f"Spin axis: ({ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f})", (200, 200, 255)))
    if result.get("flow_confidence") is not None:
        info_lines.append((f"Flow conf: {result['flow_confidence']:.2f}", (100, 255, 200)))
    if result["orientation"]:
        e = np.degrees(result["orientation"]["euler_angles"])
        info_lines.append((f"Euler: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] deg",
                           (255, 200, 100)))
        q = result["orientation"]["quaternion"]
        info_lines.append((f"Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]",
                           (255, 200, 100)))

    for text, col in info_lines:
        cv2.putText(vis, text, (10, y_txt), font, 0.55, col, 1)
        y_txt += 22

    if result["spin_axis"] is not None:
        axis = result["spin_axis"]
        end_x = int(cx + axis[0] * 70)
        end_y = int(cy + axis[1] * 70)
        cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (255, 0, 255), 3, tipLength=0.3)
        cv2.putText(vis, "spin axis", (end_x + 5, end_y), font, 0.4, (255, 0, 255), 1)

    return vis


def make_comparison_frame(seam_frame, optical_frame, seam_result, opt_result,
                          video_label, frame_idx, total_frames):
    """Create a side-by-side comparison frame with a stats bar."""
    target_h = 480
    target_w_half = 640

    left = cv2.resize(seam_frame, (target_w_half, target_h))
    right = cv2.resize(optical_frame, (target_w_half, target_h))

    # Divider
    divider = np.zeros((target_h, 4, 3), dtype=np.uint8)
    divider[:] = (120, 120, 120)
    combined = np.hstack([left, divider, right])

    # Stats bar
    bar_h = 100
    bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
    bar[:] = (25, 25, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    mid = combined.shape[1] // 2

    cv2.putText(bar, f"{video_label}  |  Frame {frame_idx}/{total_frames}",
                (10, 22), font, 0.55, (200, 200, 200), 1)
    cv2.line(bar, (mid, 0), (mid, bar_h), (80, 80, 80), 1)

    # Seam stats
    s_spin = f"{seam_result['spin_rate']:.0f} RPM" if seam_result.get("spin_rate") else "—"
    s_seam = str(seam_result.get("num_seam_pixels", 0))
    cv2.putText(bar, f"SEAM: {s_seam} px | {s_spin}",
                (10, 55), font, 0.55, (0, 200, 255), 1)
    cv2.putText(bar, "Seam detection + PnP + Flow RPM",
                (10, 80), font, 0.4, (150, 150, 150), 1)

    # Optical stats
    o_spin = f"{opt_result['spin_rate']:.0f} RPM" if opt_result.get("spin_rate") else "—"
    tf = opt_result.get("tracked_features")
    o_pts = str(len(tf["prev_points"])) if tf else "0"
    o_conf = f"{opt_result['flow_confidence']:.2f}" if opt_result.get("flow_confidence") else "—"
    cv2.putText(bar, f"OPTICAL: {o_pts} pts | {o_spin} | conf {o_conf}",
                (mid + 10, 55), font, 0.55, (255, 200, 0), 1)
    cv2.putText(bar, "Corner features + Lucas-Kanade + RANSAC RPM",
                (mid + 10, 80), font, 0.4, (150, 150, 150), 1)

    return np.vstack([combined, bar])


def process_video(video_path, label, K, dist):
    """Process one video, return best frames and metrics."""
    print(f"\n  [{label}] Initializing pipelines...")
    seam_pipe = SeamPipeline(K, dist, confidence=CONFIDENCE, model_path=MODEL_PATH)
    opt_pipe = OpticalFlowPipeline(K, dist, confidence=CONFIDENCE, model_path=MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [{label}] {total_frames} frames @ {fps:.1f} fps")

    # Best-frame tracking
    best_seam = {"score": -1, "frame": None, "result": None, "idx": 0}
    best_optical = {"score": -1, "frame": None, "result": None, "idx": 0}
    best_comparison = {"score": -1, "seam_frame": None, "opt_frame": None,
                       "seam_result": None, "opt_result": None, "idx": 0}

    # Metrics
    seam_detections = seam_orientations = 0
    opt_detections = opt_orientations = 0
    seam_spins, opt_spins, opt_confs = [], [], []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        sr = seam_pipe.process_frame(frame.copy(), timestamp)
        or_ = opt_pipe.process_frame(frame.copy(), timestamp)

        # Accumulate metrics
        if sr["ball_detected"]:
            seam_detections += 1
        if sr["orientation"] is not None:
            seam_orientations += 1
        if sr.get("spin_rate") is not None:
            seam_spins.append(sr["spin_rate"])

        if or_["ball_detected"]:
            opt_detections += 1
        if or_["orientation"] is not None:
            opt_orientations += 1
        if or_.get("spin_rate") is not None:
            opt_spins.append(or_["spin_rate"])
        if or_.get("flow_confidence") is not None:
            opt_confs.append(or_["flow_confidence"])

        # Track best frames
        ss = score_seam_frame(sr)
        if ss > best_seam["score"]:
            best_seam = {"score": ss, "frame": frame.copy(),
                         "result": sr, "idx": frame_idx}

        os_ = score_optical_frame(or_)
        if os_ > best_optical["score"]:
            best_optical = {"score": os_, "frame": frame.copy(),
                            "result": or_, "idx": frame_idx}

        combined_score = ss + os_
        if combined_score > best_comparison["score"]:
            best_comparison = {
                "score": combined_score,
                "seam_frame": frame.copy(), "opt_frame": frame.copy(),
                "seam_result": sr, "opt_result": or_, "idx": frame_idx
            }

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"    Frame {frame_idx}/{total_frames}")

    cap.release()

    metrics = {
        "label": label,
        "total_frames": frame_idx,
        "fps": fps,
        "seam": {
            "detection_count": seam_detections,
            "detection_rate_pct": 100.0 * seam_detections / max(frame_idx, 1),
            "orientation_count": seam_orientations,
            "orientation_rate_pct": 100.0 * seam_orientations / max(frame_idx, 1),
            "avg_spin_rpm": float(np.mean(seam_spins)) if seam_spins else None,
            "median_spin_rpm": float(np.median(seam_spins)) if seam_spins else None,
            "spin_samples": len(seam_spins),
        },
        "optical": {
            "detection_count": opt_detections,
            "detection_rate_pct": 100.0 * opt_detections / max(frame_idx, 1),
            "orientation_count": opt_orientations,
            "orientation_rate_pct": 100.0 * opt_orientations / max(frame_idx, 1),
            "avg_spin_rpm": float(np.mean(opt_spins)) if opt_spins else None,
            "median_spin_rpm": float(np.median(opt_spins)) if opt_spins else None,
            "spin_samples": len(opt_spins),
            "avg_flow_confidence": float(np.mean(opt_confs)) if opt_confs else None,
        }
    }

    s_rpm = metrics['seam']['avg_spin_rpm']
    o_rpm = metrics['optical']['avg_spin_rpm']
    print(f"  [{label}] Seam: {seam_detections}/{frame_idx} detected, "
          f"avg spin {f'{s_rpm:.1f} RPM' if s_rpm is not None else 'N/A'}")
    print(f"  [{label}] Optical: {opt_detections}/{frame_idx} detected, "
          f"avg spin {f'{o_rpm:.1f} RPM' if o_rpm is not None else 'N/A'}")

    return {
        "best_seam": best_seam,
        "best_optical": best_optical,
        "best_comparison": best_comparison,
        "metrics": metrics,
        "total_frames": frame_idx,
        "fps": fps,
    }


def save_frames(data, label, output_dir):
    """Annotate and save best frames to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # Best seam frame
    bs = data["best_seam"]
    if bs["frame"] is not None:
        ann = annotate_seam_frame(bs["frame"], bs["result"], bs["idx"], label)
        path = output_dir / f"{label}_seam_best.jpg"
        # Scale down for README (keep ≤ 1200px wide)
        ann = _scale_down(ann, max_w=1200)
        cv2.imwrite(str(path), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["seam_best"] = str(path)
        print(f"  Saved: {path}")

    # Best optical frame
    bo = data["best_optical"]
    if bo["frame"] is not None:
        ann = annotate_optical_frame(bo["frame"], bo["result"], bo["idx"], label)
        ann = _scale_down(ann, max_w=1200)
        path = output_dir / f"{label}_optical_best.jpg"
        cv2.imwrite(str(path), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["optical_best"] = str(path)
        print(f"  Saved: {path}")

    # Best comparison frame (side-by-side)
    bc = data["best_comparison"]
    if bc["seam_frame"] is not None:
        comp = make_comparison_frame(
            bc["seam_frame"], bc["opt_frame"],
            bc["seam_result"], bc["opt_result"],
            label, bc["idx"], data["total_frames"]
        )
        path = output_dir / f"{label}_comparison.jpg"
        cv2.imwrite(str(path), comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["comparison"] = str(path)
        print(f"  Saved: {path}")

    return saved


def _scale_down(img, max_w=1200):
    """Scale image down if wider than max_w, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract best detection frames from both pipelines for documentation.")
    parser.add_argument(
        "--videos", nargs="*",
        default=[
            "spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4",
            "spin_dataset/raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4",
        ],
        help="Input video paths (default: both spin_dataset videos)",
    )
    parser.add_argument("--output", default="docs/frames",
                        help="Output directory for frames (default: docs/frames)")
    parser.add_argument("--camera", default=CAMERA_JSON,
                        help="Camera JSON path (default: config/camera.json)")
    args = parser.parse_args()

    out_dir = Path(args.output)

    print("=" * 60)
    print("Baseball Frame Extractor — Best Detection Frames")
    print("=" * 60)

    K, dist, _ = load_camera_params(args.camera)

    # Build (path, label) pairs; auto-label as video1, video2, …
    videos = [(p, f"video{i+1}") for i, p in enumerate(args.videos)]

    all_metrics = {}
    for vpath, label in videos:
        if not os.path.exists(vpath):
            print(f"  Skipping {label}: file not found ({vpath})")
            continue

        print(f"\nProcessing {label}: {vpath}")
        data = process_video(vpath, label, K, dist)
        if data is None:
            continue

        saved = save_frames(data, label, out_dir)
        all_metrics[label] = data["metrics"]
        all_metrics[label]["saved_frames"] = saved

    # Save metrics JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Video 1':>18} {'Video 2':>18}")
    print("-" * 70)
    for k in ["seam", "optical"]:
        for v in ["video1", "video2"]:
            if v not in all_metrics:
                all_metrics[v] = {}
    if all_metrics:
        v1 = all_metrics.get("video1", {})
        v2 = all_metrics.get("video2", {})
        rows = [
            ("Frames", "total_frames", None),
            ("Seam Detection %", "seam.detection_rate_pct", ".1f"),
            ("Seam Orientation %", "seam.orientation_rate_pct", ".1f"),
            ("Seam Avg Spin (RPM)", "seam.avg_spin_rpm", ".1f"),
            ("Optical Detection %", "optical.detection_rate_pct", ".1f"),
            ("Optical Orientation %", "optical.orientation_rate_pct", ".1f"),
            ("Optical Avg Spin (RPM)", "optical.avg_spin_rpm", ".1f"),
            ("Optical Avg Confidence", "optical.avg_flow_confidence", ".3f"),
        ]
        for name, key, fmt in rows:
            def _get(d, k):
                parts = k.split(".")
                val = d
                for p in parts:
                    if isinstance(val, dict):
                        val = val.get(p)
                    else:
                        return None
                return val

            v1v = _get(v1, key)
            v2v = _get(v2, key)
            f1 = f"{v1v:{fmt}}" if v1v is not None and fmt else (str(v1v) if v1v is not None else "N/A")
            f2 = f"{v2v:{fmt}}" if v2v is not None and fmt else (str(v2v) if v2v is not None else "N/A")
            print(f"  {name:<28} {f1:>18} {f2:>18}")

    print("\nDone. Frames in:", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    exit(main())
