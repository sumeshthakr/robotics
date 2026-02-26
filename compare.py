#!/usr/bin/env python3
"""Generate side-by-side comparison videos of Seam vs Optical Flow approaches.

Creates a comparison video with:
    LEFT  half: Seam-based approach (red seam detection + PnP)
    RIGHT half: Optical flow approach (feature tracking + RANSAC)
    BOTTOM bar:  Live performance metrics (detection %, orientation %, spin rate)

Usage:
    python compare.py
    python compare.py --video1 spin_dataset/video1.mp4 --video2 spin_dataset/video2.mp4
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from camera import load_camera_params
from seam_pipeline import SeamPipeline
from optical_pipeline import OpticalFlowPipeline


def make_comparison_video(video_path, output_path, K, dist,
                          model_path="yolov8n.pt", confidence=0.25):
    """Process one video with BOTH approaches and create side-by-side comparison.

    Args:
        video_path:  Path to input video
        output_path: Path to save comparison video
        K:           Camera intrinsic matrix
        dist:        Distortion coefficients
        model_path:  YOLO model path
        confidence:  Detection confidence

    Returns:
        dict with performance stats for both approaches
    """
    # ---- Initialize both pipelines ----
    seam_pipe = SeamPipeline(K, dist, confidence=confidence, model_path=model_path)
    optical_pipe = OpticalFlowPipeline(K, dist, confidence=confidence, model_path=model_path)

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- Output dimensions ----
    # Each side is half-width, plus a stats bar at the bottom
    half_w = orig_w // 2
    side_h = orig_h // 2  # Scale down each side
    stats_h = 180          # Height of stats bar
    out_w = half_w * 2
    out_h = side_h + stats_h

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

    # ---- Tracking stats ----
    seam_stats = {"detections": 0, "orientations": 0, "spin_rates": [], "times": []}
    opt_stats = {"detections": 0, "orientations": 0, "spin_rates": [], "times": [],
                 "confidences": []}

    frame_idx = 0
    print(f"  Processing {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps if fps > 0 else None

        # ---- Run SEAM pipeline ----
        t0 = time.time()
        seam_result = seam_pipe.process_frame(frame.copy(), timestamp)
        seam_time = time.time() - t0

        # ---- Run OPTICAL FLOW pipeline ----
        t0 = time.time()
        opt_result = optical_pipe.process_frame(frame.copy(), timestamp)
        opt_time = time.time() - t0

        # ---- Update stats ----
        if seam_result["ball_detected"]:
            seam_stats["detections"] += 1
        if seam_result["orientation"] is not None:
            seam_stats["orientations"] += 1
        if seam_result["spin_rate"] is not None:
            seam_stats["spin_rates"].append(seam_result["spin_rate"])
        seam_stats["times"].append(seam_time)

        if opt_result["ball_detected"]:
            opt_stats["detections"] += 1
        if opt_result["orientation"] is not None:
            opt_stats["orientations"] += 1
        if opt_result["spin_rate"] is not None:
            opt_stats["spin_rates"].append(opt_result["spin_rate"])
        opt_stats["times"].append(opt_time)
        if opt_result.get("flow_confidence") is not None:
            opt_stats["confidences"].append(opt_result["flow_confidence"])

        # ---- Build visualization ----
        # Left side: seam approach
        left = _draw_seam_viz(frame.copy(), seam_result, frame_idx)
        left = cv2.resize(left, (half_w, side_h))

        # Right side: optical flow approach
        right = _draw_optical_viz(frame.copy(), opt_result, frame_idx)
        right = cv2.resize(right, (half_w, side_h))

        # Combine side by side
        combined = np.hstack([left, right])

        # Stats bar at the bottom
        stats_bar = _draw_stats_bar(
            out_w, stats_h, frame_idx + 1, total_frames,
            seam_result, opt_result, seam_stats, opt_stats,
            seam_time, opt_time)

        # Stack vertically
        output_frame = np.vstack([combined, stats_bar])
        writer.write(output_frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"    Frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    # ---- Compile summary stats ----
    summary = {
        "total_frames": frame_idx,
        "fps": fps,
        "seam": {
            "ball_detected": seam_stats["detections"],
            "detection_rate": seam_stats["detections"] / frame_idx * 100,
            "orientation_estimated": seam_stats["orientations"],
            "orientation_rate": seam_stats["orientations"] / frame_idx * 100,
            "avg_spin_rpm": float(np.mean(seam_stats["spin_rates"])) if seam_stats["spin_rates"] else None,
            "avg_time_ms": float(np.mean(seam_stats["times"]) * 1000),
        },
        "optical": {
            "ball_detected": opt_stats["detections"],
            "detection_rate": opt_stats["detections"] / frame_idx * 100,
            "orientation_estimated": opt_stats["orientations"],
            "orientation_rate": opt_stats["orientations"] / frame_idx * 100,
            "avg_spin_rpm": float(np.mean(opt_stats["spin_rates"])) if opt_stats["spin_rates"] else None,
            "avg_confidence": float(np.mean(opt_stats["confidences"])) if opt_stats["confidences"] else None,
            "avg_time_ms": float(np.mean(opt_stats["times"]) * 1000),
        }
    }
    return summary


# ============================================================
# Drawing helpers
# ============================================================

def _draw_seam_viz(frame, result, frame_idx):
    """Draw seam pipeline results on frame."""
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = vis.shape[:2]

    # Title bar
    cv2.rectangle(vis, (0, 0), (w, 35), (40, 40, 40), -1)
    cv2.putText(vis, "SEAM-BASED APPROACH", (10, 25), font, 0.7, (0, 200, 255), 2)

    if not result["ball_detected"]:
        cv2.putText(vis, "No ball detected", (10, 60), font, 0.6, (0, 0, 255), 2)
        return vis

    x1, y1, x2, y2 = result["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Bounding box
    color = (0, 255, 0) if not result.get("tracking") else (0, 255, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Seam pixels (red dots)
    if result.get("seam_pixels") is not None and len(result["seam_pixels"]) > 0:
        for px, py in result["seam_pixels"]:
            gx, gy = int(px + x1), int(py + y1)
            if 0 <= gx < w and 0 <= gy < h:
                cv2.circle(vis, (gx, gy), 1, (0, 0, 255), -1)

    y_text = 55
    if result["orientation"]:
        e = np.degrees(result["orientation"]["euler_angles"])
        cv2.putText(vis, f"Euler: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]",
                    (10, y_text), font, 0.5, (255, 200, 0), 1)
        y_text += 22

    if result["spin_rate"] is not None:
        cv2.putText(vis, f"Spin: {result['spin_rate']:.0f} RPM",
                    (10, y_text), font, 0.6, (0, 255, 255), 2)
        y_text += 22

    # Spin axis arrow
    if result["spin_axis"] is not None:
        axis = result["spin_axis"]
        cv2.arrowedLine(vis, (cx, cy),
                        (int(cx + axis[0]*60), int(cy + axis[1]*60)),
                        (255, 0, 255), 3, tipLength=0.3)

    return vis


def _draw_optical_viz(frame, result, frame_idx):
    """Draw optical flow pipeline results on frame."""
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = vis.shape[:2]

    # Title bar
    cv2.rectangle(vis, (0, 0), (w, 35), (40, 40, 40), -1)
    cv2.putText(vis, "OPTICAL FLOW APPROACH", (10, 25), font, 0.7, (255, 200, 0), 2)

    if not result["ball_detected"]:
        cv2.putText(vis, "No ball detected", (10, 60), font, 0.6, (0, 0, 255), 2)
        return vis

    x1, y1, x2, y2 = result["bbox"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Bounding box
    color = (255, 255, 0) if not result.get("tracking") else (0, 255, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Flow vectors
    if result.get("tracked_features"):
        tracked = result["tracked_features"]
        for p1, p2 in zip(tracked["prev_points"], tracked["curr_points"]):
            pt1 = (int(p1[0] + x1), int(p1[1] + y1))
            pt2 = (int(p2[0] + x1), int(p2[1] + y1))
            cv2.arrowedLine(vis, pt1, pt2, (0, 255, 255), 1, tipLength=0.3)

    y_text = 55
    if result["orientation"]:
        e = np.degrees(result["orientation"]["euler_angles"])
        cv2.putText(vis, f"Euler: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]",
                    (10, y_text), font, 0.5, (255, 200, 0), 1)
        y_text += 22

    if result["spin_rate"] is not None:
        cv2.putText(vis, f"Spin: {result['spin_rate']:.0f} RPM",
                    (10, y_text), font, 0.6, (0, 255, 255), 2)
        y_text += 22

    if result.get("flow_confidence") is not None:
        cv2.putText(vis, f"Confidence: {result['flow_confidence']:.2f}",
                    (10, y_text), font, 0.5, (0, 200, 255), 1)

    # Spin axis arrow
    if result["spin_axis"] is not None:
        axis = result["spin_axis"]
        cv2.arrowedLine(vis, (cx, cy),
                        (int(cx + axis[0]*60), int(cy + axis[1]*60)),
                        (255, 0, 255), 3, tipLength=0.3)

    return vis


def _draw_stats_bar(width, height, frame_num, total_frames,
                    seam_result, opt_result, seam_stats, opt_stats,
                    seam_time, opt_time):
    """Draw the performance comparison stats bar."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)  # Dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    mid = width // 2

    # Divider line
    cv2.line(bar, (mid, 0), (mid, height), (80, 80, 80), 2)

    # ---- SEAM SIDE (left) ----
    y = 25
    total = frame_num
    det_pct = seam_stats["detections"] / total * 100
    ori_pct = seam_stats["orientations"] / total * 100
    spin = np.mean(seam_stats["spin_rates"]) if seam_stats["spin_rates"] else 0

    cv2.putText(bar, "SEAM STATS", (10, y), font, 0.6, (0, 200, 255), 2)
    y += 28
    cv2.putText(bar, f"Detection:   {seam_stats['detections']}/{total} ({det_pct:.0f}%)",
                (10, y), font, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(bar, f"Orientation: {seam_stats['orientations']}/{total} ({ori_pct:.0f}%)",
                (10, y), font, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(bar, f"Avg Spin:    {spin:.0f} RPM",
                (10, y), font, 0.5, (0, 255, 255), 1)
    y += 22
    cv2.putText(bar, f"Frame time:  {seam_time*1000:.0f} ms",
                (10, y), font, 0.5, (150, 150, 150), 1)
    y += 22

    # Current frame status
    s_status = "ORIENTATION" if seam_result["orientation"] else (
        "DETECTED" if seam_result["ball_detected"] else "NO BALL")
    s_color = (0, 255, 0) if seam_result["orientation"] else (
        (0, 255, 255) if seam_result["ball_detected"] else (0, 0, 255))
    cv2.putText(bar, f"Status: {s_status}", (10, y), font, 0.5, s_color, 1)

    # ---- OPTICAL FLOW SIDE (right) ----
    y = 25
    det_pct = opt_stats["detections"] / total * 100
    ori_pct = opt_stats["orientations"] / total * 100
    spin = np.mean(opt_stats["spin_rates"]) if opt_stats["spin_rates"] else 0
    conf = np.mean(opt_stats["confidences"]) if opt_stats["confidences"] else 0

    cv2.putText(bar, "OPTICAL FLOW STATS", (mid + 10, y), font, 0.6, (255, 200, 0), 2)
    y += 28
    cv2.putText(bar, f"Detection:   {opt_stats['detections']}/{total} ({det_pct:.0f}%)",
                (mid + 10, y), font, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(bar, f"Orientation: {opt_stats['orientations']}/{total} ({ori_pct:.0f}%)",
                (mid + 10, y), font, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(bar, f"Avg Spin:    {spin:.0f} RPM",
                (mid + 10, y), font, 0.5, (0, 255, 255), 1)
    y += 22
    cv2.putText(bar, f"Avg Conf:    {conf:.2f}  |  Frame: {opt_time*1000:.0f} ms",
                (mid + 10, y), font, 0.5, (150, 150, 150), 1)
    y += 22

    o_status = "ORIENTATION" if opt_result["orientation"] else (
        "DETECTED" if opt_result["ball_detected"] else "NO BALL")
    o_color = (0, 255, 0) if opt_result["orientation"] else (
        (0, 255, 255) if opt_result["ball_detected"] else (0, 0, 255))
    cv2.putText(bar, f"Status: {o_status}", (mid + 10, y), font, 0.5, o_color, 1)

    # Frame counter
    cv2.putText(bar, f"Frame {frame_num}/{total_frames}",
                (width - 200, height - 10), font, 0.5, (120, 120, 120), 1)

    return bar


def main():
    parser = argparse.ArgumentParser(description="Compare Seam vs Optical Flow approaches")
    parser.add_argument("--video1",
                        default="spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4",
                        help="First video path")
    parser.add_argument("--video2",
                        default="spin_dataset/raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4",
                        help="Second video path")
    parser.add_argument("--camera", default="config/camera.json")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    # Load camera
    K, dist, _ = load_camera_params(args.camera)
    os.makedirs(args.output, exist_ok=True)

    all_summaries = {}
    videos = []
    if os.path.exists(args.video1):
        videos.append(("video1", args.video1))
    if os.path.exists(args.video2):
        videos.append(("video2", args.video2))

    if not videos:
        print("No input videos found!")
        return 1

    for label, vpath in videos:
        print(f"\n{'='*60}")
        print(f"Comparing approaches on: {label}")
        print(f"  Input: {vpath}")
        out_path = os.path.join(args.output, f"comparison_{label}.mp4")
        print(f"  Output: {out_path}")

        summary = make_comparison_video(
            vpath, out_path, K, dist,
            model_path=args.model, confidence=args.confidence)

        all_summaries[label] = summary

        # Print comparison table
        s = summary["seam"]
        o = summary["optical"]
        print(f"\n  {'Metric':<25} {'Seam':>12} {'Optical Flow':>14}")
        print(f"  {'-'*51}")
        print(f"  {'Frames':<25} {summary['total_frames']:>12}")
        print(f"  {'Ball Detection %':<25} {s['detection_rate']:>11.1f}% {o['detection_rate']:>13.1f}%")
        print(f"  {'Orientation %':<25} {s['orientation_rate']:>11.1f}% {o['orientation_rate']:>13.1f}%")
        sr_s = f"{s['avg_spin_rpm']:.0f}" if s['avg_spin_rpm'] else "N/A"
        sr_o = f"{o['avg_spin_rpm']:.0f}" if o['avg_spin_rpm'] else "N/A"
        print(f"  {'Avg Spin (RPM)':<25} {sr_s:>12} {sr_o:>14}")
        print(f"  {'Avg Time/Frame (ms)':<25} {s['avg_time_ms']:>11.1f} {o['avg_time_ms']:>13.1f}")
        if o['avg_confidence']:
            print(f"  {'Avg Flow Confidence':<25} {'N/A':>12} {o['avg_confidence']:>13.2f}")

    # Save JSON summary
    json_path = os.path.join(args.output, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return 0


if __name__ == "__main__":
    exit(main())
