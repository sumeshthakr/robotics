#!/usr/bin/env python3
"""Generate 3D ball trajectory and orientation visualizations.

Creates 3 figures per video (6 total):
  1. Detected 3D ball path (from bounding box + camera geometry)
  2. Seam approach: 3D path with orientation arrows
  3. Optical flow approach: 3D path with orientation arrows

Ball 3D position is recovered from the bounding box using the pinhole model:
    Z = fx * D_real / D_pixel
    X = (cx_img - cx0) * Z / fx
    Y = (cy_img - cy0) * Z / fy
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from camera import load_camera_params
from seam_pipeline import SeamPipeline
from optical_pipeline import OpticalFlowPipeline

BALL_DIAMETER_MM = 74.0
BALL_RADIUS_MM = 37.0
OUTPUT_DIR = "outputs/3d_plots"


def bbox_to_3d(bbox, K):
    """Convert bounding box to 3D position using pinhole camera model.

    Args:
        bbox: (x1, y1, x2, y2)
        K:    3x3 camera intrinsic matrix

    Returns:
        (X, Y, Z) in mm, camera frame
    """
    x1, y1, x2, y2 = bbox
    cx_img = (x1 + x2) / 2.0
    cy_img = (y1 + y2) / 2.0
    d_px = ((x2 - x1) + (y2 - y1)) / 2.0

    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]

    Z = fx * BALL_DIAMETER_MM / d_px
    X = (cx_img - cx0) * Z / fx
    Y = (cy_img - cy0) * Z / fy

    return X, Y, Z


def process_video(video_path, K, dist):
    """Run both pipelines and collect 3D positions + orientations."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    seam_pipe = SeamPipeline(K, dist, confidence=0.25)
    opt_pipe = OpticalFlowPipeline(K, dist, confidence=0.25)

    # Storage: lists of dicts per frame
    data = {
        "detected_path": [],   # Raw 3D positions from bbox
        "seam": [],            # Seam approach: position + orientation
        "optical": [],         # Optical approach: position + orientation
        "fps": fps,
        "total_frames": total,
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps

        sr = seam_pipe.process_frame(frame.copy(), ts)
        opr = opt_pipe.process_frame(frame.copy(), ts)

        # Detected 3D position (from whichever detected - they share YOLO)
        if sr["ball_detected"] and sr["bbox"]:
            X, Y, Z = bbox_to_3d(sr["bbox"], K)
            data["detected_path"].append({
                "frame": frame_idx, "t": ts,
                "X": X, "Y": Y, "Z": Z
            })

        # Seam approach
        if sr["ball_detected"] and sr["bbox"]:
            X, Y, Z = bbox_to_3d(sr["bbox"], K)
            entry = {"frame": frame_idx, "t": ts, "X": X, "Y": Y, "Z": Z}
            if sr["orientation"] is not None:
                entry["R"] = sr["orientation"]["rotation_matrix"]
            if sr["spin_axis"] is not None:
                entry["spin_axis"] = sr["spin_axis"]
            if sr["spin_rate"] is not None:
                entry["spin_rate"] = sr["spin_rate"]
            data["seam"].append(entry)

        # Optical flow approach
        if opr["ball_detected"] and opr["bbox"]:
            X, Y, Z = bbox_to_3d(opr["bbox"], K)
            entry = {"frame": frame_idx, "t": ts, "X": X, "Y": Y, "Z": Z}
            if opr["orientation"] is not None:
                entry["R"] = opr["orientation"]["rotation_matrix"]
            if opr["spin_axis"] is not None:
                entry["spin_axis"] = opr["spin_axis"]
            if opr["spin_rate"] is not None:
                entry["spin_rate"] = opr["spin_rate"]
            data["optical"].append(entry)

        frame_idx += 1

    cap.release()
    return data


def plot_detected_path(data, video_label, output_path):
    """Plot 1: Raw detected 3D ball trajectory."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    pts = data["detected_path"]
    if not pts:
        plt.close(fig)
        return

    X = [p["X"] for p in pts]
    Y = [p["Y"] for p in pts]
    Z = [p["Z"] for p in pts]
    T = [p["t"] for p in pts]

    # Color by time
    sc = ax.scatter(X, Z, Y, c=T, cmap='plasma', s=40, edgecolors='k',
                    linewidth=0.3, alpha=0.9)
    ax.plot(X, Z, Y, color='gray', alpha=0.4, linewidth=1)

    # Mark start and end
    ax.scatter([X[0]], [Z[0]], [Y[0]], color='lime', s=120, marker='^',
               edgecolors='black', linewidth=1.5, zorder=5, label='Start')
    ax.scatter([X[-1]], [Z[-1]], [Y[-1]], color='red', s=120, marker='v',
               edgecolors='black', linewidth=1.5, zorder=5, label='End')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Time (s)', fontsize=11)

    ax.set_xlabel('X (mm) — Lateral', fontsize=11)
    ax.set_ylabel('Z (mm) — Depth', fontsize=11)
    ax.set_zlabel('Y (mm) — Vertical', fontsize=11)
    ax.set_title(f'Detected Ball 3D Trajectory — {video_label}\n'
                 f'({len(pts)} frames, {data["fps"]:.0f} fps)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    # Better viewing angle
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_orientation_path(data, approach_key, approach_label, video_label,
                          output_path, color_main, color_arrow):
    """Plot 2/3: 3D path with orientation arrows from one approach."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    pts = data[approach_key]
    if not pts:
        plt.close(fig)
        return

    X = [p["X"] for p in pts]
    Y = [p["Y"] for p in pts]
    Z = [p["Z"] for p in pts]

    # Trajectory line
    ax.plot(X, Z, Y, color='gray', alpha=0.3, linewidth=1)

    # Separate: frames with/without orientation
    has_ori = [p for p in pts if "spin_axis" in p and p.get("spin_rate")]
    no_ori = [p for p in pts if "spin_axis" not in p or not p.get("spin_rate")]

    # Points without orientation (dimmer)
    if no_ori:
        ax.scatter([p["X"] for p in no_ori], [p["Z"] for p in no_ori],
                   [p["Y"] for p in no_ori], color='lightgray', s=15,
                   alpha=0.5, label='No orientation')

    # Points with orientation: colored by spin rate
    if has_ori:
        rpms = [p["spin_rate"] for p in has_ori]
        sc = ax.scatter([p["X"] for p in has_ori],
                        [p["Z"] for p in has_ori],
                        [p["Y"] for p in has_ori],
                        c=rpms, cmap='coolwarm', s=35, edgecolors='k',
                        linewidth=0.3, alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Spin Rate (RPM)', fontsize=11)

        # Draw spin axis arrows (subsample to avoid clutter)
        step = max(1, len(has_ori) // 15)
        # Scale arrows relative to the trajectory extent
        extent = max(
            max(X) - min(X),
            max(Y) - min(Y),
            max(Z) - min(Z),
            1.0
        )
        arrow_len = extent * 0.08

        for i in range(0, len(has_ori), step):
            p = has_ori[i]
            axis = p["spin_axis"]
            # Arrow from ball position in direction of spin axis
            ax.quiver(p["X"], p["Z"], p["Y"],
                      axis[0] * arrow_len,
                      axis[2] * arrow_len,
                      axis[1] * arrow_len,
                      color=color_arrow, alpha=0.7,
                      arrow_length_ratio=0.25, linewidth=1.5)

    # Start/end markers
    ax.scatter([X[0]], [Z[0]], [Y[0]], color='lime', s=120, marker='^',
               edgecolors='black', linewidth=1.5, zorder=5, label='Start')
    ax.scatter([X[-1]], [Z[-1]], [Y[-1]], color='red', s=120, marker='v',
               edgecolors='black', linewidth=1.5, zorder=5, label='End')

    ax.set_xlabel('X (mm) — Lateral', fontsize=11)
    ax.set_ylabel('Z (mm) — Depth', fontsize=11)
    ax.set_zlabel('Y (mm) — Vertical', fontsize=11)

    ori_count = len(has_ori)
    avg_rpm = np.mean(rpms) if has_ori else 0
    ax.set_title(
        f'{approach_label} — {video_label}\n'
        f'Orientation in {ori_count}/{len(pts)} frames, '
        f'Avg spin: {avg_rpm:.0f} RPM\n'
        f'(arrows = spin axis direction)',
        fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    K, dist, _ = load_camera_params("config/camera.json")

    videos = [
        ("Video 1", "spin_dataset/raw_spin_video_695d23c184c2b7ababb57a8e_1767711685.mp4"),
        ("Video 2", "spin_dataset/raw_spin_video_695d9b0a4899846853793e7d_1767742221.mp4"),
    ]

    for label, vpath in videos:
        if not os.path.exists(vpath):
            print(f"Skipping {label}: {vpath} not found")
            continue

        tag = label.lower().replace(" ", "")
        print(f"\nProcessing {label}: {vpath}")
        data = process_video(vpath, K, dist)

        print(f"  Detected: {len(data['detected_path'])} frames")
        print(f"  Seam: {len(data['seam'])} frames")
        print(f"  Optical: {len(data['optical'])} frames")

        # Plot 1: Raw detected path
        plot_detected_path(
            data, label,
            os.path.join(OUTPUT_DIR, f"{tag}_detected_path.png"))

        # Plot 2: Seam approach path + orientation
        plot_orientation_path(
            data, "seam", "Seam-Based Approach", label,
            os.path.join(OUTPUT_DIR, f"{tag}_seam_orientation.png"),
            color_main='#E74C3C', color_arrow='#C0392B')

        # Plot 3: Optical flow approach path + orientation
        plot_orientation_path(
            data, "optical", "Optical Flow Approach", label,
            os.path.join(OUTPUT_DIR, f"{tag}_optical_orientation.png"),
            color_main='#3498DB', color_arrow='#2980B9')

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
