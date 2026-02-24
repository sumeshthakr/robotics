#!/usr/bin/env python3
"""Demo script for optical flow rotation estimation.

This script creates a synthetic rotating baseball sequence and demonstrates
the optical flow approach for rotation estimation.
"""

import numpy as np
import cv2
from src.optical_flow.rotation_estimator import RotationEstimator


def create_synthetic_rotating_ball(frames=10, width=640, height=480):
    """Create a synthetic video of a rotating baseball.

    Args:
        frames: Number of frames to generate
        width: Image width
        height: Image height

    Returns:
        List of (gray_frame, bbox, color_frame) tuples
    """
    sequence = []
    center = (width // 2, height // 2)
    radius = 80

    for i in range(frames):
        # Create frame with gradient background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 180

        # Draw ball (darker circle)
        cv2.circle(frame, center, radius, (100, 100, 100), -1)
        cv2.circle(frame, center, radius, (50, 50, 50), 2)  # Edge

        # Draw rotating seam pattern - create high contrast features
        rotation_offset = i * 15  # degrees per frame

        # Draw multiple arcs that create corners for tracking
        for ring in [0.3, 0.5, 0.7, 0.9]:
            ring_radius = int(radius * ring)
            for angle_offset in [0, 90, 180, 270]:
                # Draw arc segment
                start_angle = rotation_offset + angle_offset
                for angle in range(start_angle, start_angle + 45, 5):
                    rad = np.radians(angle)
                    x = int(center[0] + ring_radius * np.cos(rad))
                    y = int(center[1] + ring_radius * np.sin(rad))
                    # High contrast dots
                    cv2.circle(frame, (x, y), 3, (40, 40, 40), -1)

        # Add radial lines that rotate
        for j in range(12):
            angle = np.radians(rotation_offset + j * 30)
            x1 = int(center[0] + 0.2 * radius * np.cos(angle))
            y1 = int(center[1] + 0.2 * radius * np.sin(angle))
            x2 = int(center[0] + 0.9 * radius * np.cos(angle))
            y2 = int(center[1] + 0.9 * radius * np.sin(angle))
            cv2.line(frame, (x1, y1), (x2, y2), (60, 60, 60), 2)

        # Add corner-like features at specific positions
        for j in range(6):
            angle = np.radians(rotation_offset + j * 60)
            x = int(center[0] + 0.7 * radius * np.cos(angle))
            y = int(center[1] + 0.7 * radius * np.sin(angle))
            # Draw a small corner pattern (L-shape)
            cv2.line(frame, (x-5, y), (x+5, y), (40, 40, 40), 2)
            cv2.line(frame, (x, y-5), (x, y+5), (40, 40, 40), 2)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bounding box
        bbox = (center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius)

        sequence.append((gray, bbox, frame))

    return sequence


def main():
    """Run the optical flow demo."""
    print("=" * 60)
    print("Optical Flow Rotation Estimation Demo")
    print("=" * 60)

    # Camera parameters
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Create rotation estimator
    estimator = RotationEstimator(
        camera_matrix=camera_matrix,
        ball_radius_mm=37.0,
        max_corners=50,
        min_flow_threshold=0.5,
        max_flow_threshold=20.0
    )

    # Generate synthetic sequence
    print("\nGenerating synthetic rotating ball sequence...")
    sequence = create_synthetic_rotating_ball(frames=20)

    print(f"Generated {len(sequence)} frames")

    # Process sequence
    print("\nProcessing frames with optical flow...")

    results = []
    for i, (gray, bbox, color_frame) in enumerate(sequence):
        result = estimator.estimate_rotation(gray, bbox, timestamp=i * 0.033)

        if result is not None:
            print(f"  Frame {i+1}: "
                  f"spin_rate={result.get('spin_rate_rps', 0)*60:.1f} RPM, "
                  f"axis=[{result.get('spin_axis', [0,0,0])[0]:.2f}, "
                  f"{result.get('spin_axis', [0,0,0])[1]:.2f}, "
                  f"{result.get('spin_axis', [0,0,0])[2]:.2f}], "
                  f"confidence={result.get('confidence', 0):.3f}")
            results.append(result)
        else:
            print(f"  Frame {i+1}: No rotation estimate (initializing/tracking)")

    # Get smoothed result
    print("\nFinal smoothed rotation estimate:")
    smoothed = estimator.get_smoothed_rotation()
    if smoothed:
        print(f"  Spin rate: {smoothed.get('spin_rate_rpm', 0):.1f} RPM")
        print(f"  Spin axis: [{smoothed.get('spin_axis', [0,0,0])[0]:.3f}, "
              f"{smoothed.get('spin_axis', [0,0,0])[1]:.3f}, "
              f"{smoothed.get('spin_axis', [0,0,0])[2]:.3f}]")
        print(f"  Confidence: {smoothed.get('confidence', 0):.3f}")

    print(f"\nSuccessfully processed {len(results)} frames with rotation estimates")

    # Save visualization
    print("\nGenerating visualization...")
    vis_frames = []

    # Reset estimator for visualization run
    estimator.reset()

    for i, (gray, bbox, color_frame) in enumerate(sequence):
        result = estimator.estimate_rotation(gray, bbox, timestamp=i * 0.033)

        vis = color_frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text
        text = f"Frame: {i+1}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if result:
            spin_rpm = result.get('spin_rate_rps', 0) * 60
            text = f"Spin: {spin_rpm:.0f} RPM"
            cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            conf = result.get('confidence', 0)
            text = f"Conf: {conf:.2f}"
            cv2.putText(vis, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        vis_frames.append(vis)

    # Save as video
    output_path = "outputs/optical_flow_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))

    for vis in vis_frames:
        writer.write(vis)

    writer.release()
    print(f"Visualization saved to: {output_path}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
