#!/usr/bin/env python3
"""Baseball Orientation Detection — Main Entry Point

Detects baseball orientation from high-speed video using either:
  - Seam-based approach:    detect red seams → 3D model matching → PnP solve
  - Optical flow approach:  track surface features → estimate rotation

Usage:
    python main.py video.mp4 --visualize
    python main.py video.mp4 --approach optical --visualize
"""
import argparse
from pathlib import Path
from camera import load_camera_params
from seam_pipeline import SeamPipeline
from optical_pipeline import OpticalFlowPipeline


def main():
    parser = argparse.ArgumentParser(description="Baseball Orientation Detection")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", default="outputs/results",
                        help="Output directory (default: outputs/results)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated output video")
    parser.add_argument("--camera", default="config/camera.json",
                        help="Camera parameters JSON (default: config/camera.json)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Detection confidence threshold 0-1 (default: 0.25)")
    parser.add_argument("--approach", choices=["seam", "optical"], default="seam",
                        help="Approach: seam (default) or optical flow")
    # Optical flow specific options
    parser.add_argument("--max-corners", type=int, default=50,
                        help="Max corners for optical flow (default: 50)")
    parser.add_argument("--min-flow", type=float, default=0.5,
                        help="Min flow magnitude in pixels (default: 0.5)")
    parser.add_argument("--max-flow", type=float, default=30.0,
                        help="Max flow magnitude in pixels (default: 30.0)")

    args = parser.parse_args()

    # Validate input video exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {args.video_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    print(f"Loading camera from {args.camera}")
    try:
        K, dist, img_shape = load_camera_params(args.camera)
        print(f"  Camera matrix: {K.shape}, Image shape: {img_shape}")
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return 1

    # Create pipeline
    approach_name = "seam-based" if args.approach == "seam" else "optical flow"
    print(f"Using {approach_name} approach with {args.model}")

    if args.approach == "optical":
        pipeline = OpticalFlowPipeline(
            K, dist, confidence=args.confidence, model_path=args.model,
            max_corners=args.max_corners, min_flow=args.min_flow,
            max_flow=args.max_flow
        )
    else:
        pipeline = SeamPipeline(
            K, dist, confidence=args.confidence, model_path=args.model
        )

    # Process video
    output_video_path = None
    if args.visualize:
        output_video_path = str(output_dir / f"{video_path.stem}_output.mp4")
        print(f"  Output video: {output_video_path}")

    print(f"Processing: {args.video_path}")

    try:
        results = pipeline.process_video(
            str(video_path), output_video_path, args.visualize
        )

        # Print summary
        total = results["total_frames"]
        detections = sum(1 for r in results["detections"] if r["ball_detected"])
        orientations = sum(1 for r in results["detections"]
                          if r["orientation"] is not None)

        print(f"\n=== Results ===")
        print(f"Approach: {approach_name}")
        print(f"Frames: {total} ({results['fps']:.1f} FPS)")
        print(f"Ball detected: {detections}/{total} "
              f"({100*detections/total:.1f}%)")
        print(f"Orientation estimated: {orientations}/{total} "
              f"({100*orientations/total:.1f}%)")

        if args.approach == "optical" and results.get("average_confidence"):
            print(f"Average flow confidence: "
                  f"{results['average_confidence']:.3f}")

        if results["average_spin_rate"] is not None:
            print(f"Average spin rate: {results['average_spin_rate']:.1f} RPM")

        if args.visualize:
            print(f"\nVisualization saved to: {output_video_path}")

        return 0

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
