#!/usr/bin/env python3
"""Main entry point for baseball orientation detection system."""
import argparse
from pathlib import Path
from src.utils.camera import load_camera_params
from src.pipeline import BaseballOrientationPipeline
from src.pipeline_optical import OpticalFlowPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Detect baseball orientation from video"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results",
        help="Output directory for results (default: outputs/results)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate video with visualization overlays"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="config/camera.json",
        help="Path to camera parameters JSON (default: config/camera.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or name (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold 0-1 (default: 0.25)"
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["seam", "optical"],
        default="seam",
        help="Orientation detection approach: 'seam' (seam detection + PnP) or 'optical' (optical flow) (default: seam)"
    )
    parser.add_argument(
        "--max-corners",
        type=int,
        default=50,
        help="Maximum corners for optical flow tracking (only for --approach optical, default: 50)"
    )
    parser.add_argument(
        "--min-flow",
        type=float,
        default=0.5,
        help="Minimum flow magnitude threshold for optical flow (default: 0.5)"
    )
    parser.add_argument(
        "--max-flow",
        type=float,
        default=30.0,
        help="Maximum flow magnitude threshold for optical flow (default: 30.0)"
    )

    args = parser.parse_args()

    # Validate video file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    print(f"Loading camera parameters from {args.camera}")
    try:
        K, dist, img_shape = load_camera_params(args.camera)
        print(f"  Camera matrix: {K.shape}")
        print(f"  Image shape: {img_shape}")
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return 1

    # Initialize pipeline based on approach
    approach_name = "seam-based" if args.approach == "seam" else "optical flow"
    print(f"Initializing {approach_name} pipeline with {args.model}")

    if args.approach == "optical":
        pipeline = OpticalFlowPipeline(
            camera_matrix=K,
            dist_coeffs=dist,
            confidence_threshold=args.confidence,
            model_path=args.model,
            max_corners=args.max_corners,
            min_flow_threshold=args.min_flow,
            max_flow_threshold=args.max_flow
        )
    else:  # default to seam-based
        pipeline = BaseballOrientationPipeline(
            camera_matrix=K,
            dist_coeffs=dist,
            confidence_threshold=args.confidence,
            model_path=args.model
        )

    # Process video
    print(f"Processing video: {args.video_path}")

    output_video_path = None
    if args.visualize:
        output_video_path = str(output_dir / f"{video_path.stem}_output.mp4")
        print(f"  Writing visualization to: {output_video_path}")

    try:
        results = pipeline.process_video(
            str(video_path),
            output_path=output_video_path,
            visualize=args.visualize
        )

        # Print summary
        print("\n=== Processing Complete ===")
        print(f"Approach: {approach_name}")
        print(f"Total frames processed: {results['total_frames']}")
        print(f"Video FPS: {results['fps']:.2f}")

        # Count detections
        detections = sum(1 for r in results['detections'] if r['ball_detected'])
        print(f"Frames with ball detected: {detections} ({100*detections/results['total_frames']:.1f}%)")

        # Count orientations
        orientations = sum(1 for r in results['detections'] if r['orientation'] is not None)
        print(f"Frames with orientation: {orientations} ({100*orientations/results['total_frames']:.1f}%)")

        # Print confidence for optical flow approach
        if args.approach == "optical" and 'average_confidence' in results:
            if results['average_confidence'] is not None:
                print(f"Average flow confidence: {results['average_confidence']:.3f}")

        if results['average_spin_rate'] is not None:
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
