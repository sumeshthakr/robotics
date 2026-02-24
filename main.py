# main.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Baseball Orientation Detection")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()

    print(f"Processing video: {args.video_path}")
    # Pipeline will be added here

if __name__ == "__main__":
    main()
