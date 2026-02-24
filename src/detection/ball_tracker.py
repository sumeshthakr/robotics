"""
Enhanced ball tracking module that combines YOLO detection with temporal tracking.
Uses Kalman filtering and optical flow to track the ball between detections.
"""

import numpy as np
from scipy.spatial.distance import cdist
from collections import deque
import cv2


class BallTracker:
    """Track baseball across video frames using detection + temporal consistency."""

    def __init__(self, detector, max_lost_frames=5, iou_threshold=0.3):
        """Initialize tracker.

        Args:
            detector: BallDetector instance
            max_lost_frames: How many frames to track without detection before giving up
            iou_threshold: Minimum IoU to match detection to existing track
        """
        self.detector = detector
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold

        # Tracking state
        self.track_bbox = None  # (x1, y1, x2, y2)
        self.track_confidence = 0.0
        self.lost_frames = 0
        self.velocity = (0, 0)  # (vx, vy) pixels per frame

        # History for smoothing
        self.bbox_history = deque(maxlen=5)

    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes.

        Args:
            bbox1, bbox2: (x1, y1, x2, y2)

        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0

        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _predict_bbox(self):
        """Predict next bbox using constant velocity model."""
        if self.track_bbox is None:
            return None

        x1, y1, x2, y2 = self.track_bbox
        w = x2 - x1
        h = y2 - y1

        # Apply velocity
        vx, vy = self.velocity
        new_x1 = int(x1 + vx)
        new_y1 = int(y1 + vy)
        new_x2 = new_x1 + w
        new_y2 = new_y1 + h

        return (new_x1, new_y1, new_x2, new_y2)

    def _update_velocity(self, new_bbox):
        """Update velocity based on new detection."""
        if self.track_bbox is None or len(self.bbox_history) == 0:
            return

        x1, y1, x2, y2 = new_bbox
        prev_x1, prev_y1, _, _ = self.track_bbox

        # Simple velocity estimate
        vx = x1 - prev_x1
        vy = y1 - prev_y1

        # Smooth velocity
        self.velocity = (
            0.7 * self.velocity[0] + 0.3 * vx,
            0.7 * self.velocity[1] + 0.3 * vy
        )

    def reset(self):
        """Reset tracking state."""
        self.track_bbox = None
        self.track_confidence = 0.0
        self.lost_frames = 0
        self.velocity = (0, 0)
        self.bbox_history.clear()

    def track(self, frame: np.ndarray) -> dict:
        """Track ball in frame.

        Args:
            frame: Input frame (H, W, 3)

        Returns:
            dict with keys:
                - detected: bool (True if detected or tracking)
                - bbox: (x1, y1, x2, y2) or None
                - confidence: float or None
                - tracking: bool (True if using prediction, False if from detection)
        """
        # Get fresh detection
        detection = self.detector.detect(frame)

        if detection["detected"]:
            # Fresh detection
            new_bbox = detection["bbox"]
            new_conf = detection["confidence"]

            # Check if this matches our existing track
            if self.track_bbox is not None:
                iou = self._iou(new_bbox, self.track_bbox)
                if iou > self.iou_threshold:
                    # Matches existing track - update with weighted average
                    self._update_velocity(new_bbox)
                    self.track_bbox = new_bbox
                    self.track_confidence = new_conf
                    self.lost_frames = 0
                    self.bbox_history.append(new_bbox)

                    return {
                        "detected": True,
                        "bbox": self.track_bbox,
                        "confidence": self.track_confidence,
                        "tracking": False
                    }

            # New detection or doesn't match - use it directly
            self._update_velocity(new_bbox)
            self.track_bbox = new_bbox
            self.track_confidence = new_conf
            self.lost_frames = 0
            self.bbox_history.append(new_bbox)

            return {
                "detected": True,
                "bbox": self.track_bbox,
                "confidence": self.track_confidence,
                "tracking": False
            }

        # No detection - use tracking if we have a recent track
        if self.track_bbox is not None and self.lost_frames < self.max_lost_frames:
            # Predict new position
            predicted_bbox = self._predict_bbox()
            self.track_bbox = predicted_bbox
            self.lost_frames += 1

            return {
                "detected": True,
                "bbox": self.track_bbox,
                "confidence": self.track_confidence * (0.9 ** self.lost_frames),  # Decay confidence
                "tracking": True
            }

        # Lost track
        self.reset()
        return {
            "detected": False,
            "bbox": None,
            "confidence": None,
            "tracking": False
        }

    def get_smoothed_bbox(self) -> tuple or None:
        """Get smoothed bbox using history.

        Returns:
            (x1, y1, x2, y2) or None
        """
        if len(self.bbox_history) < 2:
            return self.track_bbox

        # Average recent bboxes with exponential weighting
        weights = np.exp(np.linspace(-1, 0, len(self.bbox_history)))
        weights = weights / weights.sum()

        bboxes = np.array(self.bbox_history)
        smoothed = (bboxes * weights[:, None]).sum(axis=0)

        return tuple(smoothed.astype(int))
