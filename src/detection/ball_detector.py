import numpy as np
from ultralytics import YOLO

class BallDetector:
    """Baseball detector using YOLOv8."""

    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """Initialize detector.

        Args:
            model_name: YOLO model name or path
            confidence_threshold: Minimum confidence for detection (0-1)

        Raises:
            ValueError: If confidence_threshold is not between 0 and 1
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {confidence_threshold}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # Sports ball class index in COCO is 32
        self.sport_ball_class = 32

    def detect(self, image: np.ndarray) -> dict:
        """Detect baseball in image.

        Args:
            image: Input image (H, W, 3)

        Returns:
            dict with keys:
                - detected: bool
                - bbox: (x1, y1, x2, y2) or None
                - confidence: float or None

        Raises:
            TypeError: If image is not a numpy array
            ValueError: If image does not have 3 dimensions
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
        if image.ndim != 3:
            raise ValueError(f"image must have 3 dimensions (H, W, C), got {image.ndim} dimensions")

        results = self.model(image, verbose=False)

        # Default result when no ball is detected
        no_detection_result = {"detected": False, "bbox": None, "confidence": None}

        if len(results) == 0 or len(results[0].boxes) == 0:
            return no_detection_result

        # Get the most confident detection for sports ball
        best_box = None
        best_conf = 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == self.sport_ball_class and conf > self.confidence_threshold:
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            return {
                "detected": True,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": best_conf
            }

        return no_detection_result
