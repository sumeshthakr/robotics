import pytest
import numpy as np
from src.seams.edge_detector import detect_seams

def test_detect_seams_shape():
    # Create synthetic image with red curves on white background
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Draw a red circle
    import cv2
    cv2.circle(img, (100, 100), 50, (0, 0, 200), 3)

    result = detect_seams(img)

    assert "edges" in result
    assert "seam_pixels" in result
    assert result["edges"].shape == img.shape[:2]
    assert len(result["seam_pixels"]) > 0 or result["seam_pixels"].shape[1] == 2
