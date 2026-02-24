import numpy as np
import cv2

def detect_seams(image: np.ndarray,
                 canny_low=50,
                 canny_high=150,
                 use_color_filter=True) -> dict:
    """Detect baseball seams using edge detection and optional color filtering.

    Args:
        image: Input image (H, W, 3), ROI of ball only
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        use_color_filter: If True, filter for red seam color

    Returns:
        dict with keys:
            - edges: binary edge map (H, W)
            - seam_pixels: Nx2 array of (x, y) seam pixel coordinates
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Optional: color filter for red seams
    if use_color_filter:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range (wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Combine edges with red mask
        edges = cv2.bitwise_and(edges, red_mask)

    # Find seam pixel coordinates
    seam_pixels = np.column_stack(np.where(edges > 0))

    # Swap to (x, y) format for OpenCV compatibility
    if len(seam_pixels) > 0:
        seam_pixels = seam_pixels[:, [1, 0]]

    return {
        "edges": edges,
        "seam_pixels": seam_pixels,
        "num_pixels": len(seam_pixels)
    }
