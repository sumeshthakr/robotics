import numpy as np
import cv2

def detect_seams(image: np.ndarray,
                 canny_low=30,
                 canny_high=100,
                 use_color_filter=True,
                 adaptive=True,
                 color_boost=True) -> dict:
    """Detect baseball seams using edge detection and optional color filtering.

    Args:
        image: Input image (H, W, 3), ROI of ball only
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        use_color_filter: If True, filter for red seam color
        adaptive: If True, adjust parameters based on image size
        color_boost: If True, boost color saturation for pale seams

    Returns:
        dict with keys:
            - edges: binary edge map (H, W)
            - seam_pixels: Nx2 array of (x, y) seam pixel coordinates
            - num_pixels: count of seam pixels
    """
    height, width = image.shape[:2]

    # Adjust parameters for small images
    if adaptive:
        if height < 60 or width < 60:
            # Small ROI - use more sensitive thresholds
            canny_low = max(20, canny_low - 20)
            canny_high = max(50, canny_high - 50)
            # Very relaxed color thresholds for small regions
            saturation_low = 15
            value_low = 40
        else:
            saturation_low = 50
            value_low = 80
    else:
        saturation_low = 50
        value_low = 80

    # Optionally boost color saturation for pale seams
    if color_boost:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        # Boost saturation
        sat = cv2.multiply(sat, 1.5)
        sat = np.clip(sat, 0, 255).astype(np.uint8)
        hsv_boosted = cv2.merge([hue, sat, val])
        image = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur - use smaller kernel for small images
    kernel_size = min(5, max(3, (height + width) // 20))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Store raw edges in case we need them
    raw_edges = edges.copy()

    # Optional: color filter for red seams
    if use_color_filter:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range (wraps around in HSV) - very permissive for pale seams
        lower_red1 = np.array([0, saturation_low, value_low])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, saturation_low, value_low])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Combine edges with red mask
        edges_with_color = cv2.bitwise_and(raw_edges, red_mask)

        # If color filtering removes too many edges, fall back to raw edges
        if np.sum(edges_with_color) < np.sum(raw_edges) * 0.3:
            # Color filter is too aggressive - use raw edges
            edges = raw_edges
        else:
            edges = edges_with_color

    # Dilate edges slightly to connect nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

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
