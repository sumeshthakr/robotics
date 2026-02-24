import numpy as np
from scipy.optimize import least_squares

def fit_circle(points):
    """Fit circle to 2D points using least squares.

    Args:
        points: Nx2 array of (x, y) points

    Returns:
        dict with keys:
            - cx: center x coordinate
            - cy: center y coordinate
            - radius: circle radius
            - success: bool
    """
    if len(points) < 3:
        return {"cx": 0, "cy": 0, "radius": 0, "success": False}

    # Initial guess: centroid + average distance
    x_mean, y_mean = np.mean(points, axis=0)
    r_guess = np.mean(np.sqrt(np.sum((points - [x_mean, y_mean])**2, axis=1)))

    def residuals(params, points):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        return distances - r

    result = least_squares(
        residuals,
        x0=[x_mean, y_mean, r_guess],
        args=(points,),
        method='lm'
    )

    if result.success:
        cx, cy, r = result.x
        return {"cx": float(cx), "cy": float(cy), "radius": float(r), "success": True}
    else:
        return {"cx": 0, "cy": 0, "radius": 0, "success": False}
