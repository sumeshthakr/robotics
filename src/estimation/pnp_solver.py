import numpy as np
import cv2

def solve_orientation(points_2d, points_3d, camera_matrix,
                      method=cv2.SOLVEPNP_ITERATIVE) -> dict:
    """Solve for orientation using PnP with 3D seam model.

    Args:
        points_2d: Nx2 array of detected 2D seam pixel coordinates
        points_3d: Nx3 array of corresponding 3D seam model points
        camera_matrix: 3x3 camera intrinsics matrix
        method: PnP solving method

    Returns:
        dict with keys:
            - success: bool
            - rotation_matrix: 3x3 rotation matrix or None
            - translation: 3x1 translation vector or None
            - rvec: 3x1 rotation vector (Rodrigues) or None
            - tvec: 3x1 translation vector or None
    """
    if len(points_2d) < 4 or len(points_3d) < 4:
        return {
            "success": False,
            "rotation_matrix": None,
            "translation": None,
            "rvec": None,
            "tvec": None
        }

    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        distCoeffs=None,
        flags=method
    )

    if success:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        return {
            "success": True,
            "rotation_matrix": R,
            "translation": tvec,
            "rvec": rvec,
            "tvec": tvec
        }
    else:
        return {
            "success": False,
            "rotation_matrix": None,
            "translation": None,
            "rvec": None,
            "tvec": None
        }

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        [w, x, y, z] quaternion (scalar-first)
    """
    # Handle potential numerical issues
    R = np.array(R, dtype=np.float64)

    # Compute quaternion using trace method
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (ZYX convention).

    Args:
        R: 3x3 rotation matrix

    Returns:
        [roll, pitch, yaw] in radians
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])
