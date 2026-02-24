import pytest
import numpy as np
from src.estimation.pnp_solver import solve_orientation
from src.seams.seam_model import BaseballSeamModel

def test_solve_orientation_identity():
    # Generate known 3D points and project with identity rotation
    model = BaseballSeamModel(radius=37.0)
    points_3d = model.get_all_points()

    # Simple camera: identity rotation, ball at center
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float64)

    # Project 3D points (simplified - no translation)
    rvec = np.array([0, 0, 0], dtype=np.float64)
    tvec = np.array([0, 0, 500], dtype=np.float64)  # 500mm in front of camera

    import cv2
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    points_2d = points_2d.reshape(-1, 2)

    result = solve_orientation(points_2d, points_3d, K)

    assert result["success"] is True
    assert "rotation_matrix" in result
    assert "translation" in result
    assert result["rotation_matrix"].shape == (3, 3)
