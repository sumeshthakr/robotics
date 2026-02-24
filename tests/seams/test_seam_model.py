import pytest
import numpy as np
from src.seams.seam_model import BaseballSeamModel

def test_seam_model_shape():
    model = BaseballSeamModel(radius=1.0)
    points = model.get_3d_points(num_points_per_curve=50)

    # Should have 2 curves
    assert len(points) == 2

    # Each curve should have requested points
    assert points[0].shape[0] == 50
    assert points[1].shape[0] == 50

    # Points should be 3D
    assert points[0].shape[1] == 3

def test_seam_model_radius():
    model = BaseballSeamModel(radius=37.0)  # ~37mm baseball radius
    points = model.get_3d_points()

    # All points should be approximately at radius distance from origin
    for curve in points:
        distances = np.linalg.norm(curve, axis=1)
        assert np.allclose(distances, 37.0, atol=1.0)
