import pytest
import numpy as np
from src.estimation.sphere_fitter import fit_circle

def test_fit_circle_perfect():
    # Generate perfect circle points
    angles = np.linspace(0, 2*np.pi, 50)
    cx, cy, r = 100, 100, 50
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    points = np.column_stack([x, y])

    result = fit_circle(points)

    assert result["cx"] == pytest.approx(cx, abs=1)
    assert result["cy"] == pytest.approx(cy, abs=1)
    assert result["radius"] == pytest.approx(r, abs=1)
    assert result["success"] is True

def test_fit_circle_noisy():
    # Generate noisy circle points
    angles = np.linspace(0, 2*np.pi, 30)
    cx, cy, r = 150, 200, 75
    np.random.seed(42)
    x = cx + r * np.cos(angles) + np.random.randn(len(angles)) * 2
    y = cy + r * np.sin(angles) + np.random.randn(len(angles)) * 2
    points = np.column_stack([x, y])

    result = fit_circle(points)

    assert result["cx"] == pytest.approx(cx, abs=5)
    assert result["cy"] == pytest.approx(cy, abs=5)
    assert result["radius"] == pytest.approx(r, abs=5)
