import pytest
import numpy as np
from src.tracking.orientation_tracker import OrientationTracker

def test_tracker_init():
    tracker = OrientationTracker()
    assert len(tracker.history) == 0

def test_tracker_add_orientation():
    tracker = OrientationTracker()

    R1 = np.eye(3)
    tracker.add_orientation(R1, timestamp=0.0)

    assert len(tracker.history) == 1
    assert tracker.history[0]["timestamp"] == 0.0

def test_compute_spin_rate():
    tracker = OrientationTracker()

    # Add two orientations with known rotation
    from scipy.spatial.transform import Rotation

    R1 = Rotation.from_euler('z', 0).as_matrix()
    R2 = Rotation.from_euler('z', np.pi/2).as_matrix()  # 90 degree rotation

    tracker.add_orientation(R1, timestamp=0.0)
    tracker.add_orientation(R2, timestamp=0.1)  # 100ms

    spin_rate = tracker.get_spin_rate()

    # 90 degrees in 0.1 seconds = 900 deg/s = 150 RPM
    expected_rpm = 150
    assert spin_rate == pytest.approx(expected_rpm, rel=0.1)

def test_compute_spin_axis():
    tracker = OrientationTracker()

    from scipy.spatial.transform import Rotation

    R1 = Rotation.from_euler('z', 0).as_matrix()
    R2 = Rotation.from_euler('z', np.pi/4).as_matrix()  # Rotation around Z

    tracker.add_orientation(R1, timestamp=0.0)
    tracker.add_orientation(R2, timestamp=0.1)

    axis = tracker.get_spin_axis()

    # Should be approximately [0, 0, 1] (Z-axis rotation)
    assert np.allclose(axis[:2], [0, 0], atol=0.1)
    assert abs(axis[2]) > 0.9
