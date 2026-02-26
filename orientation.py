"""Orientation tracking and rotation math utilities.

Tracks baseball orientation over time to compute:
  - Spin rate (RPM)
  - Spin axis (3D unit vector)

Also provides rotation format conversions using scipy.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# Rotation Format Conversions
# ============================================================

def rotation_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].

    We use the scalar-first convention: q = w + xi + yj + zk
    where w is the real part.

    Args:
        R: 3x3 rotation matrix

    Returns:
        numpy array [w, x, y, z]
    """
    q = Rotation.from_matrix(R).as_quat()  # scipy gives [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # reorder to [w, x, y, z]


def rotation_to_euler(R):
    """Convert a 3x3 rotation matrix to Euler angles [roll, pitch, yaw].

    Uses the XYZ convention (intrinsic rotations).

    Args:
        R: 3x3 rotation matrix

    Returns:
        numpy array [roll, pitch, yaw] in radians
    """
    return Rotation.from_matrix(R).as_euler('xyz')


# ============================================================
# Orientation Tracker
# ============================================================

class OrientationTracker:
    """Track ball orientation over time to compute spin rate and spin axis.

    Keeps a sliding window of recent (rotation_matrix, timestamp) pairs.
    Spin rate and axis are computed from the RELATIVE rotation between
    consecutive frames:

        R_relative = R_prev^T @ R_curr

    This tells us "how much did the ball rotate between frames?"

    Math for spin rate:
        angle = magnitude of axis-angle representation of R_relative
        omega = angle / dt               (rad/s)
        RPM   = omega * 60 / (2 * pi)    (revolutions per minute)
    """

    def __init__(self, window_size=10):
        """
        Args:
            window_size: Number of recent measurements to keep
        """
        self.history = []  # List of (rotation_matrix, timestamp)
        self.window_size = window_size

    def add(self, rotation_matrix, timestamp):
        """Record a new orientation measurement.

        Args:
            rotation_matrix: 3x3 numpy rotation matrix
            timestamp:       Time in seconds
        """
        self.history.append((np.array(rotation_matrix), timestamp))
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_spin_rate(self):
        """Compute spin rate in RPM from the last two measurements.

        Returns:
            Spin rate in RPM, or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        R_prev, t_prev = self.history[-2]
        R_curr, t_curr = self.history[-1]

        dt = t_curr - t_prev
        if dt < 1e-6:
            return None

        # Relative rotation: how much did the ball rotate?
        R_relative = R_prev.T @ R_curr

        # Extract rotation angle (length of axis-angle vector)
        rotvec = Rotation.from_matrix(R_relative).as_rotvec()
        angle = np.linalg.norm(rotvec)  # radians

        # Convert: rad/s → RPM
        omega = angle / dt              # angular velocity (rad/s)
        rpm = omega * 60 / (2 * np.pi)  # revolutions per minute
        return rpm

    def get_spin_axis(self):
        """Get the current spin axis as a 3D unit vector.

        The spin axis is the direction around which the ball is rotating.
        Extracted from the rotation vector of the relative rotation.

        Returns:
            3D unit vector, or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        R_prev, _ = self.history[-2]
        R_curr, _ = self.history[-1]

        R_relative = R_prev.T @ R_curr
        rotvec = Rotation.from_matrix(R_relative).as_rotvec()
        magnitude = np.linalg.norm(rotvec)

        if magnitude < 1e-6:
            return np.array([0, 0, 1])  # Default if no rotation detected

        return rotvec / magnitude  # Normalize to unit vector
