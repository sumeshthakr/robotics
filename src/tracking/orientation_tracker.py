import numpy as np
from scipy.spatial.transform import Rotation

class OrientationTracker:
    """Track ball orientation over time to compute spin rate and axis."""

    def __init__(self, window_size=10):
        """Initialize tracker.

        Args:
            window_size: Number of frames to keep in history
        """
        self.history = []
        self.window_size = window_size

    def add_orientation(self, rotation_matrix, timestamp):
        """Add a new orientation measurement.

        Args:
            rotation_matrix: 3x3 rotation matrix
            timestamp: Frame timestamp in seconds
        """
        self.history.append({
            "rotation_matrix": np.array(rotation_matrix),
            "timestamp": timestamp
        })

        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_spin_rate(self):
        """Compute current spin rate in RPM.

        Returns:
            Spin rate in RPM, or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        # Use last two orientations
        R1 = self.history[-2]["rotation_matrix"]
        R2 = self.history[-1]["rotation_matrix"]
        t1 = self.history[-2]["timestamp"]
        t2 = self.history[-1]["timestamp"]

        # Compute relative rotation: R_rel = R1^T * R2
        R_rel = R1.T @ R2

        # Convert to axis-angle
        rot = Rotation.from_matrix(R_rel)
        angle = rot.as_rotvec()  # Rotation vector (axis * angle)

        # Magnitude is the rotation angle in radians
        rotation_angle = np.linalg.norm(angle)
        if rotation_angle < 1e-6:
            return 0.0

        # Time difference
        dt = t2 - t1
        if dt < 1e-6:
            return None

        # Angular velocity in rad/s
        omega = rotation_angle / dt

        # Convert to RPM
        rpm = omega * 60 / (2 * np.pi)

        return rpm

    def get_spin_axis(self):
        """Compute current spin axis in camera coordinates.

        Returns:
            3D unit vector representing spin axis, or None
        """
        if len(self.history) < 2:
            return None

        R1 = self.history[-2]["rotation_matrix"]
        R2 = self.history[-1]["rotation_matrix"]

        # Compute relative rotation
        R_rel = R1.T @ R2

        # Convert to rotation vector
        rot = Rotation.from_matrix(R_rel)
        rotvec = rot.as_rotvec()

        magnitude = np.linalg.norm(rotvec)
        if magnitude < 1e-6:
            return np.array([0, 0, 1])  # Default axis if no rotation

        # Normalize to get axis
        axis = rotvec / magnitude

        return axis

    def get_current_orientation(self):
        """Get the most recent orientation.

        Returns:
            dict with rotation_matrix, quaternion, euler_angles or None
        """
        if len(self.history) == 0:
            return None

        R = self.history[-1]["rotation_matrix"]

        # Compute quaternion
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w] scalar-last
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        # Compute Euler angles
        euler = rot.as_euler('xyz')  # [roll, pitch, yaw]

        return {
            "rotation_matrix": R,
            "quaternion": quat_wxyz,
            "euler_angles": euler
        }
