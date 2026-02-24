import numpy as np

class BaseballSeamModel:
    """3D model of baseball seam geometry.

    Baseball has two curved seams that follow a specific pattern on the sphere.
    This model generates 3D points along both seam curves.
    """

    def __init__(self, radius=1.0):
        """Initialize seam model.

        Args:
            radius: Ball radius in desired units (default: 1.0 for normalized)
        """
        self.radius = radius

    def get_3d_points(self, num_points_per_curve=100) -> list:
        """Generate 3D points along both seam curves.

        The baseball seam can be approximated as two curves that are
        rotations of a single curve pattern.

        Args:
            num_points_per_curve: Number of 3D points per seam curve

        Returns:
            List of two Nx3 arrays, one for each seam curve
        """
        t = np.linspace(0, 2*np.pi, num_points_per_curve)

        # Baseball seam approximation: curve that goes around the sphere
        # One common parameterization:
        curve1 = self._generate_seam_curve(t, phase_offset=0)
        curve2 = self._generate_seam_curve(t, phase_offset=np.pi)

        return [curve1, curve2]

    def _generate_seam_curve(self, t, phase_offset=0) -> np.ndarray:
        """Generate one seam curve.

        Uses a mathematical approximation of the baseball seam shape.
        The seam is a curve on sphere that makes approximately 2.5 revolutions.

        Args:
            t: Parameter values (0 to 2*pi)
            phase_offset: Phase shift for this curve

        Returns:
            Nx3 array of 3D points
        """
        # Baseball seam makes approximately 2.5 turns around the ball
        num_revolutions = 2.5

        # Parameterize curve on sphere
        # Using spherical coordinates with varying inclination
        phi = t * num_revolutions + phase_offset  # Azimuthal angle
        theta = np.pi/2 + 0.4 * np.sin(2.5 * t)   # Polar angle (varies)

        # Convert to Cartesian coordinates
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        return np.column_stack([x, y, z])

    def get_all_points(self) -> np.ndarray:
        """Get all seam points as a single array.

        Returns:
            (2*N)x3 array of all seam 3D points
        """
        curves = self.get_3d_points()
        return np.vstack(curves)
