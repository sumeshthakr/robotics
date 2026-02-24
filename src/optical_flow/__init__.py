"""Optical flow based rotation estimation for baseball orientation detection.

This module provides an alternative approach to seam-based orientation detection
by tracking feature points across consecutive frames using optical flow.
"""

from .rotation_estimator import RotationEstimator

__all__ = ["RotationEstimator"]
