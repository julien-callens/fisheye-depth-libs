# File: vector_math/__init__.py

"""
Vector Math Package

This package provides various mathematical operations on 3D vectors.
"""

from .operations import (
    magnitude,
    normalize,
    plane_eq,
    shift_plane,
    intersect_line_plane,
    rodrigues_rotation_matrix,
    quat_multiply,
    quat_conjugate,
    rotate_vector_quaternion,
    random_points_in_volume
)

from .display import (
    draw_vec,
    draw_plane
)
