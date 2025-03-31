"""
Vector Math Operations Module

This module provides various functions for 3D vector arithmetic and operations.
"""

import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from scipy.linalg import det
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import factorial

Vector3 = Tuple[float, float, float]

def magnitude(v: Vector3) -> float:
    """
    Calculate the magnitude (Euclidean norm) of a 3D vector.

    Args:
        v (Vector3): A 3D vector represented as a tuple (x, y, z).

    Returns:
        float: The magnitude of the vector.

    Example:
        >>> magnitude((3, 4, 0))
        5.0
    """
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

def normalize(v: Vector3) -> Vector3:
    """
    Normalize a 3D vector to produce a unit vector with the same direction.

    Args:
        v (Vector3): A 3D vector represented as a tuple (x, y, z).

    Returns:
        Vector3: A unit vector (vector with a magnitude of 1) in the same direction as v.

    Raises:
        ValueError: If the input vector is the zero vector, which cannot be normalized.

    Example:
        >>> normalize((3, 0, 0))
        (1.0, 0.0, 0.0)
    """
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return v[0] / mag, v[1] / mag, v[2] / mag

def plane_eq(pos, norm):
    """
    Computes the plane equation coefficients (a, b, c, d) given a point and a normal vector.

    The plane equation is: a*x + b*y + c*z + d = 0.

    Parameters:
    - pos: A point [x0, y0, z0] on the plane.
    - norm: The normal vector [a, b, c] to the plane.

    Returns:
    - a, b, c, d: Coefficients for the plane equation.
    """
    a, b, c = norm
    x0, y0, z0 = pos
    return a, b, c, -(a*x0 + b*y0 + c*z0)

def shift_plane(plane_eq, shift):
    """
    Shifts a plane in the direction of its normal vector.

    Parameters:
    - plane_eq: A tuple (a, b, c, d) representing the plane equation.
    - shift: The distance to shift the plane along its normal vector.

    Returns:
    - a, b, c, d: Coefficients for the shifted plane equation.
    """
    a, b, c, d = plane_eq
    norm = math.sqrt(a**2 + b**2 + c**2)
    return a, b, c, d - shift * norm

def intersect_line_plane(plane, point, direction):
    """
    Calculate the intersection of a line and a plane.

    Parameters:
      plane (tuple): A tuple (a, b, c, d) representing the plane equation
                     ax + by + cz + d = 0.
      point (tuple): A point (P_x, P_y, P_z) on the line.
      direction (tuple): A direction vector (d_x, d_y, d_z) for the line.

    Returns:
      tuple: The intersection point (x, y, z).

    Raises:
      ValueError: If the line is parallel to the plane (i.e., no unique intersection).
    """
    a, b, c, d = plane
    P_x, P_y, P_z = point
    d_x, d_y, d_z = direction

    # Compute the denominator: a*d_x + b*d_y + c*d_z.
    denominator = a * d_x + b * d_y + c * d_z

    if abs(denominator) < 1e-8:
        raise ValueError("The line is parallel to the plane; no unique intersection exists.")

    # Calculate t using the formula.
    t = -(a * P_x + b * P_y + c * P_z + d) / denominator

    # Compute the intersection point.
    x = P_x + t * d_x
    y = P_y + t * d_y
    z = P_z + t * d_z

    return x, y, z

def rodrigues_rotation_matrix(axis, theta):
    """
    Compute the rotation matrix using Rodrigues' rotation formula.

    Parameters:
    - axis: A 3-element array representing the rotation axis (will be normalized).
    - theta: The rotation angle in radians.

    Returns:
    - A 3x3 numpy array representing the rotation matrix.
    """
    # Ensure the axis is a numpy array and normalize it
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # Extract the components of the normalized axis
    u_x, u_y, u_z = axis

    # Create the skew-symmetric matrix K from the axis vector
    K = np.array([[0, -u_z, u_y],
                  [u_z, 0, -u_x],
                  [-u_y, u_x, 0]])

    # Identity matrix
    I = np.eye(3)

    # Rodrigues' rotation formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return R

def quat_multiply(q1, q2):
    """
    Multiply two quaternions.
    Quaternion components are in the order: (w, x, y, z)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

def quat_conjugate(q):
    """
    Return the conjugate of a quaternion.
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vector_quaternion(v, axis, theta):
    """
    Rotate vector v by angle theta (in radians) about the given local axis using quaternion rotation.

    Parameters:
      v     : 3D vector as a numpy array (e.g., np.array([x, y, z])).
      axis  : String indicating the axis of rotation ('x', 'y', or 'z').
              This axis is interpreted as being defined in the object's local coordinate frame.
      theta : Rotation angle in radians.

    Returns:
      The rotated 3D vector as a numpy array.

    How it works:
      1. Constructs a quaternion q for the rotation.
         - For a rotation about the z‑axis, q = [cos(theta/2), 0, 0, sin(theta/2)].
         - Similarly for x and y axes.
      2. Converts the vector into a pure quaternion (with real part 0).
      3. Performs the rotation using: v' = q * v_quat * q_conjugate.
      4. Extracts and returns the vector part of the result.
    """
    if axis == 'x':
        q = np.array([np.cos(theta/2), np.sin(theta/2), 0, 0])
    elif axis == 'y':
        q = np.array([np.cos(theta/2), 0, np.sin(theta/2), 0])
    elif axis == 'z':
        q = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    v_quat = np.array([0, v[0], v[1], v[2]])

    q_conj = quat_conjugate(q)

    v_rot_quat = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return v_rot_quat[1:]

def random_points_in_volume(vertices: "list[tuple[int, int, int]]", num_points: "int"):
    """
    Generate uniformly distributed random points inside the convex volume
    defined by the given vertices.

        :param vertices: Eight points (3D coordinates) defining the volume's corners.
        :param num_points: Number of random interior points to generate.

        :return pd.DataFrame: DataFrame with columns ['x', 'y', 'z'] of generated point coordinates.
    """
    vertices = np.array(vertices, dtype=float)
    if vertices.shape[0] < 4:
        raise ValueError("Need at least 4 points to define a 3D volume (got fewer).")

    # Compute the convex hull to get the polyhedron's vertices (in case input not ordered)
    hull = ConvexHull(vertices)
    hull_points = vertices[hull.vertices]

    # Delaunay triangulation to tetrahedralize the convex hull
    delaunay = Delaunay(hull_points)
    tetrahedra = hull_points[delaunay.simplices]

    # Compute volumes of each tetrahedron
    # Volume of tetrahedron = |det(v1-v4, v2-v4, v3-v4)| / 6
    v1, v2, v3, v4 = tetrahedra[:, 0, :], tetrahedra[:, 1, :], tetrahedra[:, 2, :], tetrahedra[:, 3, :]
    # Compute determinant for each tetra (using vectorized formula)
    # Form vectors for three edges emanating from v4: (v1-v4, v2-v4, v3-v4)
    mat = np.stack((v1 - v4, v2 - v4, v3 - v4), axis=1)
    det_values = np.abs(det(mat))
    volumes = det_values / factorial(3)

    # Choose tetrahedron indices for each point, weighted by volume
    probabilities = volumes / volumes.sum()
    tetra_choices = np.random.choice(len(tetrahedra), size=num_points, p=probabilities)

    # For each chosen tetrahedron, generate a random point using barycentric coords
    # Draw random barycentric weights from Dirichlet(1,1,1,1) for each point
    barycentric_weights = dirichlet.rvs([1, 1, 1, 1], size=num_points)
    # Use Einstein summation to multiply each set of weights with the tetra's vertices
    chosen_tetra_vertices = tetrahedra[tetra_choices]
    points = np.einsum('ij, ijk->ik', barycentric_weights, chosen_tetra_vertices)

    return pd.DataFrame(points, columns=['x', 'y', 'z'])
