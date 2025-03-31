from math import factorial

import numpy as np
import pandas as pd
from numpy.linalg import det
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import dirichlet


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

def cull_points(camera_position, rotation_matrix, points, fov_degrees=60):
    """
    Cull points that are not within the camera's field of view.

    Parameters:
    - camera_position (np.array): The position of the camera in 3D space, shape (3,).
    - rotation_matrix (np.array): The 3x3 rotation matrix defining the camera's orientation.
      It is assumed that the third column is the forward (view) direction.
    - points (np.array): Array of points in 3D space, shape (N, 3).
    - fov_degrees (float): The camera's full field of view in degrees.
      Points within half this angle from the camera's forward direction are considered visible.

    Returns:
    - np.array: The array of points that are considered visible.
    """

    half_fov_radians = np.radians(fov_degrees / 2)
    threshold = np.cos(half_fov_radians)

    forward_vector = rotation_matrix[:, 2]

    vectors = points - camera_position

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_vectors = vectors / norms

    dots = normalized_vectors.dot(forward_vector)

    visible_mask = dots > threshold

    visible_points = points[visible_mask]
    return visible_points

def compute_volume_corners(plane1: tuple, plane2: tuple, vector: tuple, distance: float) -> list[tuple[float, float, float]]:
    """
    Given two planes (each defined as (a, b, c, d) for the plane ax+by+cz+d=0),
    a vector (which may not lie on the plane), and a distance,
    compute 4 corner points on each plane (by projecting the vector onto the plane
    and offsetting by the given distance along two orthogonal in-plane axes).

    Returns a list of eight points as tuples of floats: list[tuple[float, float, float]].
    """
    def project_point_on_plane(point, plane):
        # Project a point onto the plane ax+by+cz+d = 0
        a, b, c, d = plane
        p = np.array(point, dtype=float)
        normal = np.array([a, b, c], dtype=float)
        # Calculate the signed distance from the point to the plane
        dist_to_plane = (np.dot(normal, p) + d) / np.dot(normal, normal)
        return p - dist_to_plane * normal

    def get_plane_axes(normal, reference_up=np.array([0, 0, 1], dtype=float)):
        # Ensure normal is a unit vector.
        normal = normal / np.linalg.norm(normal)
        # If the reference up vector is (nearly) parallel to the normal, choose an alternative.
        if np.isclose(abs(np.dot(normal, reference_up)), 1.0):
            reference_up = np.array([1, 0, 0], dtype=float)
        axis1 = np.cross(normal, reference_up)
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = np.cross(normal, axis1)
        axis2 = axis2 / np.linalg.norm(axis2)
        return axis1, axis2

    result_points = []

    # Process each plane
    for plane in [plane1, plane2]:
        # Extract plane normal and compute the center on the plane by projecting the given vector.
        a, b, c, d = plane
        normal = np.array([a, b, c], dtype=float)
        center = project_point_on_plane(vector, plane)
        # Get two orthogonal axes lying on the plane.
        axis1, axis2 = get_plane_axes(normal)
        # Compute the 4 corners on the plane:
        corners = [
            center + distance * (axis1 + axis2),
            center + distance * (axis1 - axis2),
            center + distance * (-axis1 + axis2),
            center + distance * (-axis1 - axis2)
        ]
        # Instead of converting each corner to integers, preserve precision as floats.
        # Here, each coordinate is rounded to three decimal places for clarity.
        for corner in corners:
            corner_float = tuple(round(coord, 3) for coord in corner)
            result_points.append(corner_float)

    return result_points
