import numpy as np

from vecmath import normalize


def draw_vec(ax, pos, direction, length=0.1, color='r', arrow_length_ratio=0.1):
    """
    Adds a vector (as an arrow) to a given 3D plot.

    Parameters:
    - ax: The 3D axis object from matplotlib.
    - pos: A sequence (list, tuple, or array) with three elements (x, y, z) specifying the starting position of the vector.
    - direction: A sequence with three elements (dx, dy, dz) representing the vector's direction.
    - length: The length of the arrow (default is 0.1).
    - color: The color of the arrow (default is 'r' for red).
    - arrow_length_ratio: The ratio of the arrow head size relative to the arrow length (default is 0.1).
    """
    ax.quiver(pos[0], pos[1], pos[2],
              direction[0], direction[1], direction[2],
              length=length, color=color, arrow_length_ratio=arrow_length_ratio)

def draw_plane(ax, a, b, c, d, size=0.1, resolution=20, color='r', alpha=0.3):
    """
    Draw a plane patch based on the plane equation: a*x + b*y + c*z + d = 0.

    Parameters:
        ax         : The 3D axis to plot on.
        a, b, c, d : Coefficients for the plane equation.
        size       : The half-size of the patch (the total patch will be 2*size x 2*size).
        resolution : The number of points in each direction (controls mesh resolution).
        color      : The color for the plane.
        alpha      : The transparency level for the plane.
    """
    # Compute the center of the plane (closest point to the origin)
    normal = np.array([a, b, c])
    norm_sq = np.dot(normal, normal)
    if norm_sq == 0:
        raise ValueError("The normal vector cannot be zero.")
    center = -d / norm_sq * normal

    # Create an arbitrary vector that is not collinear with the normal
    arbitrary = np.array([0, 0, 1])
    if np.allclose(normal, arbitrary) or np.allclose(normal, -arbitrary):
        arbitrary = np.array([0, 1, 0])

    # Compute two vectors spanning the plane using the cross product
    v1 = np.cross(normal, arbitrary)
    v1 = normalize(v1)
    v2 = np.cross(normal, v1)
    v2 = normalize(v2)

    # Create a grid in the parameter space (u, v)
    u = np.linspace(-size, size, resolution)
    v = np.linspace(-size, size, resolution)
    U, V = np.meshgrid(u, v)

    # Parametric equation of the plane patch: P(u,v) = center + u*v1 + v*v2
    X = center[0] + U * v1[0] + V * v2[0]
    Y = center[1] + U * v1[1] + V * v2[1]
    Z = center[2] + U * v1[2] + V * v2[2]

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
