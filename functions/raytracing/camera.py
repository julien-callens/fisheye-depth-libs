import math

import numpy as np
import pandas as pd

from functions.raytracing import compute_volume_corners, random_points_in_volume, cull_points
from vecmath import shift_plane, plane_eq, normalize


def project_points_to_sensor(points_world, cam_pos, R_cam, focal_length, sensor_radius_equisolid, sensor_diameter,
                             sensor_resolution):
    """
    For each point in world coordinates, compute its projection on the camera sensor using
    an equisolid fisheye model scaled to a physical sensor size.

    Steps:
    - Compute the vector from the camera center to each point.
    - Transform the vector into the camera coordinate system.
    - Discard points behind the camera (z <= 0 in camera space).
    - Normalize the vectors and compute the angle theta between each ray and the optical (z) axis.
    - Apply the equisolid fisheye model: r = 2 * focal_length * sin(theta/2)
    - Determine the azimuth angle phi.
    - Compute preliminary sensor coordinates (u, v) in mm (with sensor center at (0,0)).
    - Scale these coordinates to the physical sensor size.
    - Filter out points falling outside the sensor’s circular area.
    - Map the sensor coordinates (in mm) to pixel coordinates.

    Parameters:
      points_world           : (N,3) array of 3D world points.
      cam_pos                : Camera position (3-element array).
      R_cam                  : 3x3 rotation matrix (world-to-camera).
      focal_length           : Focal length in mm.
      sensor_radius_equisolid: Maximum radius from equisolid projection (in mm).
      sensor_diameter        : Physical sensor diameter in mm.
      sensor_resolution      : Sensor resolution (assumed square).

    Returns:
      pixel_y, pixel_x: Arrays of pixel coordinates (in pixels) for points within the sensor.
    """
    # 1. Compute the vector from camera to each point (world space)
    vectors = points_world - np.array(cam_pos)  # shape: (N, 3)

    # 2. Transform vectors into the camera coordinate system.
    #    Using R_cam.T (the inverse of R_cam) transforms world to camera space.
    vectors_cam = (R_cam.T @ vectors.T).T  # shape: (N, 3)

    # 3. Discard points behind the camera (z <= 0)
    front_mask = vectors_cam[:, 2] > 0
    vectors_cam = vectors_cam[front_mask]

    # 4. Normalize camera-space vectors
    norms = np.linalg.norm(vectors_cam, axis=1)
    v_cam_norm = vectors_cam / norms[:, np.newaxis]

    # 5. Compute the incident angle theta (angle between ray and optical axis)
    theta = np.arccos(v_cam_norm[:, 2])

    # 6. Equisolid fisheye projection: r = 2 * focal_length * sin(theta/2)
    r = 2 * focal_length * np.sin(theta / 2)

    # 7. Compute the azimuth angle phi in the sensor plane
    phi = np.arctan2(v_cam_norm[:, 1], v_cam_norm[:, 0])

    # 8. Preliminary sensor coordinates (in mm, with center at (0,0))
    u = r * np.cos(phi)
    v = r * np.sin(phi)

    # 9. Scale equisolid coordinates to physical sensor size.
    #    Effective sensor radius (physical) is half the sensor diameter.
    effective_sensor_radius = sensor_diameter / 2
    scale = effective_sensor_radius / sensor_radius_equisolid
    u_scaled = u * scale
    v_scaled = v * scale

    # 10. Filter points that fall outside the physical sensor circle.
    within_sensor = np.sqrt(u_scaled ** 2 + v_scaled ** 2) <= effective_sensor_radius
    u_scaled = u_scaled[within_sensor]
    v_scaled = v_scaled[within_sensor]

    # 11. Convert sensor coordinates (in mm) to pixel coordinates.
    #     Sensor coordinates are in range [-effective_sensor_radius, effective_sensor_radius].
    #     We map these to [0, sensor_resolution].
    pixel_x = ((u_scaled + effective_sensor_radius) / sensor_diameter) * sensor_resolution
    pixel_y = ((-v_scaled + effective_sensor_radius) / sensor_diameter) * sensor_resolution

    return pixel_x, pixel_y


def generate_dataset_two_cameras(
        world_up=np.array([0, 0, 1]),
        cam1_pos=np.array([0.05, 0.15, 0.15]),
        cam2_pos=np.array([-0.05, 0.15, 0.15]),
        look_point_cam1=np.array([0, 0, 0]),
        look_point_cam2=np.array([0, 0, 0]),
        horizontal_fov=175,
        vertical_fov=175,
        min_depth=-10,
        max_depth=150,
        volume_width=100,
        sensor_resolution=720,
        focal_length=10.5,
        sensor_diameter=36,
        num_data_points=10_000_000
):
    """
    Generates a merged DataFrame containing:
      - 3D positions (x, y, z) of culled random points
      - Projected sensor coordinates for two cameras (u_L, v_L, u_R, v_R)

    The final DataFrame has the following columns:
        ['frame_index', 'x', 'y', 'z', 'u_L', 'v_L', 'u_R', 'v_R']

    Parameters:
      world_up: 3-element array for the world up vector.
      cam1_pos, cam2_pos: 3-element arrays for the positions of camera 1 and camera 2.
      look_point_cam1, look_point_cam2: 3-element arrays for the look points of each camera.
      horizontal_fov, vertical_fov: Field of view in degrees (vertical_fov is kept for parameter consistency).
      min_depth, max_depth: Depth limits for volume generation.
      volume_width: The width of the volume from which random points are generated.
      sensor_resolution: The sensor resolution for projection.
      focal_length: Focal length used in projection.
      sensor_diameter: Sensor diameter used in projection.
      num_data_points: Number of random 3D data points to generate.

    Returns:
      merged_df: Merged DataFrame with columns:
                 ['frame_index', 'x', 'y', 'z', 'u_L', 'v_L', 'u_R', 'v_R']
    """

    sensor_radius_equisolid = sensor_diameter / 2

    cam1_z = normalize(look_point_cam1 - cam1_pos)
    cam2_z = normalize(look_point_cam2 - cam2_pos)

    avg_look_point = (look_point_cam1 + look_point_cam2) / 2
    avg_pos = (cam1_pos + cam2_pos) / 2
    avg_z = normalize(avg_look_point - avg_pos)

    avg_plane = plane_eq(avg_pos, avg_z)
    near_plane = shift_plane(avg_plane, min_depth)
    far_plane = shift_plane(avg_plane, max_depth)

    def safe_cross_product(v1, v2, fallback=np.array([1, 0, 0])):
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        if norm < 1e-6:
            if np.abs(np.dot(fallback, v2)) > 0.9:
                fallback = np.array([0, 1, 0])
            cross = np.cross(fallback, v2)
        return normalize(cross)

    cam1_y = safe_cross_product(world_up, cam1_z)
    cam1_x = safe_cross_product(cam1_z, cam1_y)
    cam2_y = safe_cross_product(world_up, cam2_z)
    cam2_x = safe_cross_product(cam2_z, cam2_y)
    avg_y = safe_cross_product(world_up, avg_z)
    avg_x = safe_cross_product(avg_z, avg_y)

    R_cam1 = np.column_stack((cam1_x, cam1_y, cam1_z))
    R_cam2 = np.column_stack((cam2_x, cam2_y, cam2_z))

    volume_corners = compute_volume_corners(near_plane, far_plane, avg_pos, volume_width)
    points = np.array(random_points_in_volume(volume_corners, num_points=num_data_points))

    points = cull_points(cam1_pos, R_cam1, points, horizontal_fov)
    points = cull_points(cam2_pos, R_cam2, points, horizontal_fov)

    positions_df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    positions_df['frame_index'] = np.arange(len(points))

    pixel_y_cam1, pixel_x_cam1 = project_points_to_sensor(
        points, cam1_pos, R_cam1, focal_length,
        sensor_radius_equisolid, sensor_diameter, sensor_resolution
    )
    pixel_y_cam2, pixel_x_cam2 = project_points_to_sensor(
        points, cam2_pos, R_cam2, focal_length,
        sensor_radius_equisolid, sensor_diameter, sensor_resolution
    )

    num_points = len(pixel_x_cam1)
    df_camL = pd.DataFrame({
        'frame_index': np.arange(num_points),
        'file_name': [""] * num_points,
        'u': pixel_x_cam1,
        'v': pixel_y_cam1
    })
    df_camR = pd.DataFrame({
        'frame_index': np.arange(len(pixel_x_cam2)),
        'file_name': [""] * len(pixel_x_cam2),
        'u': pixel_x_cam2,
        'v': pixel_y_cam2
    })

    merged_df = pd.merge(df_camL, df_camR, on='frame_index', suffixes=('_L', '_R'))
    merged_df.dropna(inplace=True)
    merged_df = pd.merge(merged_df, positions_df, on='frame_index')[
        ['frame_index', 'x', 'y', 'z', 'u_L', 'v_L', 'u_R', 'v_R']
    ]

    return merged_df
