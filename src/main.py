import numpy as np
import cv2

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    Rotation order: Rz (yaw) -> Ry (pitch) -> Rx (roll)
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,           0,            1]
    ])
    R = Rz @ Ry @ Rx
    return R

def project_world_point_to_image(world_point, drone_pos, roll, pitch, yaw, K, R_mount=None):
    """
    Project a 3D world point onto the camera image plane.
    
    Parameters:
    - world_point: numpy array, the 3D point in world coordinates.
    - drone_pos: numpy array, the position of the drone (camera center) in world coordinates.
    - roll, pitch, yaw: floats, drone's orientation in radians.
    - K: numpy array, camera intrinsic parameter matrix.
    - R_mount: numpy array, mounting correction matrix (default: identity).
    
    Returns:
    - A numpy array containing the [u, v] pixel coordinates.
    """
    if R_mount is None:
        R_mount = np.eye(3)
    
    # Calculate the rotation matrix from drone orientation
    R_body = euler_to_rotation_matrix(roll, pitch, yaw)
    # Effective rotation matrix including mounting correction
    R_eff = R_mount @ R_body
    # Transform world point to camera coordinates
    X_cam = R_eff @ (world_point - drone_pos)
    
    # Check if the point is in front of the camera (Z > 0)
    if X_cam[2] <= 0:
        print("Warning: Point is behind the camera!")
        return None
    
    # Perspective projection (in homogeneous coordinates)
    x_proj = K @ X_cam
    u = x_proj[0] / x_proj[2]
    v = x_proj[1] / x_proj[2]
    return np.array([u, v])

def main():
    # --- Camera intrinsic parameters ---
    # Example: focal lengths (fx, fy) and principal point (cx, cy) in pixels.
    fx = 800
    fy = 800
    cx = 640
    cy = 360
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ])
    
    # --- Drone position and orientation ---
    # Drone position in world coordinates (in meters)
    drone_pos = np.array([0, 0, 100])  # Example: 100m altitude
    # Drone orientation (roll, pitch, yaw) in radians. For simplicity, assume level flight.
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    
    # --- Camera mounting correction matrix ---
    # For a downward facing camera, convert world downward direction (negative Z)
    # to camera's forward direction (positive Z) by flipping the Y and Z axes.
    R_mount = np.diag([1, -1, -1])
    
    # --- Target world point ---
    # Example: A ground point in local coordinates (e.g., converted from GPS)
    target_world_point = np.array([50, 30, 0])
    
    # --- Method 1: Using custom projection function ---
    pixel_coords = project_world_point_to_image(
        world_point=target_world_point,
        drone_pos=drone_pos,
        roll=roll, pitch=pitch, yaw=yaw,
        K=K,
        R_mount=R_mount
    )
    if pixel_coords is not None:
        print("Projection result (pixel coordinates) using custom function:", pixel_coords)
    
    # --- Method 2: Using OpenCV's projectPoints ---
    R_body = euler_to_rotation_matrix(roll, pitch, yaw)
    R_eff = R_mount @ R_body
    # Convert rotation matrix to rotation vector using Rodrigues formula
    rvec, _ = cv2.Rodrigues(R_eff)
    # tvec is the camera center in camera coordinates: tvec = -R_eff * drone_pos
    tvec = -R_eff @ drone_pos.reshape(3, 1)
    world_points = np.array([target_world_point], dtype=np.float32)
    image_points, _ = cv2.projectPoints(world_points, rvec, tvec, K, None)
    print("Projection result (pixel coordinates) using OpenCV projectPoints:", image_points.ravel())

if __name__ == '__main__':
    main()
