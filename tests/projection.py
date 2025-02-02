import numpy as np
from src.main import euler_to_rotation_matrix, project_world_point_to_image
import cv2

def test_projection_in_front():
    # Test that a point in front of the camera projects to valid pixel coordinates.
    fx, fy, cx, cy = 800, 800, 640, 360
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ])
    drone_pos = np.array([0, 0, 100])
    roll, pitch, yaw = 0.0, 0.0, 0.0
    R_mount = np.diag([1, -1, -1])
    target_world_point = np.array([50, 30, 0])
    
    pixel_coords = project_world_point_to_image(
        world_point=target_world_point,
        drone_pos=drone_pos,
        roll=roll, pitch=pitch, yaw=yaw,
        K=K,
        R_mount=R_mount
    )
    # Check that projection returns a valid array and that coordinates are within a reasonable range.
    assert pixel_coords is not None
    assert pixel_coords[0] > 0 and pixel_coords[1] > 0

def test_point_behind_camera(capfd):
    # Test that a point behind the camera triggers a warning.
    fx, fy, cx, cy = 800, 800, 640, 360
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ])
    drone_pos = np.array([0, 0, 100])
    roll, pitch, yaw = 0.0, 0.0, 0.0
    R_mount = np.diag([1, -1, -1])
    # Create a point that is behind the camera by setting z > drone_pos.z
    target_world_point = np.array([0, 0, 200])
    
    pixel_coords = project_world_point_to_image(
        world_point=target_world_point,
        drone_pos=drone_pos,
        roll=roll, pitch=pitch, yaw=yaw,
        K=K,
        R_mount=R_mount
    )
    # Should print a warning and return None
    out, err = capfd.readouterr()
    assert "Warning: Point is behind the camera!" in out
    assert pixel_coords is None
