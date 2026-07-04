"""
src.calibration — 相机标定层

提供：
  - 棋盘格角点检测与亚像素优化
  - 相机内参 / 畸变系数估计
  - 参数持久化（.npz）
  - 图像畸变校正
"""

from src.calibration.calibrate_camera import (
    calibrate_camera,
    find_chessboard_corners,
    get_subpix_criteria,
    load_camera_params,
    load_chessboard_images,
    run_calibration,
    save_camera_params,
    undistort_image,
)
from src.calibration.measurement_plane import (
    create_measurement_plane_homography,
    load_measurement_plane_homography,
    transform_points_to_world,
)
