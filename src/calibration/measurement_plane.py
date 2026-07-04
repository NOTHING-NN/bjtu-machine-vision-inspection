"""
measurement_plane.py — 测量平面毫米坐标标定

使用棋盘格 10 mm 单格建立 undistorted image pixel -> world mm 的平面单应性。
该单应性是尺寸测量的尺度来源；PCB 的 100 mm/94 mm 标称值不能参与这里。
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src import config
from src.calibration.calibrate_camera import (
    find_chessboard_corners,
    load_chessboard_images,
    undistort_image,
)
from src.utils import load_image, save_image


def _object_points_2d(pattern_size=None, square_size_mm: float = None) -> np.ndarray:
    if pattern_size is None:
        pattern_size = config.CHESSBOARD_PATTERN_SIZE
    if square_size_mm is None:
        square_size_mm = config.SQUARE_SIZE_MM

    points = np.zeros((pattern_size[0] * pattern_size[1], 2), np.float32)
    points[:, :] = np.mgrid[0:pattern_size[0],
                            0:pattern_size[1]].T.reshape(-1, 2)
    points *= square_size_mm
    return points


def create_measurement_plane_homography(
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]],
    output_path: Path = None,
) -> bool:
    """
    从第一张成功检测到棋盘格的标定图像生成测量平面单应性。

    注意：该棋盘格图像必须代表 PCB 拍摄时所在的测量平面，且相机位置不变。
    若相机或测量平面发生移动，需要重新采集平面标定图并重新生成该文件。
    """
    if output_path is None:
        output_path = config.MEASUREMENT_PLANE_HOMOGRAPHY_FILE

    camera_matrix, dist_coeffs = camera_params or (None, None)
    image_paths = load_chessboard_images()
    if not image_paths:
        print("[ERROR] 无棋盘格图像，无法建立测量平面单应性")
        return False

    world_points = _object_points_2d()

    for path in image_paths:
        image = load_image(path)
        if image is None:
            continue

        if camera_matrix is not None and dist_coeffs is not None:
            work_image = undistort_image(image, camera_matrix, dist_coeffs)
        else:
            work_image = image

        success, corners, gray = find_chessboard_corners(work_image)
        if not success:
            continue

        image_points = corners.reshape(-1, 2).astype(np.float32)
        image_to_world, _ = cv2.findHomography(image_points, world_points, 0)
        world_to_image, _ = cv2.findHomography(world_points, image_points, 0)
        if image_to_world is None or world_to_image is None:
            continue

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(output_path),
            image_to_world=image_to_world,
            world_to_image=world_to_image,
            reference_image=path.name,
            square_size_mm=config.SQUARE_SIZE_MM,
            pattern_size=np.array(config.CHESSBOARD_PATTERN_SIZE),
        )

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, config.CHESSBOARD_PATTERN_SIZE,
                                  corners, success)
        save_image(vis, config.CALIB_OUTPUT_DIR / "measurement_plane_reference.png")

        print(f"[INFO] 测量平面单应性已保存: {output_path}")
        print(f"[INFO] 平面参考图像: {path.name}")
        return True

    print("[ERROR] 没有可用于测量平面标定的棋盘格图像")
    return False


def load_measurement_plane_homography(
    file_path: Path = None,
) -> Optional[np.ndarray]:
    """加载 image pixel -> world mm 单应性矩阵。"""
    if file_path is None:
        file_path = config.MEASUREMENT_PLANE_HOMOGRAPHY_FILE

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[WARNING] 测量平面单应性不存在: {file_path}")
        print("  请运行: python main.py calibrate")
        return None

    data = np.load(str(file_path), allow_pickle=True)
    return data["image_to_world"]


def transform_points_to_world(
    points_px: np.ndarray,
    image_to_world: np.ndarray,
) -> np.ndarray:
    """将图像像素点转换为测量平面毫米坐标。"""
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(points, image_to_world)
    return world.reshape(-1, 2)
