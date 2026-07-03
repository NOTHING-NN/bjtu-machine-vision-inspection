"""
calibrate_camera.py — 相机标定模块

使用黑白棋盘格对相机进行常规标定。
标定板仅做灰度化、角点检测和亚像素优化，不进行光照矫正。

输出：
  - outputs/calibration/camera_params.npz  (相机内参和畸变系数)
  - outputs/calibration/corners_*.png     (角点检测可视化图)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils import get_image_paths, load_image, save_image, ensure_grayscale


def get_subpix_criteria() -> Tuple[int, int, float]:
    """获取亚像素角点优化的终止条件。"""
    return (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.SUBPIX_MAX_ITER,
        config.SUBPIX_EPSILON,
    )


def load_chessboard_images() -> List[Path]:
    """加载棋盘格标定图像路径列表。"""
    paths = get_image_paths(config.CALIB_IMAGE_DIR)
    print(f"[INFO] 找到 {len(paths)} 张棋盘格标定图像")
    return paths


def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    在一张图像中检测棋盘格内角点，并进行亚像素优化。
    仅做灰度化和角点检测，不进行光照矫正。

    Returns:
        (success, corners_subpix, gray_image)
    """
    if pattern_size is None:
        pattern_size = config.CHESSBOARD_PATTERN_SIZE

    gray = ensure_grayscale(image)

    success, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not success:
        return False, None, gray

    subpix_criteria = get_subpix_criteria()
    corners_subpix = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), subpix_criteria
    )

    return True, corners_subpix, gray


def calibrate_camera(
    image_paths: List[Path],
    pattern_size: Tuple[int, int] = None,
    square_size_mm: float = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray],
           Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """对一批棋盘格图像执行相机标定，输出相机内参和畸变系数。"""
    if pattern_size is None:
        pattern_size = config.CHESSBOARD_PATTERN_SIZE
    if square_size_mm is None:
        square_size_mm = config.SQUARE_SIZE_MM

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                           0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []
    imgpoints = []
    good_images = []

    for idx, path in enumerate(image_paths):
        img = load_image(path)
        if img is None:
            continue

        success, corners, gray = find_chessboard_corners(img, pattern_size)

        if success:
            objpoints.append(objp)
            imgpoints.append(corners)
            good_images.append(idx)

            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, success)
            save_path = config.CALIB_OUTPUT_DIR / f"corners_{path.stem}.png"
            save_image(vis, save_path)
            print(f"  [OK] {path.name} — 角点检测成功")
        else:
            print(f"  [SKIP] {path.name} — 无法检测到所有角点，跳过")

    if len(objpoints) == 0:
        print("[ERROR] 没有成功检测到任何棋盘格角点，标定失败")
        return False, None, None, None

    print(f"\n[INFO] 共 {len(objpoints)} 张图像可用于标定")

    first_good_img = load_image(image_paths[good_images[0]])
    if first_good_img is None:
        return False, None, None, None
    h, w = first_good_img.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    if not ret:
        print("[ERROR] 相机标定不收敛")
        return False, None, None, None

    print(f"\n[INFO] 标定完成，重投影误差 RMS = {ret:.4f} px")
    print(f"[INFO] 相机内参矩阵:\n{mtx}")
    print(f"[INFO] 畸变系数: {dist.ravel()}")

    return True, mtx, dist, (rvecs, tvecs, objpoints, imgpoints)


def save_camera_params(camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray,
                       output_path: Path = None) -> None:
    """将相机参数保存为 .npz 文件。"""
    if output_path is None:
        output_path = config.CALIB_OUTPUT_DIR / "camera_params.npz"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path),
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print(f"[INFO] 相机参数已保存到: {output_path}")


def load_camera_params(file_path: Path = None) -> Tuple[Optional[np.ndarray],
                                                        Optional[np.ndarray]]:
    """从 .npz 文件加载相机参数。"""
    if file_path is None:
        file_path = config.CALIB_OUTPUT_DIR / "camera_params.npz"

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[WARNING] 相机参数文件不存在: {file_path}")
        print("  请先运行: python main.py calibrate")
        return None, None

    data = np.load(str(file_path))
    return data["camera_matrix"], data["dist_coeffs"]


def undistort_image(image: np.ndarray,
                    camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """对图像进行畸变校正（去畸变）。"""
    if camera_matrix is None or dist_coeffs is None:
        print("[WARNING] 相机参数无效，跳过畸变校正")
        return image

    h, w = image.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs,
                                None, new_mtx)
    return undistorted


def run_calibration() -> None:
    """执行完整的相机标定流程。"""
    print("=" * 60)
    print("  相机标定")
    print("=" * 60)

    config.ensure_output_dirs()

    image_paths = load_chessboard_images()
    if len(image_paths) == 0:
        print("[ERROR] 未找到棋盘格标定图像，请将图像放入:")
        print(f"  {config.CALIB_IMAGE_DIR}")
        return

    print(f"\n[INFO] 棋盘格内角点数量: {config.CHESSBOARD_PATTERN_SIZE}")
    print(f"[INFO] 棋盘格单格边长:    {config.SQUARE_SIZE_MM} mm")

    success, mtx, dist, _ = calibrate_camera(image_paths)

    if success:
        save_camera_params(mtx, dist)
        print("\n[INFO] 相机标定流程完成 [OK]")
    else:
        print("\n[ERROR] 相机标定流程失败 [FAIL]")
