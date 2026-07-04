"""
geometry.py — 几何尺寸测量与误差计算

测量尺度只来自棋盘格 10 mm 单格建立的测量平面单应性。
PCB 的 100 mm 外框和 94 mm 孔距只用于最终误差评价，不能参与检测、
透视矫正尺度换算或孔位搜索。
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from src import config
from src.calibration import transform_points_to_world
from src.utils import DetectionResult


def _nan_measurement() -> Dict:
    data = {
        "board_width_top_mm": np.nan,
        "board_width_bottom_mm": np.nan,
        "board_height_left_mm": np.nan,
        "board_height_right_mm": np.nan,
        "board_width_mm": np.nan,
        "board_height_mm": np.nan,
        "board_width_error_mm": np.nan,
        "board_height_error_mm": np.nan,
        "resolution_x_mm_per_px": np.nan,
        "resolution_y_mm_per_px": np.nan,
        "resolution_mean_mm_per_px": np.nan,
        "resolution_x_px_per_mm": np.nan,
        "resolution_y_px_per_mm": np.nan,
        "resolution_mean_px_per_mm": np.nan,
        "resolution_mean_um_per_px": np.nan,
        "pitch_x_mm": np.nan,
        "pitch_y_mm": np.nan,
        "abs_error_x_mm": np.nan,
        "abs_error_y_mm": np.nan,
        "rel_error_x_pct": np.nan,
        "rel_error_y_pct": np.nan,
        "mean_pitch_mm": np.nan,
        "mean_abs_error_mm": np.nan,
        "hole_diameters_mm": [np.nan] * 4,
        "hole_centers_world_mm": [(np.nan, np.nan)] * 4,
    }
    for i in range(4):
        data[f"hole{i+1}_x_mm"] = np.nan
        data[f"hole{i+1}_y_mm"] = np.nan
        data[f"hole{i+1}_diameter_mm"] = np.nan
    return data


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def compute_board_dimensions(corners_world_mm: np.ndarray) -> Dict[str, float]:
    """由四角点毫米坐标计算电路板长宽。角点顺序：tl, tr, br, bl。"""
    tl, tr, br, bl = corners_world_mm
    width_top = _dist(tl, tr)
    width_bottom = _dist(bl, br)
    height_left = _dist(tl, bl)
    height_right = _dist(tr, br)
    width = (width_top + width_bottom) / 2.0
    height = (height_left + height_right) / 2.0

    return {
        "board_width_top_mm": width_top,
        "board_width_bottom_mm": width_bottom,
        "board_height_left_mm": height_left,
        "board_height_right_mm": height_right,
        "board_width_mm": width,
        "board_height_mm": height,
        "board_width_error_mm": abs(width - config.BOARD_WIDTH_MM),
        "board_height_error_mm": abs(height - config.BOARD_HEIGHT_MM),
    }


def compute_measurement_resolution(
    board_dimensions: Dict[str, float],
    board_size_px: Tuple[int, int],
) -> Dict[str, float]:
    """
    计算透视矫正测量图中的空间分辨率。

    这里使用检测外框在测量平面中的实际毫米长度，以及对应俯视图像素尺寸。
    """
    bw_px, bh_px = board_size_px
    width_mm = board_dimensions["board_width_mm"]
    height_mm = board_dimensions["board_height_mm"]

    res_x = width_mm / bw_px if bw_px > 0 else np.nan
    res_y = height_mm / bh_px if bh_px > 0 else np.nan
    res_mean = float(np.nanmean([res_x, res_y]))

    px_x = 1.0 / res_x if np.isfinite(res_x) and res_x > 0 else np.nan
    px_y = 1.0 / res_y if np.isfinite(res_y) and res_y > 0 else np.nan
    px_mean = 1.0 / res_mean if np.isfinite(res_mean) and res_mean > 0 else np.nan

    return {
        "resolution_x_mm_per_px": res_x,
        "resolution_y_mm_per_px": res_y,
        "resolution_mean_mm_per_px": res_mean,
        "resolution_x_px_per_mm": px_x,
        "resolution_y_px_per_mm": px_y,
        "resolution_mean_px_per_mm": px_mean,
        "resolution_mean_um_per_px": res_mean * 1000.0 if np.isfinite(res_mean) else np.nan,
    }


def compute_hole_pitch(
    hole_centers_world_mm: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """由四个孔心毫米坐标计算水平和竖直方向孔距。"""
    tl, tr, br, bl = [np.asarray(p, dtype=float) for p in hole_centers_world_mm]
    pitch_x = (_dist(tl, tr) + _dist(bl, br)) / 2.0
    pitch_y = (_dist(tl, bl) + _dist(tr, br)) / 2.0
    return pitch_x, pitch_y


def compute_errors(pitch_x_mm: float, pitch_y_mm: float) -> Dict[str, float]:
    """计算孔距与标称 94 mm 的误差。标称值只在这里用于评价。"""
    nominal_x = config.HOLE_PITCH_X_MM
    nominal_y = config.HOLE_PITCH_Y_MM

    abs_error_x = abs(pitch_x_mm - nominal_x)
    abs_error_y = abs(pitch_y_mm - nominal_y)
    rel_error_x = abs_error_x / nominal_x * 100.0 if nominal_x != 0 else 0.0
    rel_error_y = abs_error_y / nominal_y * 100.0 if nominal_y != 0 else 0.0

    return {
        "pitch_x_mm": pitch_x_mm,
        "pitch_y_mm": pitch_y_mm,
        "abs_error_x_mm": abs_error_x,
        "abs_error_y_mm": abs_error_y,
        "rel_error_x_pct": rel_error_x,
        "rel_error_y_pct": rel_error_y,
        "mean_pitch_mm": (pitch_x_mm + pitch_y_mm) / 2.0,
        "mean_abs_error_mm": (abs_error_x + abs_error_y) / 2.0,
    }


def _warped_points_to_image(points_px: np.ndarray, board_homography: np.ndarray) -> np.ndarray:
    """将俯视图坐标反投影回去畸变后的原图坐标。"""
    inv_h = np.linalg.inv(board_homography)
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    image_points = cv2.perspectiveTransform(points, inv_h)
    return image_points.reshape(-1, 2)


def _hole_diameters_world(
    centers_warped_px: List[Tuple[float, float]],
    radii_warped_px: List[float],
    board_homography: np.ndarray,
    image_to_world: np.ndarray,
) -> List[float]:
    """将俯视图圆采样到毫米平面，计算直径。"""
    diameters = []
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)

    for center, radius in zip(centers_warped_px, radii_warped_px):
        if radius <= 0:
            diameters.append(np.nan)
            continue

        cx, cy = center
        warped_samples = np.array(
            [[cx + radius * np.cos(a), cy + radius * np.sin(a)] for a in angles],
            dtype=np.float32,
        )
        warped_center = np.array([[cx, cy]], dtype=np.float32)

        image_center = _warped_points_to_image(warped_center, board_homography)
        image_samples = _warped_points_to_image(warped_samples, board_homography)
        world_center = transform_points_to_world(image_center, image_to_world)[0]
        world_samples = transform_points_to_world(image_samples, image_to_world)

        radii_mm = np.linalg.norm(world_samples - world_center, axis=1)
        diameters.append(float(2.0 * np.mean(radii_mm)))

    return diameters


def run_measurement(
    board_result: DetectionResult,
    hole_result: DetectionResult,
    image_to_world: np.ndarray,
) -> Dict:
    """
    对单张图像执行几何测量。

    Args:
        board_result: 外框检测结果，包含原图角点与 image->warped 单应性
        hole_result:  安装孔检测结果，孔心/半径位于俯视图坐标系
        image_to_world: 去畸变后原图像素 -> 测量平面毫米坐标
    """
    if image_to_world is None:
        out = _nan_measurement()
        out.update({
            "board_detect_success": bool(board_result and board_result.success),
            "holes_detect_success": False,
            "measurement_valid": False,
            "measurement_message": "缺少测量平面单应性",
        })
        return out

    if not board_result.success:
        out = _nan_measurement()
        out.update({
            "board_detect_success": False,
            "holes_detect_success": False,
            "measurement_valid": False,
            "measurement_message": "电路板外框检测失败",
        })
        return out

    corners_px = board_result.data["corners"]
    corners_world = transform_points_to_world(corners_px, image_to_world)
    board_dims = compute_board_dimensions(corners_world)
    resolution = compute_measurement_resolution(
        board_dims, board_result.data["board_size_px"]
    )

    out = {
        "board_detect_success": True,
        **board_dims,
        **resolution,
        "measurement_valid": False,
        "measurement_message": "",
    }

    if not hole_result.success:
        out.update(_nan_measurement())
        out.update(board_dims)
        out.update(resolution)
        out.update({
            "board_detect_success": True,
            "holes_detect_success": False,
            "measurement_valid": False,
            "measurement_message": "安装孔检测失败",
        })
        return out

    board_h = board_result.data["homography"]
    hole_centers_warped = hole_result.data["hole_centers_px"]
    hole_radii_warped = hole_result.data["hole_radii_px"]

    hole_centers_image = _warped_points_to_image(
        np.array(hole_centers_warped, dtype=np.float32), board_h
    )
    hole_centers_world = transform_points_to_world(hole_centers_image, image_to_world)
    hole_centers_world_list = [tuple(map(float, p)) for p in hole_centers_world]

    pitch_x, pitch_y = compute_hole_pitch(hole_centers_world_list)
    errors = compute_errors(pitch_x, pitch_y)
    diameters = _hole_diameters_world(
        hole_centers_warped, hole_radii_warped, board_h, image_to_world
    )

    flat = {}
    for i, (center, diameter) in enumerate(zip(hole_centers_world_list, diameters), start=1):
        flat[f"hole{i}_x_mm"] = center[0]
        flat[f"hole{i}_y_mm"] = center[1]
        flat[f"hole{i}_diameter_mm"] = diameter

    out.update({
        "holes_detect_success": True,
        **errors,
        "hole_diameters_mm": diameters,
        "hole_centers_world_mm": hole_centers_world_list,
        **flat,
        "measurement_valid": True,
        "measurement_message": "OK",
    })
    return out


def summarize_measurements(measurements: List[Dict]) -> Dict[str, float]:
    """对多张图像的测量结果进行统计汇总。"""
    if len(measurements) == 0:
        return {}

    keys = [
        "board_width_mm", "board_height_mm",
        "board_width_error_mm", "board_height_error_mm",
        "resolution_x_mm_per_px", "resolution_y_mm_per_px",
        "resolution_mean_mm_per_px", "resolution_x_px_per_mm",
        "resolution_y_px_per_mm", "resolution_mean_px_per_mm",
        "resolution_mean_um_per_px",
        "pitch_x_mm", "pitch_y_mm",
        "abs_error_x_mm", "abs_error_y_mm",
        "rel_error_x_pct", "rel_error_y_pct",
        "mean_pitch_mm", "mean_abs_error_mm",
    ]
    keys.extend([f"hole{i}_diameter_mm" for i in range(1, 5)])

    summary = {}
    for key in keys:
        values = [
            m[key] for m in measurements
            if key in m and np.isfinite(m.get(key, np.nan))
        ]
        if values:
            arr = np.array(values, dtype=float)
            summary[f"{key}_mean"] = float(np.mean(arr))
            summary[f"{key}_std"] = float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{key}_min"] = float(np.min(arr))
            summary[f"{key}_max"] = float(np.max(arr))

    summary["num_images"] = len(measurements)
    return summary
