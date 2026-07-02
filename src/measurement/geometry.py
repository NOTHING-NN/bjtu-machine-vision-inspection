"""
geometry.py — 几何尺寸测量与误差计算

提供俯视图下的像素-mm转换、孔距计算、误差评估和
多图测量汇总统计。

注意：
  由于使用 100×100 mm 电路板外框作为透视矫正尺度基准，
  电路板长宽不作为独立验证指标。
  重点使用 94×94 mm 安装孔中心距作为独立验证指标。
"""

from typing import Dict, List, Tuple

import numpy as np

from src import config
from src.utils import DetectionResult


def pixel_to_mm_after_warp(board_size_px: Tuple[int, int]) -> float:
    """
    计算透视矫正后图像的像素到 mm 的转换比例。

    透视矫正时将电路板外框（100×100 mm）映射到 board_size_px，
    因此：scale = BOARD_WIDTH_MM / board_width_px
    """
    bw_px, bh_px = board_size_px
    scale_x = config.BOARD_WIDTH_MM / bw_px
    scale_y = config.BOARD_HEIGHT_MM / bh_px
    return (scale_x + scale_y) / 2.0


def compute_hole_pitch(
    hole_centers_px: List[Tuple[float, float]],
    scale_mm_per_px: float,
) -> Tuple[float, float]:
    """
    计算四个安装孔之间的 X 方向和 Y 方向间距。

    孔排序：左上(0), 右上(1), 右下(2), 左下(3)
    X 间距：孔 0→1 和孔 3→2 的平均水平距离
    Y 间距：孔 0→3 和孔 1→2 的平均垂直距离

    Returns:
        (pitch_x_mm, pitch_y_mm)
    """
    tl, tr, br, bl = hole_centers_px

    dx_top = abs(tr[0] - tl[0])
    dx_bottom = abs(br[0] - bl[0])
    pitch_x_px = (dx_top + dx_bottom) / 2.0

    dy_left = abs(bl[1] - tl[1])
    dy_right = abs(br[1] - tr[1])
    pitch_y_px = (dy_left + dy_right) / 2.0

    pitch_x_mm = pitch_x_px * scale_mm_per_px
    pitch_y_mm = pitch_y_px * scale_mm_per_px

    return pitch_x_mm, pitch_y_mm


def compute_hole_diameter(
    hole_radii_px: List[float],
    scale_mm_per_px: float,
) -> List[float]:
    """计算四个安装孔的直径（mm）。"""
    return [2.0 * r * scale_mm_per_px for r in hole_radii_px]


def compute_errors(pitch_x_mm: float, pitch_y_mm: float) -> Dict[str, float]:
    """
    计算与标称值 (94.0 mm) 之间的误差。

    Returns:
        dict 包含绝对误差和相对误差
    """
    nominal_x = config.HOLE_PITCH_X_MM
    nominal_y = config.HOLE_PITCH_Y_MM

    abs_error_x = abs(pitch_x_mm - nominal_x)
    abs_error_y = abs(pitch_y_mm - nominal_y)
    rel_error_x = abs_error_x / nominal_x * 100.0 if nominal_x != 0 else 0.0
    rel_error_y = abs_error_y / nominal_y * 100.0 if nominal_y != 0 else 0.0

    mean_pitch_mm = (pitch_x_mm + pitch_y_mm) / 2.0
    mean_abs_error = (abs_error_x + abs_error_y) / 2.0

    return {
        "pitch_x_mm": pitch_x_mm,
        "pitch_y_mm": pitch_y_mm,
        "abs_error_x_mm": abs_error_x,
        "abs_error_y_mm": abs_error_y,
        "rel_error_x_pct": rel_error_x,
        "rel_error_y_pct": rel_error_y,
        "mean_pitch_mm": mean_pitch_mm,
        "mean_abs_error_mm": mean_abs_error,
    }


def summarize_measurements(measurements: List[Dict]) -> Dict[str, float]:
    """
    对多张图像的测量结果进行统计汇总。

    Returns:
        汇总统计 dict，包含均值、std、min、max 和 num_images
    """
    if len(measurements) == 0:
        return {}

    keys = ["pitch_x_mm", "pitch_y_mm", "abs_error_x_mm", "abs_error_y_mm",
            "rel_error_x_pct", "rel_error_y_pct", "mean_pitch_mm", "mean_abs_error_mm"]

    summary = {}
    for key in keys:
        values = [m[key] for m in measurements if key in m]
        if values:
            arr = np.array(values)
            summary[f"{key}_mean"] = float(np.mean(arr))
            summary[f"{key}_std"] = float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{key}_min"] = float(np.min(arr))
            summary[f"{key}_max"] = float(np.max(arr))

    summary["num_images"] = len(measurements)
    return summary


def run_measurement(
    hole_result: DetectionResult,
    board_size_px: Tuple[int, int],
) -> Dict:
    """
    对单张图像的检测结果执行完整几何测量。
    """
    if not hole_result.success:
        return {
            "board_detect_success": True,
            "holes_detect_success": False,
            "pitch_x_mm": np.nan,
            "pitch_y_mm": np.nan,
            "abs_error_x_mm": np.nan,
            "abs_error_y_mm": np.nan,
            "rel_error_x_pct": np.nan,
            "rel_error_y_pct": np.nan,
            "mean_pitch_mm": np.nan,
            "mean_abs_error_mm": np.nan,
            "hole_diameters_mm": [np.nan] * 4,
        }

    scale = pixel_to_mm_after_warp(board_size_px)
    hole_centers = hole_result.data["hole_centers_px"]
    pitch_x, pitch_y = compute_hole_pitch(hole_centers, scale)
    errors = compute_errors(pitch_x, pitch_y)

    hole_radii = hole_result.data["hole_radii_px"]
    diameters = compute_hole_diameter(hole_radii, scale)

    return {
        "board_detect_success": True,
        "holes_detect_success": hole_result.success,
        **errors,
        "hole_diameters_mm": diameters,
        "scale_mm_per_px": scale,
    }
