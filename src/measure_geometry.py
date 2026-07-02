"""
measure_geometry.py — 几何尺寸测量与误差计算模块

功能：
  1. 根据透视矫正后的图像计算像素与 mm 的比例关系
  2. 计算安装孔间距（核心验证指标）
  3. 计算孔径
  4. 计算与标称值的误差
  5. 汇总多张图像的测量统计

注意：
  由于使用 100×100 mm 电路板外框作为透视矫正尺度基准，
  电路板长宽不作为独立验证指标。
  重点使用 94×94 mm 安装孔中心距作为独立验证指标。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.utils import DetectionResult


# ============================================================
# 像素到 mm 转换
# ============================================================

def pixel_to_mm_after_warp(
    board_size_px: Tuple[int, int],
) -> float:
    """
    计算透视矫正后图像的像素到 mm 的转换比例。

    透视矫正时将电路板外框（100×100 mm）映射到 board_size_px，
    因此：scale = BOARD_WIDTH_MM / board_width_px

    Args:
        board_size_px: 俯视图尺寸 (width, height) 像素

    Returns:
        scale: 每个像素对应的 mm 数 (mm/px)
    """
    bw_px, bh_px = board_size_px
    # 使用宽方向的比例（宽高比例不一定完全相等，取平均值更稳健）
    scale_x = config.BOARD_WIDTH_MM / bw_px
    scale_y = config.BOARD_HEIGHT_MM / bh_px
    scale = (scale_x + scale_y) / 2.0
    return scale


# ============================================================
# 间距计算
# ============================================================

def compute_hole_pitch(
    hole_centers_px: List[Tuple[float, float]],
    scale_mm_per_px: float,
) -> Tuple[float, float]:
    """
    计算四个安装孔之间的 X 方向和 Y 方向间距。

    孔排序：左上(0), 右上(1), 右下(2), 左下(3)

    X 间距：孔 0→1 和孔 3→2 的平均水平距离
    Y 间距：孔 0→3 和孔 1→2 的平均垂直距离

    Args:
        hole_centers_px: 四个孔心的像素坐标 [(cx,cy), ...]
        scale_mm_per_px: 像素到 mm 的转换比例

    Returns:
        (pitch_x_mm, pitch_y_mm) 孔间距，单位 mm
    """
    tl, tr, br, bl = hole_centers_px

    # X 方向：上边 (0→1) 和下边 (3→2) 的水平距离
    dx_top = abs(tr[0] - tl[0])
    dx_bottom = abs(br[0] - bl[0])
    pitch_x_px = (dx_top + dx_bottom) / 2.0

    # Y 方向：左边 (0→3) 和右边 (1→2) 的垂直距离
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
    """
    计算四个安装孔的直径（mm）。

    Args:
        hole_radii_px: 四个孔的半径（像素）
        scale_mm_per_px: 像素到 mm 转换比例

    Returns:
        四个孔的直径列表 [d1, d2, d3, d4]，单位 mm
    """
    diameters = [2.0 * r * scale_mm_per_px for r in hole_radii_px]
    return diameters


# ============================================================
# 误差计算
# ============================================================

def compute_errors(
    pitch_x_mm: float,
    pitch_y_mm: float,
) -> Dict[str, float]:
    """
    计算与标称值 (94.0 mm) 之间的误差。

    Args:
        pitch_x_mm: 实测 X 方向孔间距 (mm)
        pitch_y_mm: 实测 Y 方向孔间距 (mm)

    Returns:
        dict，包含绝对误差和相对误差
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


# ============================================================
# 汇总统计
# ============================================================

def summarize_measurements(
    measurements: List[Dict],
) -> Dict[str, float]:
    """
    对多张图像的测量结果进行统计汇总。

    Args:
        measurements: 测量结果 dict 列表，每个元素来自 compute_errors

    Returns:
        汇总统计 dict，包含均值、标准差、最大误差等
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


# ============================================================
# 完整测量流程
# ============================================================

def run_measurement(
    hole_result: DetectionResult,
    board_size_px: Tuple[int, int],
) -> Dict:
    """
    对单张图像的检测结果执行完整几何测量。

    Args:
        hole_result: 安装孔检测结果 (DetectionResult)
        board_size_px: 俯视图像素尺寸 (w, h)

    Returns:
        包含所有测量指标的 dict
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

    # 计算像素到 mm 比例
    scale = pixel_to_mm_after_warp(board_size_px)

    # 计算孔间距
    hole_centers = hole_result.data["hole_centers_px"]
    pitch_x, pitch_y = compute_hole_pitch(hole_centers, scale)

    # 计算误差
    errors = compute_errors(pitch_x, pitch_y)

    # 计算孔径
    hole_radii = hole_result.data["hole_radii_px"]
    diameters = compute_hole_diameter(hole_radii, scale)

    return {
        "board_detect_success": True,
        "holes_detect_success": hole_result.success,
        **errors,
        "hole_diameters_mm": diameters,
        "scale_mm_per_px": scale,
    }
