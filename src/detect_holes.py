"""
detect_holes.py — 安装孔检测模块

功能：
  在透视矫正后的 PCB 俯视图中，在估计的理论位置附近开辟 ROI，
  在每个 ROI 内独立检测安装孔（圆形），提高检测鲁棒性。

策略：
  不使用全图 HoughCircles（容易误检），
  而是利用已知电路板尺寸和孔间距估计大致位置，
  仅在局部 ROI 内进行圆检测。
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils import DetectionResult


# ============================================================
# 坐标与位置计算
# ============================================================

def get_expected_hole_positions() -> List[Tuple[float, float]]:
    """
    返回四个安装孔的理论位置（单位：mm，相对于电路板左上角）。

    Returns:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        顺序：左上、右上、右下、左下
    """
    return config.EXPECTED_HOLE_POSITIONS_MM


def mm_to_pixel(
    pos_mm: Tuple[float, float],
    board_mm: Tuple[float, float],
    board_px: Tuple[int, int],
) -> Tuple[int, int]:
    """
    将 mm 坐标转换为像素坐标。

    根据电路板物理尺寸和俯视图像素尺寸建立线性映射。

    Args:
        pos_mm:     (x, y) 坐标，单位 mm
        board_mm:   (宽, 高)，单位 mm
        board_px:   (宽, 高)，单位像素

    Returns:
        (x_px, y_px) 像素坐标
    """
    x_mm, y_mm = pos_mm
    bw_mm, bh_mm = board_mm
    bw_px, bh_px = board_px

    # 线性映射
    scale_x = bw_px / bw_mm
    scale_y = bh_px / bh_mm

    x_px = int(x_mm * scale_x)
    y_px = int(y_mm * scale_y)

    return x_px, y_px


def crop_hole_rois(
    image: np.ndarray,
    hole_positions_px: List[Tuple[int, int]],
    roi_half_size_px: int,
) -> List[Tuple[int, int, int, int]]:
    """
    根据孔心预期位置，裁剪出各孔的 ROI 区域。

    每个 ROI 为正方形，边长 = 2 * roi_half_size_px。

    Args:
        image: 输入图像（用于检查边界）
        hole_positions_px: 四个孔心预期像素坐标
        roi_half_size_px: ROI 半边长（像素）

    Returns:
        ROI 列表，每个元素为 (x, y, w, h)，若超出边界则裁剪
    """
    h, w = image.shape[:2]
    rois = []

    for cx, cy in hole_positions_px:
        x1 = max(0, cx - roi_half_size_px)
        y1 = max(0, cy - roi_half_size_px)
        x2 = min(w, cx + roi_half_size_px)
        y2 = min(h, cy + roi_half_size_px)

        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w <= 0 or roi_h <= 0:
            # ROI 完全越界，用默认值
            x1, y1 = 0, 0
            roi_w, roi_h = 10, 10

        rois.append((x1, y1, roi_w, roi_h))

    return rois


# ============================================================
# 圆孔检测
# ============================================================

def detect_hole_in_roi(
    roi_image: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """
    在单个 ROI 内检测圆孔。

    方法：HoughCircles 圆检测（初始版本）。
    TODO: 后续可改为椭圆拟合或轮廓圆度筛选以提高精度。

    Args:
        roi_image: ROI 区域的灰度图像

    Returns:
        (cx, cy, r) 圆心坐标（相对于 ROI）和半径，
        检测失败返回 None
    """
    circles = cv2.HoughCircles(
        roi_image,
        cv2.HOUGH_GRADIENT,
        dp=config.HOUGH_DP,
        minDist=config.HOUGH_MIN_DIST,
        param1=config.HOUGH_PARAM1,
        param2=config.HOUGH_PARAM2,
        minRadius=config.HOUGH_MIN_RADIUS,
        maxRadius=config.HOUGH_MAX_RADIUS,
    )

    if circles is None or len(circles) == 0:
        return None

    # 取检测到的第一个（最显著的）圆
    circle = circles[0, 0]
    cx, cy, r = circle
    return float(cx), float(cy), float(r)


def detect_all_holes(
    warped_image: np.ndarray,
) -> DetectionResult:
    """
    在透视矫正后的电路板图像中检测四个安装孔。

    流程：
      1. 根据电路板物理尺寸和图像尺寸计算四个孔心的预期像素位置。
      2. 在每个预期位置附近开辟 ROI。
      3. 在每个 ROI 内独立进行圆检测。
      4. 汇总检测结果。

    Args:
        warped_image: 透视矫正后的俯视图（BGR 或灰度）

    Returns:
        DetectionResult:
            - success: 是否全部四个孔都检测成功
            - data:
                - "hole_centers_px":  四个孔心像素坐标 [(cx,cy), ...]
                - "hole_radii_px":    四个孔的半径（像素）
                - "individual_success": [bool, bool, bool, bool]
                - "roi_regions":      四个 ROI 区域 [(x,y,w,h), ...]
                - "visualization":    叠加检测结果的可视化图像
            - message: 描述信息
    """
    if warped_image is None:
        return DetectionResult(success=False, message="输入图像为空")

    # 转为灰度图
    if len(warped_image.shape) == 3:
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_image.copy()

    h, w = gray.shape[:2]
    board_px = (w, h)

    # 1. 计算理论孔心像素位置
    expected_positions_mm = get_expected_hole_positions()
    board_mm = (config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM)

    expected_positions_px = [
        mm_to_pixel(pos_mm, board_mm, board_px)
        for pos_mm in expected_positions_mm
    ]

    # 2. 计算 ROI 半边长（像素）
    scale_x = w / config.BOARD_WIDTH_MM
    scale_y = h / config.BOARD_HEIGHT_MM
    scale = (scale_x + scale_y) / 2.0
    roi_half_px = int(config.HOLE_ROI_SIZE_MM * scale / 2)

    # 3. 裁剪 ROI
    rois = crop_hole_rois(gray, expected_positions_px, roi_half_px)

    # 4. 在每个 ROI 内检测圆孔
    hole_centers = []
    hole_radii = []
    individual_success = []
    roi_regions = []

    # 用于可视化的彩色图
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(warped_image.shape) == 2 \
        else warped_image.copy()

    for i, (roi_x, roi_y, roi_w, roi_h) in enumerate(rois):
        roi_image = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        roi_regions.append((roi_x, roi_y, roi_w, roi_h))

        result = detect_hole_in_roi(roi_image)

        if result is not None:
            cx_roi, cy_roi, r = result
            # 转换到全图像素坐标
            cx_full = cx_roi + roi_x
            cy_full = cy_roi + roi_y
            hole_centers.append((cx_full, cy_full))
            hole_radii.append(r)
            individual_success.append(True)

            # 可视化
            cv2.circle(vis, (int(cx_full), int(cy_full)),
                       int(r), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx_full), int(cy_full)),
                       3, (0, 0, 255), -1)
            cv2.putText(vis, f"H{i+1}", (int(cx_full) + 10, int(cy_full) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            hole_centers.append((0.0, 0.0))
            hole_radii.append(0.0)
            individual_success.append(False)
            print(f"  [WARNING] 安装孔 {i+1} 检测失败（ROI: {roi_x},{roi_y} {roi_w}×{roi_h}）")

        # 在可视化图上画出 ROI 方框
        cv2.rectangle(vis, (roi_x, roi_y),
                      (roi_x + roi_w, roi_y + roi_h),
                      (255, 255, 0), 1)

    # 判断整体是否成功（所有孔都检测到）
    all_success = all(individual_success)
    num_detected = sum(individual_success)

    return DetectionResult(
        success=all_success,
        message=f"安装孔检测: {num_detected}/4 成功" if not all_success
                else "四个安装孔全部检测成功",
        data={
            "hole_centers_px": hole_centers,
            "hole_radii_px": hole_radii,
            "individual_success": individual_success,
            "roi_regions": roi_regions,
            "visualization": vis,
            "expected_positions_px": expected_positions_px,
        },
    )
