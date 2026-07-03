"""
hole_detector.py — 安装孔检测模块（俯视图版）

在透视矫正后的俯视图中，根据理论孔位（mm）开 ROI，
在 ROI 内用轮廓圆度 + HoughCircles 综合检测。

优势：
  - 圆心和半径天然在俯视图尺度，无需坐标映射
  - 俯视图为 1:1 正方形，计算简单可靠
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils import DetectionResult


def get_expected_hole_positions() -> List[Tuple[float, float]]:
    """返回四个安装孔的理论位置（mm）。"""
    return config.EXPECTED_HOLE_POSITIONS_MM


def mm_to_pixel(
    pos_mm: Tuple[float, float],
    board_mm: Tuple[float, float],
    board_px: Tuple[int, int],
) -> Tuple[int, int]:
    """将 mm 坐标转换为俯视图像素坐标。"""
    scale_x = board_px[0] / board_mm[0]
    scale_y = board_px[1] / board_mm[1]
    return int(pos_mm[0] * scale_x), int(pos_mm[1] * scale_y)


def _contour_circularity(contour: np.ndarray) -> float:
    """轮廓圆度：4π×area/perimeter²。完美圆=1.0。"""
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return 4.0 * np.pi * area / (peri * peri)


def detect_hole_in_roi(
    roi_gray: np.ndarray,
    min_radius: int = 10,
    max_radius: int = 300,
    expected_center: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    在单个 ROI 灰度图中检测圆孔。

    方法：CLAHE 增强 → 多源二值化 → 轮廓圆度筛选 → HoughCircles 备选。

    Returns:
        (cx, cy, r) 相对于 ROI 的圆心和半径，失败返回 None
    """
    candidates = []
    roi_h, roi_w = roi_gray.shape[:2]
    if expected_center is None:
        expected_center = (roi_w / 2.0, roi_h / 2.0)
    max_center_dist = max(np.hypot(roi_w, roi_h) * 0.5, 1.0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_gray)

    bin_sources = []
    th = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    bin_sources.append(th)
    bin_sources.append(cv2.bitwise_not(th))
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_sources.append(otsu)
    bin_sources.append(cv2.bitwise_not(otsu))

    edges = cv2.Canny(enhanced, 30, 100)
    bin_sources.append(cv2.dilate(edges, None, iterations=2))

    for bin_img in bin_sources:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * min_radius * min_radius * 0.3:
                continue
            if area > np.pi * max_radius * max_radius * 2:
                continue

            circularity = _contour_circularity(cnt)
            if circularity < 0.55:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < min_radius or r > max_radius:
                continue

            area_score = (
                1.0 - abs(area - np.pi * r * r) / max(np.pi * r * r, 1.0)
            )
            center_dist = np.hypot(cx - expected_center[0], cy - expected_center[1])
            position_score = 1.0 - min(center_dist / max_center_dist, 1.0)
            score = (
                0.40 * circularity +
                0.35 * area_score +
                0.25 * position_score
            )
            candidates.append((score, cx, cy, r))

    circles = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_radius,
        param1=40, param2=max(10, int(max_radius / 12)),
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            center_dist = np.hypot(cx - expected_center[0], cy - expected_center[1])
            position_score = 1.0 - min(center_dist / max_center_dist, 1.0)
            candidates.append((0.45 + 0.35 * position_score, cx, cy, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, cx, cy, r = candidates[0]
    return float(cx), float(cy), float(r)


def detect_all_holes(
    warped_gray: np.ndarray,
    board_size_px: Tuple[int, int],
    warped_color: np.ndarray = None,
) -> DetectionResult:
    """
    在透视矫正后的俯视图中检测四个安装孔。

    Args:
        warped_gray:   俯视灰度图
        board_size_px: 俯视图尺寸 (w, h)
        warped_color:  俯视彩色图（可视化用）

    Returns:
        DetectionResult:
            data["hole_centers_px"]:    四个孔心俯视图像素坐标
            data["hole_radii_px"]:      四个孔半径（俯视图像素）
            data["individual_success"]:  [bool, bool, bool, bool]
            data["visualization"]:      叠加检测结果的图
            data["expected_positions_px"]: 理论孔心像素位置
    """
    if warped_gray is None:
        return DetectionResult(success=False, message="俯视图为空")

    h, w = warped_gray.shape[:2]

    board_mm = (config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM)
    expected_positions_px = [
        mm_to_pixel(pos, board_mm, board_size_px)
        for pos in get_expected_hole_positions()
    ]

    scale = (board_size_px[0] / board_mm[0] + board_size_px[1] / board_mm[1]) / 2.0
    roi_half = int(config.HOLE_ROI_SIZE_MM * scale)

    min_r = max(8, int(0.5 * scale))
    max_r = min(int(w / 6), int(3.0 * scale))

    hole_centers = []
    hole_radii = []
    individual_success = []

    vis = warped_color.copy() if warped_color is not None \
        else cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    for i, (ex, ey) in enumerate(expected_positions_px):
        x1 = max(0, ex - roi_half)
        y1 = max(0, ey - roi_half)
        x2 = min(w, ex + roi_half)
        y2 = min(h, ey + roi_half)
        roi_gray = warped_gray[y1:y2, x1:x2]

        expected_in_roi = (ex - x1, ey - y1)
        result = detect_hole_in_roi(roi_gray, min_r, max_r, expected_in_roi)

        if result is not None:
            cx_roi, cy_roi, r = result
            cx_full = cx_roi + x1
            cy_full = cy_roi + y1
            hole_centers.append((cx_full, cy_full))
            hole_radii.append(r)
            individual_success.append(True)

            cv2.circle(vis, (int(cx_full), int(cy_full)), int(r), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx_full), int(cy_full)), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"H{i+1}", (int(cx_full) + 10, int(cy_full) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            hole_centers.append((0.0, 0.0))
            hole_radii.append(0.0)
            individual_success.append(False)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.drawMarker(vis, (ex, ey), (255, 255, 0),
                       cv2.MARKER_CROSS, 8, 1)

    num_ok = sum(individual_success)

    return DetectionResult(
        success=(num_ok == 4),
        message=f"安装孔检测: {num_ok}/4 成功" if num_ok < 4
                else "四个安装孔全部检测成功",
        data={
            "hole_centers_px": hole_centers,
            "hole_radii_px": hole_radii,
            "individual_success": individual_success,
            "visualization": vis,
            "expected_positions_px": expected_positions_px,
        },
    )
