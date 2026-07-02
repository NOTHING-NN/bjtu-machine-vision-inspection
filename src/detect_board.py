"""
detect_board.py — 电路板外框检测模块

功能：
  从预处理后的二值图或边缘图中检测电路板外框（四边形），
  提取四角点并进行透视矫正，输出俯视图。

普通组和改进组共用此模块，确保对比实验的公平性。
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils import DetectionResult


def find_board_contour(
    binary_or_edge_image: np.ndarray,
    min_area_ratio: float = None,
) -> Optional[np.ndarray]:
    """
    从二值图或边缘图中寻找电路板外框轮廓。

    策略：
      1. findContours 提取所有轮廓。
      2. 按面积降序排序。
      3. 从大到小遍历轮廓，找到第一个近似四边形。

    Args:
        binary_or_edge_image: 二值图或边缘图
        min_area_ratio: 最小面积占比（相对图像总面积），
                        默认为 config.BOARD_MIN_AREA_RATIO

    Returns:
        面积最大的候选四边形轮廓点集 (N, 1, 2)，未找到返回 None
    """
    if min_area_ratio is None:
        min_area_ratio = config.BOARD_MIN_AREA_RATIO

    h, w = binary_or_edge_image.shape[:2]
    image_area = h * w
    min_area = image_area * min_area_ratio

    # 查找轮廓（仅外轮廓）
    contours, hierarchy = cv2.findContours(
        binary_or_edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        print("  [WARNING] 未找到任何轮廓")
        return None

    # 按面积降序排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 多边形逼近
        peri = cv2.arcLength(cnt, True)
        epsilon = config.BOARD_EPSILON_RATIO * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 如果逼近结果为四边形，则认为是候选电路板外框
        if len(approx) == 4:
            return approx

    # 如果没找到四边形，返回面积最大的轮廓（后续可进一步处理）
    print("  [WARNING] 未找到四边形轮廓，尝试使用最大轮廓")
    return contours[0] if len(contours) > 0 else None


def approximate_board_corners(contour: np.ndarray) -> np.ndarray:
    """
    从轮廓中逼近四角点。

    Args:
        contour: 轮廓点集

    Returns:
        四角点坐标 (4, 2)，形状为 4×2
    """
    peri = cv2.arcLength(contour, True)
    epsilon = config.BOARD_EPSILON_RATIO * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 如果逼近结果不是四边形，则取凸包后再逼近
    if len(approx) != 4:
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, epsilon, True)

    # reshape 为 (4, 2)
    if approx.shape[1] == 1:
        approx = approx.reshape(-1, 2)

    return approx


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    将四角点排序为：左上 → 右上 → 右下 → 左下。

    排序方法：
      1. 按 y 坐标分为上两点和下两点。
      2. 上两点中 x 较小的为左上。
      3. 下两点中 x 较小的为左下。

    Args:
        corners: 四角点坐标数组 (4, 2) 或 (4, 1, 2)

    Returns:
        排序后的四角点 (4, 2)，顺序：TL, TR, BR, BL
    """
    # 统一为 (4, 2)
    if corners.ndim == 3:
        corners = corners.reshape(4, 2)

    # 按 y 坐标排序
    y_sorted = corners[np.argsort(corners[:, 1])]

    # 前两个（y 较小）为上排
    top = y_sorted[:2]
    # 后两个（y 较大）为下排
    bottom = y_sorted[2:]

    # 上排中 x 较小的为左上
    tl = top[np.argmin(top[:, 0])]
    tr = top[np.argmax(top[:, 0])]

    # 下排中 x 较小的为左下
    bl = bottom[np.argmin(bottom[:, 0])]
    br = bottom[np.argmax(bottom[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography(corners: np.ndarray,
                       board_size_pixels: Tuple[int, int] = None,
                       ) -> np.ndarray:
    """
    计算从原图到俯视图的单应性变换矩阵。

    目标：将电路板四角点透视变换到 (board_width_px × board_height_px) 的俯视图。

    Args:
        corners: 排序后的四角点 (4, 2)，顺序：TL, TR, BR, BL
        board_size_pixels: 目标俯视图尺寸 (width, height)，
                           默认根据 BOARD_WIDTH_MM/BOARD_HEIGHT_MM 等比缩放

    Returns:
        3×3 透视变换矩阵 H
    """
    tl, tr, br, bl = corners

    if board_size_pixels is None:
        # 根据实际角点间距来确定像素尺寸，保持 1:1 宽高比
        width_px = int(
            (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        )
        height_px = int(
            (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        )
        size_px = max(width_px, height_px)
        board_size_pixels = (size_px, size_px)

    dst = np.array([
        [0, 0],
        [board_size_pixels[0] - 1, 0],
        [board_size_pixels[0] - 1, board_size_pixels[1] - 1],
        [0, board_size_pixels[1] - 1],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(corners, dst, cv2.RANSAC, 5.0)
    return H


def warp_board(image: np.ndarray,
               corners: np.ndarray,
               board_size_pixels: Tuple[int, int] = None,
               ) -> np.ndarray:
    """
    对电路板图像进行透视变换，输出俯视图。

    Args:
        image: 输入图像（BGR 或灰度）
        corners: 排序后的四角点 (4, 2)：TL, TR, BR, BL
        board_size_pixels: 目标输出尺寸 (w, h)

    Returns:
        矫正后的俯视图
    """
    H = compute_homography(corners, board_size_pixels)

    if board_size_pixels is None:
        tl, tr, br, bl = corners
        width_px = int(
            (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        )
        height_px = int(
            (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        )
        size_px = max(width_px, height_px)
        board_size_pixels = (size_px, size_px)

    warped = cv2.warpPerspective(
        image, H, board_size_pixels,
        flags=cv2.INTER_LINEAR,
    )
    return warped


def detect_board(
    image: np.ndarray,
    binary_image: np.ndarray,
    edge_image: np.ndarray,
) -> DetectionResult:
    """
    完整的电路板外框检测流程。

    1. 从二值图/边缘图寻找四边形轮廓
    2. 逼近四角点
    3. 排序角点
    4. 计算透视变换并输出俯视图

    Args:
        image: 畸变校正后的 BGR 原图
        binary_image: 二值图
        edge_image: 边缘图

    Returns:
        DetectionResult:
            - success: 是否成功检测到电路板外框
            - data:
                - "corners":       排序后的四角点像素坐标 (4, 2)
                - "warped_image":  透视矫正后的俯视图（BGR 或灰度）
                - "warped_binary": 透视矫正后的二值图
                - "homography":    透视变换矩阵 3×3
                - "board_size_px": 俯视图尺寸 (w, h)
            - message: 描述信息
    """
    # 优先使用二值图寻找轮廓，失败则用边缘图
    contour = find_board_contour(binary_image)

    if contour is None:
        contour = find_board_contour(edge_image)

    if contour is None:
        return DetectionResult(
            success=False,
            message="无法检测到电路板外框轮廓",
        )

    # 逼近四角点
    corners_raw = approximate_board_corners(contour)

    if len(corners_raw) != 4:
        return DetectionResult(
            success=False,
            message=f"无法提取四角点，当前检测到 {len(corners_raw)} 个角点",
            data={"corners": corners_raw},
        )

    # 排序角点
    corners = order_corners(corners_raw)

    # 透视变换
    warped_image = warp_board(image, corners)
    warped_binary = warp_board(binary_image, corners)

    return DetectionResult(
        success=True,
        message="电路板外框检测成功",
        data={
            "corners": corners,
            "warped_image": warped_image,
            "warped_binary": warped_binary,
            "homography": compute_homography(corners),
            "board_size_px": warped_image.shape[:2][::-1],  # (w, h)
        },
    )
