"""
candidate_filter.py — PCB 候选四边形筛选与打分

从 green_mask 中提取连通域，经过面积、贴边、宽高比、中心位置
等多重约束筛选，对合法候选打分排序。
"""

from typing import List, Tuple

import cv2
import numpy as np

from src import config


def _is_near_boundary(contour: np.ndarray,
                      image_shape: Tuple[int, int],
                      margin: int = None) -> bool:
    """检查轮廓任一角点是否贴近图像边界。"""
    if margin is None:
        margin = config.BOUNDARY_MARGIN_PX
    h, w = image_shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    return (x <= margin or y <= margin or
            x + bw >= w - margin or y + bh >= h - margin)


def _check_aspect_ratio(corners: np.ndarray) -> bool:
    """检查四边形宽高比是否接近 1:1。"""
    tl, tr, br, bl = corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    w_avg = (w_top + w_bot) / 2.0
    h_avg = (h_left + h_right) / 2.0
    if min(w_avg, h_avg) < 1.0:
        return False
    ratio = w_avg / h_avg
    return config.BOARD_ASPECT_MIN <= ratio <= config.BOARD_ASPECT_MAX


def _center_inbounds(corners: np.ndarray,
                     image_shape: Tuple[int, int]) -> bool:
    """检查四边形中心是否离图像边界不太近。"""
    h, w = image_shape[:2]
    cx = corners[:, 0].mean()
    cy = corners[:, 1].mean()
    margin = config.BOARD_CENTER_MARGIN_RATIO
    return (w * margin <= cx <= w * (1 - margin) and
            h * margin <= cy <= h * (1 - margin))


def _score_candidate(corners: np.ndarray, area: float,
                     image_area: float) -> float:
    """
    对候选四边形打分。
      方形度（宽高比接近 1.0）— 权重 0.6
      面积合理性（接近 10%）    — 权重 0.4
    """
    tl, tr, br, bl = corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    w_avg = (w_top + w_bot) / 2.0
    h_avg = (h_left + h_right) / 2.0

    ratio = w_avg / max(h_avg, 1.0)
    square_score = 1.0 - min(abs(ratio - 1.0), 0.5) / 0.5

    area_ratio = area / image_area
    ideal = 0.10
    area_score = 1.0 - min(abs(area_ratio - ideal) / 0.10, 1.0)

    return 0.6 * square_score + 0.4 * area_score


def find_board_contours(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> List[Tuple[np.ndarray, float]]:
    """
    从 PCB mask 中查找所有合法候选四边形。

    Args:
        mask: PCB 区域二值图
        image_shape: (h, w)

    Returns:
        [(corners, score), ...] 按 score 降序排列
    """
    h, w = image_shape[:2]
    image_area = h * w
    min_area = image_area * config.BOARD_MIN_AREA_RATIO
    max_area = image_area * config.BOARD_MAX_AREA_RATIO

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        if _is_near_boundary(cnt, image_shape):
            continue

        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)

        best_approx = None
        for eps_ratio in [0.01, 0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
            if len(approx) == 4:
                best_approx = approx
                break

        if best_approx is None:
            rect = cv2.minAreaRect(hull)
            best_approx = np.array(cv2.boxPoints(rect), dtype=np.float32)

        if best_approx.shape[0] != 4:
            continue

        corners = best_approx.reshape(4, 2).astype(np.float32)

        if not _check_aspect_ratio(corners):
            continue

        if not _center_inbounds(corners, image_shape):
            continue

        score = _score_candidate(corners, area, image_area)
        candidates.append((corners, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates
