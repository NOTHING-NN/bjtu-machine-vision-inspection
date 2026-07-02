"""
perspective.py — 透视变换

提供角点排序、单应性矩阵计算和透视矫正功能。
"""

from typing import Tuple

import cv2
import numpy as np


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    将四角点按极角排序为：左上 → 右上 → 右下 → 左下。
    """
    if corners.ndim == 3:
        corners = corners.reshape(4, 2)

    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1],
                        corners[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    return corners[sorted_idx].astype(np.float32)


def compute_homography(corners: np.ndarray,
                       board_size_pixels: Tuple[int, int] = None) -> np.ndarray:
    """计算原图到俯视图的单应性变换矩阵。"""
    if board_size_pixels is None:
        tl, tr, br, bl = corners
        wp = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        hp = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        size_px = int(max(wp, hp))
        board_size_pixels = (size_px, size_px)

    dst = np.array([
        [0, 0],
        [board_size_pixels[0] - 1, 0],
        [board_size_pixels[0] - 1, board_size_pixels[1] - 1],
        [0, board_size_pixels[1] - 1],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(corners, dst, cv2.RANSAC, 5.0)
    return H


def warp_board(image: np.ndarray, corners: np.ndarray,
               board_size_pixels: Tuple[int, int] = None) -> np.ndarray:
    """透视变换输出俯视图。"""
    H = compute_homography(corners, board_size_pixels)
    if board_size_pixels is None:
        tl, tr, br, bl = corners
        wp = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        hp = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        board_size_pixels = (int(max(wp, hp)), int(max(wp, hp)))
    return cv2.warpPerspective(image, H, board_size_pixels,
                               flags=cv2.INTER_LINEAR)
