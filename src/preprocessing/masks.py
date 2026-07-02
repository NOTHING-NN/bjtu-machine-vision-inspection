"""
masks.py — 各类 mask 生成与后处理

提供：
  - PCB 绿色区域 mask（HSV 颜色分割）
  - 高光区域 mask
  - 通用 mask 后处理
"""

import cv2
import numpy as np

from src import config


def segment_pcb_green(image: np.ndarray) -> np.ndarray:
    """
    用 HSV 绿色范围粗分割 PCB 基板区域。

    Args:
        image: BGR 彩色图像

    Returns:
        二值 mask（PCB 区域=255，背景=0）
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([config.HSV_GREEN_H_LOW,
                      config.HSV_GREEN_S_LOW,
                      config.HSV_GREEN_V_LOW], dtype=np.uint8)
    upper = np.array([config.HSV_GREEN_H_HIGH, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # 形态学闭运算：填充丝印文字/焊盘造成的小洞
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.MORPH_CLOSE_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 填充 mask 内部可能残留的细小空洞
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, contours, -1, 255, cv2.FILLED)

    return mask_filled


def compute_highlight_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    从 HSV 图像中提取高光区域二值 mask。

    判定条件：V > HIGHLIGHT_V_THRESH 且 S < HIGHLIGHT_S_MAX，
    膨胀覆盖过渡区。

    Args:
        hsv_image: HSV 图像 (uint8, float32 均可)

    Returns:
        高光区域 mask (uint8, 255=高光)
    """
    h = hsv_image[:, :, 0].astype(np.float32)
    s = hsv_image[:, :, 1].astype(np.float32)
    v = hsv_image[:, :, 2].astype(np.float32)

    highlight = (
        (v > config.HIGHLIGHT_V_THRESH) &
        (s < config.HIGHLIGHT_S_MAX)
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    highlight = cv2.dilate(highlight, kernel, iterations=2)

    return highlight


def postprocess_mask(mask: np.ndarray,
                     open_kernel: tuple = (3, 3),
                     fill_holes: bool = True) -> np.ndarray:
    """
    通用 mask 后处理：去噪（开运算）、可选空洞填充。

    Args:
        mask:       输入二值 mask
        open_kernel: 开运算核大小
        fill_holes:  是否填充连通域内部空洞

    Returns:
        后处理后的 mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if fill_holes:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        mask = filled

    return mask
