"""
highlight.py — 高光检测与抑制

检测白色反光区域（高 V + 低 S），并对反光区进行 V 通道压制。
用于光照矫正预处理流程，以及区域分离时的高光 mask 生成。
"""

import cv2
import numpy as np

from src import config


def detect_highlights(h_channel: np.ndarray,
                      s_channel: np.ndarray,
                      v_channel: np.ndarray) -> np.ndarray:
    """
    检测白色反光区域：V 高 + S 低（颜色不饱和），
    膨胀 mask 覆盖反光边缘过渡区。

    Args:
        h_channel: HSV 的 H 通道 (float32 或 uint8)
        s_channel: HSV 的 S 通道
        v_channel: HSV 的 V 通道

    Returns:
        高光二值 mask，uint8 (255=高光区域)
    """
    highlight = (
        (v_channel > config.HIGHLIGHT_V_THRESH) &
        (s_channel < config.HIGHLIGHT_S_MAX)
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    highlight = cv2.dilate(highlight, kernel, iterations=2)

    return highlight


def suppress_highlights(v_channel: np.ndarray,
                        highlight_mask: np.ndarray,
                        illumination: np.ndarray) -> np.ndarray:
    """
    将高光区域的 V 值强力压暗到光照场的指定比例。

    Args:
        v_channel:      HSV 的 V 通道 (float32)
        highlight_mask: 高光二值 mask (uint8, 0/255 或 0/1)
        illumination:   光照场 (float32)

    Returns:
        压制高光后的 V 通道 (float32)
    """
    if highlight_mask.sum() > 0:
        mask_bool = highlight_mask > 0
        v_channel[mask_bool] = (
            illumination[mask_bool] * config.HIGHLIGHT_SUPPRESS_FACTOR
        )
    return v_channel
