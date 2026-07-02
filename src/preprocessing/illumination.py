"""
illumination.py — 光照估计与暗区处理

提供大尺度光照场估计和暗区提升功能，
用于光照矫正预处理流程。
"""

import cv2
import numpy as np

from src import config


def compute_illumination_field(v_channel: np.ndarray) -> np.ndarray:
    """
    大尺度高斯模糊估计 V 通道光照场（低频背景亮度）。

    Args:
        v_channel: HSV 的 V 通道 (float32 或 uint8)

    Returns:
        光照场，与输入同尺寸，uint8
    """
    illumination = cv2.GaussianBlur(
        v_channel.astype(np.float32),
        config.LARGE_GAUSSIAN_KERNEL_SIZE,
        config.LARGE_GAUSSIAN_SIGMA,
    )
    return np.clip(illumination, 0, 255).astype(np.uint8)


def boost_shadows(v_channel: np.ndarray,
                  illumination: np.ndarray) -> np.ndarray:
    """
    对极暗区域用光照场做适度提亮。

    Args:
        v_channel:     HSV 的 V 通道 (float32)
        illumination:  光照场 (float32)

    Returns:
        提亮后的 V 通道 (float32)
    """
    shadow_mask = v_channel < config.SHADOW_V_MAX
    if shadow_mask.sum() > 0:
        v_channel[shadow_mask] = np.maximum(
            v_channel[shadow_mask],
            illumination[shadow_mask] * config.SHADOW_BOOST_FACTOR,
        )
    return v_channel
