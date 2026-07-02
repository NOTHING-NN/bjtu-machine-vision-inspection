"""
board_detector.py — 电路板外框检测

完整检测流程：
  1. 获取 PCB 绿色 mask（默认调 segment_pcb_green，可外部传入 mask_override）
  2. 从 mask 提取候选四边形（严格筛选 + 打分）
  3. 选最佳候选 → 透视矫正

普通组和实验组共用此模块，确保对比实验的公平性。
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils import DetectionResult
from src.preprocessing.masks import segment_pcb_green
from src.board_detection.candidate_filter import find_board_contours
from src.board_detection.perspective import order_corners, compute_homography, warp_board


def detect_board(
    image: np.ndarray,
    mask_override: np.ndarray = None,
    candidates_override: List[Tuple[np.ndarray, float]] = None,
) -> DetectionResult:
    """
    完整的电路板外框检测流程（HSV 颜色分割版）。

    Args:
        image:               畸变校正后的 BGR 原图
        mask_override:       外部传入的 PCB mask（如果不为 None，替代默认的 segment_pcb_green）
        candidates_override: 外部传入的候选列表（通常为 None，由内部从 mask 查找）

    Returns:
        DetectionResult:
            data["mask"]:         PCB 绿色 mask
            data["corners"]:      排序后的四角点 (4, 2)
            data["warped_image"]: 俯视图（BGR）
            data["warped_gray"]:  俯视灰度图
            data["warped_binary"]:俯视二值图
            data["warped_edge"]:  俯视边缘图
            data["board_size_px"]:俯视图尺寸 (w, h)
            data["homography"]:   单应性矩阵
            data["candidates"]:   所有候选四边形（可视化用）
            data["score"]:        最佳候选得分
    """
    if image is None:
        return DetectionResult(success=False, message="输入图像为空")

    h, w = image.shape[:2]

    # 1. 获取 PCB mask
    if mask_override is not None:
        mask = mask_override
    else:
        mask = segment_pcb_green(image)

    # 2. 从 mask 找合法候选四边形
    if candidates_override is not None:
        candidates = candidates_override
    else:
        candidates = find_board_contours(mask, (h, w))

    if len(candidates) == 0:
        return DetectionResult(
            success=False,
            message="未找到合法 PCB 候选四边形",
            data={"mask": mask},
        )

    # 3. 选最佳候选
    best_corners, best_score = candidates[0]

    # 4. 排序角点
    corners = order_corners(best_corners)

    # 5. 透视变换
    warped_image = warp_board(image, corners)
    warped_mask = warp_board(mask, corners)

    # 灰度 & 二值 & 边缘俯视图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    warped_gray = warp_board(gray, corners)
    warped_binary = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, config.ADAPTIVE_THRESH_BLOCK_SIZE,
        config.ADAPTIVE_THRESH_C,
    )
    warped_edge = cv2.Canny(warped_gray, config.CANNY_LOW_THRESH,
                            config.CANNY_HIGH_THRESH)

    return DetectionResult(
        success=True,
        message=f"电路板外框检测成功 (score={best_score:.3f})",
        data={
            "mask": mask,
            "corners": corners,
            "warped_image": warped_image,
            "warped_mask": warped_mask,
            "warped_gray": warped_gray,
            "warped_binary": warped_binary,
            "warped_edge": warped_edge,
            "homography": compute_homography(corners),
            "board_size_px": warped_image.shape[:2][::-1],
            "candidates": candidates,
            "score": best_score,
        },
    )
