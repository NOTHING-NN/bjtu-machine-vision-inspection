"""
region_separation.py — 光斑与 PCB 粘连分离

面向 B 类样本（邻近强光斑），当绿色 mask 中 PCB 区域与
桌面反光区域粘连为一个连通域时，利用高光 mask 引导切除
反光连接处，将 PCB 从反光干扰中分离出来。

算法：
  1. 连通域分析，找出贴边界的连通域（粘连候选）
  2. 计算每个贴边连通域内高光 mask 的重叠率
  3. 重叠率 > 阈值 → 判定为粘连：
     膨胀高光 mask → 从 green_mask 中切除重叠区 →
     形态学开运算断开细连接
  4. 安全回退：如果分离后无合法候选，返回原始 mask
"""

from typing import Tuple

import cv2
import numpy as np

from src import config


def separate_highlight_from_mask(
    green_mask: np.ndarray,
    highlight_mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    利用高光 mask 切除 green_mask 中与反光粘连的区域。

    Args:
        green_mask:     PCB 绿色区域二值 mask (255=前景)
        highlight_mask: 高光区域二值 mask (255=高光)
        image_shape:    原图尺寸 (h, w)

    Returns:
        清理后的 green_mask。如果分离后无合法候选，返回原始 mask。
    """
    h, w = image_shape[:2]
    margin = config.BOUNDARY_MARGIN_PX

    # 将 mask 转为 0/1 便于操作
    gm = (green_mask > 0).astype(np.uint8)
    hm = (highlight_mask > 0).astype(np.uint8)

    # 1. 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        gm, connectivity=8
    )

    modified = False

    # 跳过 label 0（背景）
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]

        # 2. 检查是否贴边
        touches_boundary = (
            x <= margin or y <= margin or
            x + bw >= w - margin or y + bh >= h - margin
        )
        if not touches_boundary:
            continue

        # 3. 计算该连通域内高光 mask 的重叠率
        component_mask = (labels == label_id).astype(np.uint8)
        overlap = cv2.bitwise_and(component_mask, hm)
        overlap_ratio = float(overlap.sum()) / max(area, 1)

        if overlap_ratio < config.HIGHLIGHT_OVERLAP_RATIO:
            continue

        # 4. 判定为粘连 — 执行分离
        # 膨胀高光 mask 覆盖过渡带
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.HIGHLIGHT_DILATE_KERNEL_SIZE)
        hm_dilated = cv2.dilate(hm, dilate_kernel, iterations=1)

        # 在该连通域内切除与高光重叠的区域
        cut_region = cv2.bitwise_and(component_mask, hm_dilated)
        gm[cut_region > 0] = 0
        modified = True

    if not modified:
        return green_mask

    # 5. 形态学开运算断开可能残留的细连接
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, config.REGION_SEPARATION_OPEN_KERNEL_SIZE)
    gm = cv2.morphologyEx(gm, cv2.MORPH_OPEN, open_kernel)

    # 6. 填充内部空洞
    contours, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    gm_filled = np.zeros_like(gm)
    cv2.drawContours(gm_filled, contours, -1, 255, cv2.FILLED)

    # 7. 安全回退：检查分离后是否有合法候选
    from src.board_detection.candidate_filter import find_board_contours
    candidates = find_board_contours(gm_filled, image_shape)

    if len(candidates) == 0:
        # 分离过度，返回原始 mask
        print("  [INFO] 区域分离后无合法候选，回退到原始 mask")
        return green_mask

    print(f"  [INFO] 区域分离完成，产生 {len(candidates)} 个候选")
    return gm_filled
