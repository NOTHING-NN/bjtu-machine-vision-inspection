"""强光斑粘连场景下的 PCB 四边形恢复。"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def _foreground_intervals(row: np.ndarray) -> List[Tuple[int, int]]:
    xs = np.flatnonzero(row > 0)
    if xs.size == 0:
        return []

    gaps = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[gaps + 1]]
    ends = np.r_[xs[gaps], xs[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _split_stable_runs(rows: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
    if not rows:
        return []

    runs = [[rows[0]]]
    for item in rows[1:]:
        prev = runs[-1][-1]
        y, x1, x2 = item
        py, px1, px2 = prev
        width = max(px2 - px1, 1)
        jump_limit = max(45, int(width * 0.20))
        if y != py + 1 or abs(x1 - px1) > jump_limit or abs(x2 - px2) > jump_limit:
            runs.append([item])
        else:
            runs[-1].append(item)
    return runs


def recover_quad_from_mask(
    mask: np.ndarray,
    min_run_height_ratio: float = 0.20,
    min_interval_width_ratio: float = 0.08,
) -> Optional[Tuple[np.ndarray, float, dict]]:
    """从粘连或残缺 mask 中恢复 PCB 外框四边形。"""
    if mask is None:
        return None

    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )

    margin_x = max(8, int(w * 0.01))
    margin_y = max(8, int(h * 0.01))
    min_width = max(20, int(w * min_interval_width_ratio))

    row_items: List[Tuple[int, int, int]] = []
    for y in range(margin_y, h - margin_y):
        intervals = _foreground_intervals(binary[y])
        intervals = [
            (x1, x2) for x1, x2 in intervals
            if x2 - x1 >= min_width and x1 > margin_x and x2 < w - margin_x
        ]
        if not intervals:
            continue
        x1, x2 = max(intervals, key=lambda p: p[1] - p[0])
        row_items.append((y, x1, x2))

    runs = _split_stable_runs(row_items)
    min_run_height = int(h * min_run_height_ratio)
    runs = [r for r in runs if len(r) >= min_run_height]
    if not runs:
        return None

    def run_score(run: List[Tuple[int, int, int]]) -> float:
        ys = np.array([r[0] for r in run], dtype=np.float32)
        widths = np.array([r[2] - r[1] for r in run], dtype=np.float32)
        stability = 1.0 / (1.0 + float(np.std(widths)) / max(float(np.mean(widths)), 1.0))
        return float(len(run)) * float(np.mean(widths)) * stability

    best_run = max(runs, key=run_score)
    ys = np.array([r[0] for r in best_run], dtype=np.float32)
    left_x = np.array([r[1] for r in best_run], dtype=np.float32)
    right_x = np.array([r[2] for r in best_run], dtype=np.float32)

    # 去除首尾少量不稳定行。
    lo = int(len(best_run) * 0.03)
    hi = max(lo + 2, int(len(best_run) * 0.97))
    fit_y = ys[lo:hi]
    fit_l = left_x[lo:hi]
    fit_r = right_x[lo:hi]

    if fit_y.size < 2:
        return None

    left_coef = np.polyfit(fit_y, fit_l, 1)
    right_coef = np.polyfit(fit_y, fit_r, 1)
    y_top = float(np.min(ys))
    y_bottom = float(np.max(ys))

    corners = np.array([
        [np.polyval(left_coef, y_top), y_top],
        [np.polyval(right_coef, y_top), y_top],
        [np.polyval(right_coef, y_bottom), y_bottom],
        [np.polyval(left_coef, y_bottom), y_bottom],
    ], dtype=np.float32)

    if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= w):
        return None
    if np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= h):
        return None
    if abs(cv2.contourArea(corners)) < h * w * 0.01:
        return None

    poly_mask = np.zeros_like(binary)
    cv2.fillConvexPoly(poly_mask, corners.astype(np.int32), 255)
    fill_ratio = cv2.countNonZero(cv2.bitwise_and(binary, poly_mask)) / max(
        cv2.countNonZero(poly_mask), 1
    )

    score = float(fill_ratio * run_score(best_run))
    diagnostics = {
        # 诊断标签 "row_interval_quad"：
        # 指“逐行扫描前景区间+线性拟合左右边界”的四边形恢复策略——
        # 对 mask 每行取前景最大水平区间，合并为稳定行段后，
        # 用 polyfit 拟合左右边界直线，再由行段首尾 y 坐标构造四角点。
        "recovery": "row_interval_quad",
        "run_height": int(len(best_run)),
        "fill_ratio": float(fill_ratio),
        "num_runs": int(len(runs)),
    }
    return corners, score, diagnostics
