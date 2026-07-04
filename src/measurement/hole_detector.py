"""
hole_detector.py — 安装孔检测模块（无 PCB 尺寸先验版）

在透视矫正后的 PCB 俯视图中检测四个圆形暗孔。不使用 PCB 标称尺寸、
孔距、孔边距或理论孔位开 ROI。
"""

from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils import DetectionResult


def _contour_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _order_holes(centers: List[Tuple[float, float]]) -> List[int]:
    """按左上、右上、右下、左下排序，仅用于结果输出。"""
    pts = np.asarray(centers, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return list(np.argsort(angles))


def _deduplicate_candidates(candidates: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """合并重复圆候选。候选格式：(score, cx, cy, r)。"""
    merged: List[Tuple[float, float, float, float]] = []
    for cand in sorted(candidates, key=lambda x: x[0], reverse=True):
        _, cx, cy, r = cand
        duplicate = False
        for _, mx, my, mr in merged:
            if np.hypot(cx - mx, cy - my) < max(r, mr) * 0.65:
                duplicate = True
                break
        if not duplicate:
            merged.append(cand)
    return merged


def _select_four_holes(candidates: List[Tuple[float, float, float, float]],
                       image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, float]]:
    """
    从候选中选择四个圆孔。

    使用安装孔的拓扑语义：四个孔分别落在板的四个象限。这里不使用已知孔距、
    孔边距、孔径或 PCB 标称尺寸。
    """
    if len(candidates) < 4:
        return candidates

    h, w = image_shape[:2]
    corners = [
        (0.0, 0.0),
        (float(w - 1), 0.0),
        (float(w - 1), float(h - 1)),
        (0.0, float(h - 1)),
    ]
    diag = max(float(np.hypot(w, h)), 1.0)
    quadrant_candidates = [[] for _ in range(4)]

    for cand in candidates:
        score, cx, cy, _ = cand
        if cx < w / 2 and cy < h / 2:
            q = 0
        elif cx >= w / 2 and cy < h / 2:
            q = 1
        elif cx >= w / 2 and cy >= h / 2:
            q = 2
        else:
            q = 3

        corner_dist = np.hypot(cx - corners[q][0], cy - corners[q][1])
        corner_score = 1.0 - min(corner_dist / (diag * 0.5), 1.0)
        combined = 0.72 * score + 0.28 * corner_score
        quadrant_candidates[q].append((combined, cand))

    selected = []
    for q_items in quadrant_candidates:
        if not q_items:
            break
        q_items.sort(key=lambda x: x[0], reverse=True)
        selected.append(q_items[0][1])

    if len(selected) == 4:
        return selected

    image_area = max(float(h * w), 1.0)
    pool = candidates[: min(16, len(candidates))]
    best_combo = None
    best_score = -1.0

    for combo in combinations(pool, 4):
        pts = np.array([[c[1], c[2]] for c in combo], dtype=np.float32)
        hull_area = abs(cv2.contourArea(cv2.convexHull(pts)))
        spread_score = min(hull_area / (image_area * 0.45), 1.0)
        quality_score = float(np.mean([c[0] for c in combo]))
        score = 0.65 * quality_score + 0.35 * spread_score
        if score > best_score:
            best_score = score
            best_combo = combo

    return list(best_combo) if best_combo is not None else candidates[:4]


def detect_hole_in_roi(
    roi_gray: np.ndarray,
    min_radius: int = 10,
    max_radius: int = 300,
    expected_center: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    兼容旧接口：在 ROI 内检测最可信圆孔。

    expected_center 参数保留但不用于已知孔位先验，只作为旧调用兼容。
    """
    result = _find_hole_candidates(roi_gray, min_radius, max_radius)
    if not result:
        return None
    _, cx, cy, r = result[0]
    return float(cx), float(cy), float(r)


def _find_hole_candidates(
    gray: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> List[Tuple[float, float, float, float]]:
    candidates: List[Tuple[float, float, float, float]] = []
    h, w = gray.shape[:2]
    min_area = np.pi * min_radius * min_radius * 0.45
    max_area = np.pi * max_radius * max_radius * 1.8

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    masks = []
    _, otsu_dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    masks.append(otsu_dark)

    dark_cut = int(np.percentile(blurred, 18))
    _, percentile_dark = cv2.threshold(
        blurred, dark_cut, 255, cv2.THRESH_BINARY_INV
    )
    masks.append(percentile_dark)

    adaptive_dark = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 91, 7,
    )
    masks.append(adaptive_dark)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for mask in masks:
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            circularity = _contour_circularity(cnt)
            if circularity < 0.52:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < min_radius or r > max_radius:
                continue

            if cx < r or cy < r or cx > w - r or cy > h - r:
                continue

            circle_area = np.pi * r * r
            area_score = 1.0 - min(abs(area - circle_area) / max(circle_area, 1.0), 1.0)
            circle_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(circle_mask, (int(cx), int(cy)), int(max(r * 0.8, 1)), 255, -1)
            mean_intensity = cv2.mean(blurred, mask=circle_mask)[0]
            dark_score = 1.0 - min(mean_intensity / 255.0, 1.0)
            radius_score = min(r / max_radius, 1.0)

            score = (
                0.38 * circularity +
                0.27 * area_score +
                0.25 * dark_score +
                0.10 * radius_score
            )
            candidates.append((float(score), float(cx), float(cy), float(r)))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_radius * 4, 20),
        param1=80,
        param2=max(14, int(max_radius / 7)),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        for cx, cy, r in circles[0]:
            if cx < r or cy < r or cx > w - r or cy > h - r:
                continue
            circle_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(circle_mask, (int(cx), int(cy)), int(max(r * 0.8, 1)), 255, -1)
            mean_intensity = cv2.mean(blurred, mask=circle_mask)[0]
            dark_score = 1.0 - min(mean_intensity / 255.0, 1.0)
            candidates.append((0.52 + 0.30 * dark_score, float(cx), float(cy), float(r)))

    return _deduplicate_candidates(candidates)


def detect_all_holes(
    warped_gray: np.ndarray,
    board_size_px: Tuple[int, int],
    warped_color: np.ndarray = None,
) -> DetectionResult:
    """
    在俯视 PCB 图像中检测四个安装孔。

    board_size_px 只用于相对半径范围估计，不含任何 PCB 真实尺寸信息。
    """
    if warped_gray is None:
        return DetectionResult(success=False, message="俯视图为空")

    h, w = warped_gray.shape[:2]
    min_dim = max(1, min(w, h))
    min_r = max(6, int(min_dim * 0.006))
    max_r = max(min_r + 2, int(min_dim * 0.045))

    candidates = _find_hole_candidates(warped_gray, min_r, max_r)
    selected = _select_four_holes(candidates, (h, w))

    vis = warped_color.copy() if warped_color is not None \
        else cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    if len(selected) < 4:
        for _, cx, cy, r in candidates[:20]:
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 165, 255), 1)
        return DetectionResult(
            success=False,
            message=f"安装孔检测: {len(selected)}/4 成功",
            data={
                "hole_centers_px": [(0.0, 0.0)] * 4,
                "hole_radii_px": [0.0] * 4,
                "individual_success": [False] * 4,
                "visualization": vis,
                "hole_candidates": candidates,
            },
        )

    centers = [(c[1], c[2]) for c in selected]
    order = _order_holes(centers)
    selected = [selected[i] for i in order]

    hole_centers = []
    hole_radii = []
    for i, (_, cx, cy, r) in enumerate(selected):
        hole_centers.append((float(cx), float(cy)))
        hole_radii.append(float(r))
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(vis, f"H{i+1}", (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return DetectionResult(
        success=True,
        message="四个安装孔全部检测成功",
        data={
            "hole_centers_px": hole_centers,
            "hole_radii_px": hole_radii,
            "individual_success": [True] * 4,
            "visualization": vis,
            "hole_candidates": candidates,
        },
    )
