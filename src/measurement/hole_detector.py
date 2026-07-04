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
    """在 ROI 内检测最可信圆孔。"""
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


def _sample_radial_profile(
    gray: np.ndarray,
    cx: float,
    cy: float,
    max_radius: int,
    num_angles: int = 96,
) -> Optional[np.ndarray]:
    """采样以孔心为中心的径向灰度均值曲线。"""
    h, w = gray.shape[:2]
    if max_radius < 4:
        return None

    angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    profile = []

    for r in range(max_radius + 1):
        xs = np.rint(cx + r * cos_a).astype(np.int32)
        ys = np.rint(cy + r * sin_a).astype(np.int32)
        valid = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
        if np.count_nonzero(valid) < num_angles * 0.65:
            profile.append(np.nan)
            continue
        profile.append(float(np.mean(gray[ys[valid], xs[valid]])))

    profile_arr = np.asarray(profile, dtype=np.float32)
    if np.count_nonzero(np.isfinite(profile_arr)) < 8:
        return None
    return profile_arr


def _refine_hole_radius_by_intensity(
    gray: np.ndarray,
    cx: float,
    cy: float,
    initial_radius: float,
    min_radius: int,
    max_radius: int,
) -> float:
    """
    基于径向灰度剖面精修孔半径。

    候选检测阶段的外接圆容易把焊盘黑环、阴影或板边背景一起包进去；
    这里只寻找“暗孔区域 -> 较亮 PCB 表面”的第一处稳定跃迁。
    不使用孔径标称值。
    """
    if gray is None or initial_radius <= 0:
        return float(initial_radius)

    h, w = gray.shape[:2]
    safe_max = int(min(max_radius, cx, cy, w - 1 - cx, h - 1 - cy))
    if safe_max <= max(4, min_radius):
        return float(initial_radius)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    search_max = max(
        min_radius + 3,
        int(min(safe_max, max(max_radius * 0.95, initial_radius * 1.15))),
    )
    profile = _sample_radial_profile(blurred, cx, cy, search_max)
    if profile is None:
        return float(initial_radius)

    valid = np.isfinite(profile)
    if not np.any(valid):
        return float(initial_radius)

    # 用线性插值补齐少量越界点，再平滑，避免单个焊盘孔或丝印造成尖峰。
    radii = np.arange(len(profile), dtype=np.float32)
    profile = np.interp(radii, radii[valid], profile[valid]).astype(np.float32)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth = np.convolve(profile, kernel, mode="same")

    inner_end = max(3, int(min_radius * 0.75))
    inner_level = float(np.percentile(smooth[:inner_end + 1], 35))
    outer_start = max(inner_end + 2, int(search_max * 0.55))
    outer_level = float(np.percentile(smooth[outer_start:], 70))
    contrast = outer_level - inner_level
    if contrast < 8.0:
        return float(initial_radius)

    start = max(3, int(min_radius * 0.65))
    stop = max(start + 3, search_max - 2)
    threshold = inner_level + 0.32 * contrast

    crossing_radius = None
    for r in range(start, stop):
        if smooth[r] >= threshold:
            crossing_radius = float(r)
            break

    gradient = np.gradient(smooth)
    grad_slice = gradient[start:stop]
    edge_radius = float(start + int(np.argmax(grad_slice))) if len(grad_slice) else None

    if crossing_radius is not None and edge_radius is not None:
        refined = 0.78 * crossing_radius + 0.22 * edge_radius
    elif crossing_radius is not None:
        refined = crossing_radius
    elif edge_radius is not None:
        refined = edge_radius
    else:
        return float(initial_radius)

    lower = max(3.0, min_radius * 0.65)
    upper = min(float(safe_max), max_radius * 0.9)
    refined = float(np.clip(refined, lower, upper))

    # 极端跳变通常来自板边背景或丝印，保留原候选半径更稳。
    if refined > initial_radius * 1.8 or refined < initial_radius * 0.25:
        return float(initial_radius)

    return refined


def _refine_hole_radius_by_dark_component(
    gray: np.ndarray,
    cx: float,
    cy: float,
    initial_radius: float,
    min_radius: int,
    max_radius: int,
) -> Optional[float]:
    """
    用孔心附近的暗连通域估计孔半径。

    相比最小外接圆，面积等效半径对焊盘外缘毛刺和局部阴影更不敏感。
    """
    if gray is None or initial_radius <= 0:
        return None

    h, w = gray.shape[:2]
    roi_r = int(max(min_radius * 2.0, min(max_radius, initial_radius * 1.35)))
    x1 = max(0, int(round(cx - roi_r)))
    y1 = max(0, int(round(cy - roi_r)))
    x2 = min(w, int(round(cx + roi_r + 1)))
    y2 = min(h, int(round(cy + roi_r + 1)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    roi = gray[y1:y2, x1:x2]
    local_cx = int(round(cx - x1))
    local_cy = int(round(cy - y1))
    if not (0 <= local_cx < roi.shape[1] and 0 <= local_cy < roi.shape[0]):
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    dist = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
    inner_mask = dist <= max(2.0, min_radius * 0.55)
    outer_mask = (dist >= max(min_radius * 1.2, roi_r * 0.45)) & (dist <= roi_r)
    if not np.any(inner_mask) or not np.any(outer_mask):
        return None

    inner_level = float(np.percentile(blurred[inner_mask], 45))
    outer_level = float(np.percentile(blurred[outer_mask], 70))
    contrast = outer_level - inner_level
    if contrast < 6.0:
        return None

    threshold = inner_level + 0.46 * contrast
    dark = (blurred <= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return None

    center_label = labels[local_cy, local_cx]
    if center_label == 0:
        search = labels[
            max(0, local_cy - 2): min(labels.shape[0], local_cy + 3),
            max(0, local_cx - 2): min(labels.shape[1], local_cx + 3),
        ]
        labels_nonzero = search[search > 0]
        if labels_nonzero.size == 0:
            return None
        center_label = int(np.bincount(labels_nonzero.ravel()).argmax())

    area = float(stats[center_label, cv2.CC_STAT_AREA])
    if area <= 0:
        return None

    component = labels == center_label
    comp_ys, comp_xs = np.where(component)
    comp_dist = np.sqrt((comp_xs - local_cx) ** 2 + (comp_ys - local_cy) ** 2)
    if comp_dist.size == 0:
        return None

    area_radius = float(np.sqrt(area / np.pi))
    radial_radius = float(np.percentile(comp_dist, 82))
    refined = 0.72 * area_radius + 0.28 * radial_radius
    refined = float(np.clip(refined, max(3.0, min_radius * 0.65), max_radius * 0.9))

    if refined > initial_radius * 1.45 or refined < initial_radius * 0.22:
        return None
    return refined


def _stabilize_radii_consistency(radii: List[float]) -> List[float]:
    """
    对同一张 PCB 的四个安装孔半径做离群抑制。

    这里不使用孔径标准值，只利用同一图像中四个安装孔属于同类孔这一事实；
    过小的半径通常来自径向剖面被噪声提前截断，过大的半径通常来自阴影/焊盘被并入。
    """
    if len(radii) < 4:
        return radii

    arr = np.asarray(radii, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > 0)
    if np.count_nonzero(finite) < 3:
        return radii

    med = float(np.median(arr[finite]))
    if med <= 0:
        return radii

    stabilized = []
    for r in arr:
        if not np.isfinite(r) or r <= 0:
            stabilized.append(float(r))
            continue
        ratio = float(r / med)
        if ratio < 0.72 or ratio > 1.28:
            stabilized.append(med)
        else:
            stabilized.append(float(r))
    return stabilized


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
    radial_radii = []
    component_radii = []
    for _, cx, cy, r in selected:
        radial_r = _refine_hole_radius_by_intensity(
            warped_gray, cx, cy, r, min_r, max_r
        )
        component_r = _refine_hole_radius_by_dark_component(
            warped_gray, cx, cy, r, min_r, max_r
        )
        radial_radii.append(float(radial_r))
        component_radii.append(float(component_r) if component_r is not None else np.nan)

    refined_radii = _stabilize_radii_consistency(radial_radii)

    finite_radii = np.asarray(refined_radii, dtype=np.float32)
    valid = np.isfinite(finite_radii) & (finite_radii > 0)
    group_median = float(np.median(finite_radii[valid])) if np.any(valid) else np.nan

    # 暗连通域仅用于抑制明显偏大的径向半径估计。
    if np.isfinite(group_median) and group_median > 0:
        guarded_radii = []
        for radial_r, component_r in zip(refined_radii, component_radii):
            if (
                np.isfinite(component_r)
                and component_r > group_median * 0.72
                and component_r < radial_r
                and radial_r > group_median * 1.18
            ):
                guarded_radii.append(float(0.55 * radial_r + 0.45 * component_r))
            else:
                guarded_radii.append(float(radial_r))
        refined_radii = _stabilize_radii_consistency(guarded_radii)

    for i, ((_, cx, cy, _), refined_r) in enumerate(zip(selected, refined_radii)):
        hole_centers.append((float(cx), float(cy)))
        hole_radii.append(float(refined_r))
        cv2.circle(vis, (int(cx), int(cy)), int(refined_r), (0, 255, 0), 2)
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
