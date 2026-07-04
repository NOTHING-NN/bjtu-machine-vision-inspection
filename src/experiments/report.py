"""
report.py — 结果可视化与报告生成

提供检测结果叠加绘制、对比拼图、调试用图保存和
综合报告生成功能。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 中文字体配置 ──
def _setup_cjk_font():
    """按优先级查找系统可用中文字体，避免 matplotlib 方块/乱码。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "KaiTi", "FangSong",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 回退：扫描 ttf 列表找任意含 CJK 字符的字体名
    for f in fm.fontManager.ttflist:
        if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in f.name):
            plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return

_setup_cjk_font()

from src import config
from src.utils import save_image


# ============================================================
# 叠加绘制函数
# ============================================================

def _draw_text_box(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.75,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (20, 20, 20),
    thickness: int = 2,
    pad: int = 6,
) -> None:
    """绘制带半透明底色的清晰文字标签。"""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = max(0, x - pad)
    y1 = max(0, y - th - baseline - pad)
    x2 = min(image.shape[1] - 1, x + tw + pad)
    y2 = min(image.shape[0] - 1, y + baseline + pad)

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, dst=image)
    cv2.putText(image, text, (x, y), font, font_scale, color,
                thickness, cv2.LINE_AA)


def draw_board_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制检测到的电路板四角点。"""
    vis = image.copy()

    if corners.ndim == 3:
        corners = corners.reshape(4, 2)

    pts = corners.astype(np.int32)
    cv2.polylines(vis, [pts], True, color, thickness)

    labels = ["TL", "TR", "BR", "BL"]
    for i, (pt, label) in enumerate(zip(pts, labels)):
        pt = tuple(pt)
        cv2.circle(vis, pt, 8, (0, 0, 255), -1)
        cv2.putText(vis, label, (pt[0] + 12, pt[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return vis


def draw_holes(
    image: np.ndarray,
    hole_centers: List[Tuple[float, float]],
    hole_radii: List[float],
    hole_diameters_mm: List[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制检测到的安装孔。"""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    labels = ["H1", "H2", "H3", "H4"]
    for i, ((cx, cy), r) in enumerate(zip(hole_centers, hole_radii)):
        if r > 0:
            center = (int(cx), int(cy))
            cv2.circle(vis, center, int(r), (0, 180, 0), max(thickness + 4, 6))
            cv2.circle(vis, center, int(r), color, max(thickness + 1, 3))
            cv2.circle(vis, center, 7, (0, 0, 255), -1)
            cv2.circle(vis, center, 11, (255, 255, 255), 2)
            if i < len(labels):
                label_text = labels[i]
                if hole_diameters_mm is not None and i < len(hole_diameters_mm):
                    d = hole_diameters_mm[i]
                    if d is not None and not np.isnan(d):
                        label_text = f"{label_text}  D={d:.2f}mm"

                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
                )
                tx = center[0] + 14
                if tx + tw + 14 >= vis.shape[1]:
                    tx = center[0] - tw - 20
                tx = max(10, min(tx, vis.shape[1] - tw - 10))
                ty = max(center[1] - 14, th + baseline + 14)
                if center[1] > int(vis.shape[0] * 0.75):
                    ty = center[1] - max(72, int(r) * 3)
                if ty + baseline + 8 >= vis.shape[0]:
                    ty = max(th + baseline + 14, center[1] - int(r) - 18)
                _draw_text_box(
                    vis, label_text, (tx, ty),
                    font_scale=0.85,
                    color=(255, 255, 255),
                    bg_color=(37, 99, 235),
                    thickness=2,
                )

    return vis


def draw_board_size_annotations(
    image: np.ndarray,
    board_width_mm: float = np.nan,
    board_height_mm: float = np.nan,
) -> np.ndarray:
    """在俯视图上绘制板宽、板高尺寸箭头。"""
    vis = image.copy()
    h, w = vis.shape[:2]
    if h <= 0 or w <= 0:
        return vis

    arrow_color = (255, 255, 0)
    shadow = (0, 0, 0)
    margin = max(26, int(min(w, h) * 0.035))
    arrow_thickness = max(3, int(min(w, h) * 0.004))

    if not np.isnan(board_width_mm):
        y = h - margin
        x1 = margin
        x2 = w - margin
        cv2.arrowedLine(vis, (x1, y), (x2, y), shadow,
                        arrow_thickness + 3, tipLength=0.025)
        cv2.arrowedLine(vis, (x2, y), (x1, y), shadow,
                        arrow_thickness + 3, tipLength=0.025)
        cv2.arrowedLine(vis, (x1, y), (x2, y), arrow_color,
                        arrow_thickness, tipLength=0.025)
        cv2.arrowedLine(vis, (x2, y), (x1, y), arrow_color,
                        arrow_thickness, tipLength=0.025)
        _draw_text_box(
            vis, f"W = {board_width_mm:.2f} mm",
            (max(margin, int(w * 0.40)), max(34, y - 14)),
            font_scale=0.78,
            color=(255, 255, 255),
            bg_color=(2, 132, 199),
            thickness=2,
        )

    if not np.isnan(board_height_mm):
        x = w - margin
        y1 = margin
        y2 = h - margin
        cv2.arrowedLine(vis, (x, y1), (x, y2), shadow,
                        arrow_thickness + 3, tipLength=0.025)
        cv2.arrowedLine(vis, (x, y2), (x, y1), shadow,
                        arrow_thickness + 3, tipLength=0.025)
        cv2.arrowedLine(vis, (x, y1), (x, y2), arrow_color,
                        arrow_thickness, tipLength=0.025)
        cv2.arrowedLine(vis, (x, y2), (x, y1), arrow_color,
                        arrow_thickness, tipLength=0.025)
        _draw_text_box(
            vis, f"H = {board_height_mm:.2f} mm",
            (max(10, x - 230), max(58, int(h * 0.50))),
            font_scale=0.78,
            color=(255, 255, 255),
            bg_color=(2, 132, 199),
            thickness=2,
        )

    return vis


def draw_measurement_text(
    image: np.ndarray,
    pitch_x_mm: float,
    pitch_y_mm: float,
    abs_error_x: float,
    abs_error_y: float,
    board_width_mm: float = np.nan,
    board_height_mm: float = np.nan,
    board_width_error_mm: float = np.nan,
    board_height_error_mm: float = np.nan,
    hole_diameters_mm: List[float] = None,
    hole_centers_world_mm: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """在图像上叠加测量结果文字（板尺寸+孔距+孔直径+孔心坐标）。"""
    vis = image.copy()

    texts = [
        f"Pitch X = {pitch_x_mm:.3f} mm   err = {abs_error_x:.3f} mm",
        f"Pitch Y = {pitch_y_mm:.3f} mm   err = {abs_error_y:.3f} mm",
    ]

    # 板尺寸
    if not np.isnan(board_width_mm):
        texts.append(
            f"Board W = {board_width_mm:.3f} mm   err = {board_width_error_mm:.3f} mm"
        )
    if not np.isnan(board_height_mm):
        texts.append(
            f"Board H = {board_height_mm:.3f} mm   err = {board_height_error_mm:.3f} mm"
        )

    # 孔直径
    if hole_diameters_mm is not None:
        dia_strs = []
        for i, d in enumerate(hole_diameters_mm, start=1):
            if not np.isnan(d):
                dia_strs.append(f"D{i}={d:.2f}")
        if dia_strs:
            texts.append("Hole Dia: " + "  ".join(dia_strs) + " mm")

    # 孔心坐标 (world mm)
    if hole_centers_world_mm is not None:
        coord_parts = []
        for i, (cx, cy) in enumerate(hole_centers_world_mm, start=1):
            if not np.isnan(cx) and not np.isnan(cy):
                coord_parts.append(f"H{i}({cx:.1f},{cy:.1f})")
        if coord_parts:
            texts.append("Hole Pos: " + "  ".join(coord_parts) + " mm")

    panel_lines = len(texts)
    line_h = 32
    panel_w = min(vis.shape[1] - 20, 900)
    panel_h = panel_lines * line_h + 16
    panel_x = 10
    panel_y = max(10, int(vis.shape[0] * 0.14))
    overlay = vis.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, dst=vis)
    cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (148, 163, 184), 2)

    y0 = panel_y + 28
    for i, text in enumerate(texts):
        cv2.putText(vis, text, (panel_x + 14, y0 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255),
                    2, cv2.LINE_AA)

    return vis


# ============================================================
# 对比排版
# ============================================================

def make_side_by_side_comparison(
    image_a: np.ndarray,
    image_b: np.ndarray,
    title_a: str = "算法 A (基础测量)",
    title_b: str = "算法 B (强光斑感知改进)",
    output_path: Optional[Path] = None,
) -> Path:
    """生成左右对比拼图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=config.FIGURE_DPI)

    axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
                   if len(image_a.shape) == 3 else image_a, cmap="gray")
    axes[0].set_title(title_a, fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
                   if len(image_b.shape) == 3 else image_b, cmap="gray")
    axes[1].set_title(title_b, fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=config.FIGURE_DPI,
                    bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)
    return output_path


def save_debug_figure(
    image: np.ndarray,
    output_path: Path,
    title: str = "",
    cmap: str = None,
) -> None:
    """保存单张调试用图。"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=config.FIGURE_DPI)

    if len(image.shape) == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap=cmap or "gray")

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=config.FIGURE_DPI,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def draw_largest_component_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> Optional[np.ndarray]:
    """Overlay the largest foreground connected component on the source image."""
    if image is None or mask is None:
        return None

    vis = image.copy()
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return vis

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest_label).astype(np.uint8)

    overlay = vis.copy()
    overlay[component > 0] = color
    vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 5)

    x, y, w, h, area = stats[largest_label]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 4)
    cv2.putText(vis, f"largest component area={int(area)}",
                (max(10, x), max(30, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return vis


# ============================================================
# 综合报告图
# ============================================================

def generate_report_figures(
    image_name: str,
    original_image: np.ndarray,
    algo_a_result: Dict,
    algo_b_result: Dict,
    output_base_dir: Path = None,
) -> None:
    """
    为单张图像生成全套报告用图，按统一编号命名。

    对于 A 类样本（仅算法 A 成功）：
      {name}_01_original.png
      {name}_02_algo_a_mask.png
      {name}_04a_algo_a_candidates.png
      {name}_05a_algo_a_warped.png
      {name}_06a_algo_a_detection.png

    对于 B 类样本（仅算法 B 成功）：
      {name}_01_original.png
      {name}_03_algo_b_mask.png
      {name}_03b_highlight_mask.png
      {name}_04b_algo_b_candidates.png
      {name}_05b_algo_b_warped.png
      {name}_06b_algo_b_detection.png

    算法 A 在 B 类上失败时额外输出：
      {name}_02b_algo_a_largest_component.png
    """
    if output_base_dir is None:
        output_base_dir = config.REPORTS_OUTPUT_DIR / image_name
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # 01 — 原图
    if original_image is not None:
        save_debug_figure(original_image,
                          output_base_dir / f"{image_name}_01_original.png",
                          "Original Image")

    # 02 — 算法 A mask
    if algo_a_result is not None:
        a_green = algo_a_result.get("green_mask")
        a_board = algo_a_result.get("board_result")
        if a_green is None and a_board is not None:
            a_green = a_board.data.get("mask")
        if a_green is not None:
            save_debug_figure(a_green,
                              output_base_dir / f"{image_name}_02_algo_a_mask.png",
                              "算法 A — PCB Mask", cmap="gray")
        # 算法 A 失败时：显示最大连通域
        a_board = algo_a_result.get("board_result")
        if a_green is not None and a_board is not None and not a_board.success:
            component_vis = draw_largest_component_overlay(original_image, a_green)
            if component_vis is not None:
                save_debug_figure(
                    component_vis,
                    output_base_dir / f"{image_name}_02b_algo_a_largest_component.png",
                    "算法 A 失败: 最大连通域",
                )

        if a_board is not None and a_board.success:
            candidates_vis = original_image.copy() if original_image is not None else None
            if candidates_vis is not None and "corners" in a_board.data:
                vis = draw_board_corners(candidates_vis, a_board.data["corners"])
                save_debug_figure(vis,
                                  output_base_dir / f"{image_name}_04a_algo_a_candidates.png",
                                  "算法 A — Board Corners")
            # 05 — Warped
            warped = a_board.data["warped_image"]
            save_debug_figure(warped,
                              output_base_dir / f"{image_name}_05a_algo_a_warped.png",
                              "算法 A — Warped")
            # 06 — Detection
            a_hole = algo_a_result.get("hole_result")
            vis = warped.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if a_hole is not None and a_hole.success:
                vis = draw_holes(vis, a_hole.data["hole_centers_px"],
                                 a_hole.data["hole_radii_px"],
                                 algo_a_result.get("hole_diameters_mm"))
            vis = draw_board_size_annotations(
                vis,
                algo_a_result.get("board_width_mm", np.nan),
                algo_a_result.get("board_height_mm", np.nan),
            )
            if not np.isnan(algo_a_result.get("pitch_x_mm", np.nan)):
                vis = draw_measurement_text(
                    vis, algo_a_result["pitch_x_mm"],
                    algo_a_result["pitch_y_mm"],
                    algo_a_result["abs_error_x_mm"],
                    algo_a_result["abs_error_y_mm"],
                    algo_a_result.get("board_width_mm", np.nan),
                    algo_a_result.get("board_height_mm", np.nan),
                    algo_a_result.get("board_width_error_mm", np.nan),
                    algo_a_result.get("board_height_error_mm", np.nan),
                    algo_a_result.get("hole_diameters_mm"),
                    algo_a_result.get("hole_centers_world_mm"),
                )
            save_debug_figure(vis,
                              output_base_dir / f"{image_name}_06a_algo_a_detection.png",
                              "算法 A — Detection & Measurement")

    # 03 — 算法 B mask
    if algo_b_result is not None:
        b_green = algo_b_result.get("green_mask")
        if b_green is not None:
            save_debug_figure(b_green,
                              output_base_dir / f"{image_name}_03_algo_b_mask.png",
                              "算法 B — PCB Mask", cmap="gray")
        # Highlight mask
        b_preproc = algo_b_result.get("preproc_results", {})
        hl_mask = b_preproc.get("highlight_mask")
        if hl_mask is not None:
            save_debug_figure(hl_mask,
                              output_base_dir / f"{image_name}_03b_highlight_mask.png",
                              "算法 B — Highlight Mask", cmap="gray")

        b_board = algo_b_result.get("board_result")
        if b_board is not None and b_board.success:
            candidates_vis = original_image.copy() if original_image is not None else None
            if candidates_vis is not None and "corners" in b_board.data:
                vis = draw_board_corners(candidates_vis, b_board.data["corners"])
                save_debug_figure(vis,
                                  output_base_dir / f"{image_name}_04b_algo_b_candidates.png",
                                  "算法 B — Board Corners")
            warped = b_board.data["warped_image"]
            save_debug_figure(warped,
                              output_base_dir / f"{image_name}_05b_algo_b_warped.png",
                              "算法 B — Warped")
            b_hole = algo_b_result.get("hole_result")
            vis = warped.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if b_hole is not None and b_hole.success:
                vis = draw_holes(vis, b_hole.data["hole_centers_px"],
                                 b_hole.data["hole_radii_px"],
                                 algo_b_result.get("hole_diameters_mm"))
            vis = draw_board_size_annotations(
                vis,
                algo_b_result.get("board_width_mm", np.nan),
                algo_b_result.get("board_height_mm", np.nan),
            )
            if not np.isnan(algo_b_result.get("pitch_x_mm", np.nan)):
                vis = draw_measurement_text(
                    vis, algo_b_result["pitch_x_mm"],
                    algo_b_result["pitch_y_mm"],
                    algo_b_result["abs_error_x_mm"],
                    algo_b_result["abs_error_y_mm"],
                    algo_b_result.get("board_width_mm", np.nan),
                    algo_b_result.get("board_height_mm", np.nan),
                    algo_b_result.get("board_width_error_mm", np.nan),
                    algo_b_result.get("board_height_error_mm", np.nan),
                    algo_b_result.get("hole_diameters_mm"),
                    algo_b_result.get("hole_centers_world_mm"),
                )
            save_debug_figure(vis,
                              output_base_dir / f"{image_name}_06b_algo_b_detection.png",
                              "算法 B — Detection & Measurement")

    print(f"  [INFO] 报告图已保存到: {output_base_dir}")
