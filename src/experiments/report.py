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
import numpy as np

from src import config
from src.utils import save_image


# ============================================================
# 叠加绘制函数
# ============================================================

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
            cv2.circle(vis, center, int(r), color, thickness)
            cv2.circle(vis, center, 3, (0, 0, 255), -1)
            if i < len(labels):
                cv2.putText(vis, labels[i], (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return vis


def draw_measurement_text(
    image: np.ndarray,
    pitch_x_mm: float,
    pitch_y_mm: float,
    abs_error_x: float,
    abs_error_y: float,
) -> np.ndarray:
    """在图像上叠加测量结果文字。"""
    vis = image.copy()

    texts = [
        f"Pitch X: {pitch_x_mm:.3f} mm  (error: {abs_error_x:.3f} mm)",
        f"Pitch Y: {pitch_y_mm:.3f} mm  (error: {abs_error_y:.3f} mm)",
    ]

    y0 = 25
    for i, text in enumerate(texts):
        cv2.putText(vis, text, (10, y0 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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


# ============================================================
# 综合报告图
# ============================================================

def generate_report_figures(
    image_name: str,
    original_image: np.ndarray,
    baseline_result: Dict,
    light_corrected_result: Dict,
    output_base_dir: Path = None,
) -> None:
    """
    为单张图像生成全套报告用图，按统一编号命名。

    输出：
      {name}_01_original.png
      {name}_02_algo_a_mask.png
      {name}_03_algo_b_mask.png
      {name}_04_candidates.png
      {name}_05_warped.png
      {name}_06_detection.png
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

    # 02 — Baseline mask
    if baseline_result is not None:
        b_green = baseline_result.get("green_mask")
        if b_green is not None:
            save_debug_figure(b_green,
                              output_base_dir / f"{image_name}_02_algo_a_mask.png",
                              "算法 A — PCB Mask", cmap="gray")
        # 04 — Candidates
        b_board = baseline_result.get("board_result")
        if b_board is not None and b_board.success:
            candidates_vis = original_image.copy() if original_image is not None else None
            if candidates_vis is not None and "candidates" in b_board.data:
                vis = draw_board_corners(candidates_vis, b_board.data["corners"])
                save_debug_figure(vis,
                                  output_base_dir / f"{image_name}_04a_algo_a_candidates.png",
                                  "算法 A — Board Corners")
            # 05 — Warped
            warped = b_board.data["warped_image"]
            save_debug_figure(warped,
                              output_base_dir / f"{image_name}_05a_algo_a_warped.png",
                              "算法 A — Warped")
            # 06 — Detection
            b_hole = baseline_result.get("hole_result")
            vis = warped.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if b_hole is not None and b_hole.success:
                vis = draw_holes(vis, b_hole.data["hole_centers_px"],
                                 b_hole.data["hole_radii_px"])
            if not np.isnan(baseline_result.get("pitch_x_mm", np.nan)):
                vis = draw_measurement_text(
                    vis, baseline_result["pitch_x_mm"],
                    baseline_result["pitch_y_mm"],
                    baseline_result["abs_error_x_mm"],
                    baseline_result["abs_error_y_mm"],
                )
            save_debug_figure(vis,
                              output_base_dir / f"{image_name}_06a_algo_a_detection.png",
                              "算法 A — Detection & Measurement")

    # 03 — Light Corrected mask
    if light_corrected_result is not None:
        l_green = light_corrected_result.get("green_mask")
        if l_green is not None:
            save_debug_figure(l_green,
                              output_base_dir / f"{image_name}_03_algo_b_mask.png",
                              "算法 B — PCB Mask", cmap="gray")
        # Highlight mask
        l_preproc = light_corrected_result.get("preproc_results", {})
        hl_mask = l_preproc.get("highlight_mask")
        if hl_mask is not None:
            save_debug_figure(hl_mask,
                              output_base_dir / f"{image_name}_03b_highlight_mask.png",
                              "算法 B — Highlight Mask", cmap="gray")

        l_board = light_corrected_result.get("board_result")
        if l_board is not None and l_board.success:
            candidates_vis = original_image.copy() if original_image is not None else None
            if candidates_vis is not None and "candidates" in l_board.data:
                vis = draw_board_corners(candidates_vis, l_board.data["corners"])
                save_debug_figure(vis,
                                  output_base_dir / f"{image_name}_04b_algo_b_candidates.png",
                                  "算法 B — Board Corners")
            warped = l_board.data["warped_image"]
            save_debug_figure(warped,
                              output_base_dir / f"{image_name}_05b_algo_b_warped.png",
                              "算法 B — Warped")
            l_hole = light_corrected_result.get("hole_result")
            vis = warped.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if l_hole is not None and l_hole.success:
                vis = draw_holes(vis, l_hole.data["hole_centers_px"],
                                 l_hole.data["hole_radii_px"])
            if not np.isnan(light_corrected_result.get("pitch_x_mm", np.nan)):
                vis = draw_measurement_text(
                    vis, light_corrected_result["pitch_x_mm"],
                    light_corrected_result["pitch_y_mm"],
                    light_corrected_result["abs_error_x_mm"],
                    light_corrected_result["abs_error_y_mm"],
                )
            save_debug_figure(vis,
                              output_base_dir / f"{image_name}_06b_algo_b_detection.png",
                              "算法 B — Detection & Measurement")

    print(f"  [INFO] 报告图已保存到: {output_base_dir}")
