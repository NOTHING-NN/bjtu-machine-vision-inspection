"""
visualize_results.py — 结果可视化模块

提供检测结果叠加绘制、对比拼图和报告用图的生成功能。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合批量生成报告图片
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
    """
    在图像上绘制检测到的电路板四角点。

    Args:
        image: BGR 图像
        corners: 四角点坐标 (4, 2)
        color: 绘制颜色 (B, G, R)
        thickness: 线条粗细

    Returns:
        叠加了角点的图像副本
    """
    vis = image.copy()

    # 确保 corners 格式为 (4, 2)
    if corners.ndim == 3:
        corners = corners.reshape(4, 2)

    # 绘制四边形轮廓
    pts = corners.astype(np.int32)
    cv2.polylines(vis, [pts], True, color, thickness)

    # 绘制四个角点为实心圆，并标注序号
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
    """
    在图像上绘制检测到的安装孔。

    Args:
        image: BGR 或灰度图像
        hole_centers: 孔心像素坐标列表
        hole_radii: 孔半径列表
        color: 绘制颜色
        thickness: 圆周线宽

    Returns:
        叠加了圆孔标记的图像副本
    """
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
    """
    在图像上叠加测量结果文字。

    Args:
        image: BGR 图像
        pitch_x_mm: X 方向孔间距 (mm)
        pitch_y_mm: Y 方向孔间距 (mm)
        abs_error_x: X 方向绝对误差 (mm)
        abs_error_y: Y 方向绝对误差 (mm)

    Returns:
        叠加了文字信息的图像副本
    """
    vis = image.copy()
    h, w = vis.shape[:2]

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
    title_a: str = "Baseline",
    title_b: str = "Light Corrected",
    output_path: Optional[Path] = None,
) -> Path:
    """
    生成左右对比拼图（baseline vs light_corrected）。

    Args:
        image_a: 左侧图像（普通预处理结果）
        image_b: 右侧图像（光照矫正结果）
        title_a: 左图标题
        title_b: 右图标题
        output_path: 保存路径

    Returns:
        保存的文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=config.FIGURE_DPI)

    # 左图
    axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
                   if len(image_a.shape) == 3 else image_a, cmap="gray")
    axes[0].set_title(title_a, fontsize=14)
    axes[0].axis("off")

    # 右图
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
    """
    保存单张调试用图。

    Args:
        image: BGR 或灰度图像
        output_path: 保存路径
        title: 图片标题
        cmap: matplotlib colormap，灰度图用 "gray"
    """
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
# 综合报告图生成
# ============================================================

def generate_report_figures(
    image_name: str,
    original_image: np.ndarray,
    baseline_results: Dict,
    light_corrected_results: Dict,
) -> None:
    """
    为单张图像生成全套报告用图。

    包括：
      1. 原图
      2. 普通算法二值图
      3. 改进算法光照矫正图
      4. 普通算法边缘图
      5. 改进算法边缘图
      6. 普通算法最终检测结果
      7. 改进算法最终检测结果
      8. 对比拼图（边对边检测结果对比）

    Args:
        image_name: 图像名称（不含扩展名）
        original_image: 原始 BGR 图像
        baseline_results: 普通预处理流程的全部结果 dict
        light_corrected_results: 光照矫正流程的全部结果 dict
    """
    comp_dir = config.COMPARISON_OUTPUT_DIR / image_name
    comp_dir.mkdir(parents=True, exist_ok=True)

    # 1. 原图
    save_debug_figure(
        original_image,
        comp_dir / f"{image_name}_01_original.png",
        "Original Image",
    )

    # 2. 普通算法二值图
    if "binary" in baseline_results.get("preproc_results", {}):
        save_debug_figure(
            baseline_results["preproc_results"]["binary"],
            comp_dir / f"{image_name}_02_baseline_binary.png",
            "Baseline — Binary",
            cmap="gray",
        )

    # 3. 改进算法光照矫正图（ExG 或 illum_corrected）
    lc_preproc = light_corrected_results.get("preproc_results", {})
    if "illum_corrected" in lc_preproc:
        save_debug_figure(
            lc_preproc["illum_corrected"],
            comp_dir / f"{image_name}_03_light_corrected.png",
            "Light Corrected — Illumination Corrected",
            cmap="gray",
        )

    # 4. 普通算法边缘图
    if "edge" in baseline_results.get("preproc_results", {}):
        save_debug_figure(
            baseline_results["preproc_results"]["edge"],
            comp_dir / f"{image_name}_04_baseline_edge.png",
            "Baseline — Canny Edge",
            cmap="gray",
        )

    # 5. 改进算法边缘图
    if "edge" in lc_preproc:
        save_debug_figure(
            lc_preproc["edge"],
            comp_dir / f"{image_name}_05_light_corrected_edge.png",
            "Light Corrected — Canny Edge",
            cmap="gray",
        )

    # 6. 普通算法最终检测结果
    if baseline_results.get("board_result") is not None:
        board_r = baseline_results["board_result"]
        if board_r.success:
            warped = board_r.data["warped_image"]
            hole_r = baseline_results.get("hole_result")
            vis_baseline = warped.copy()
            if len(vis_baseline.shape) == 2:
                vis_baseline = cv2.cvtColor(vis_baseline, cv2.COLOR_GRAY2BGR)

            # 叠加孔检测
            if hole_r is not None and hole_r.success:
                vis_baseline = draw_holes(
                    vis_baseline,
                    hole_r.data["hole_centers_px"],
                    hole_r.data["hole_radii_px"],
                )

            # 叠加测量文字
            if not np.isnan(baseline_results.get("pitch_x_mm", np.nan)):
                vis_baseline = draw_measurement_text(
                    vis_baseline,
                    baseline_results["pitch_x_mm"],
                    baseline_results["pitch_y_mm"],
                    baseline_results["abs_error_x_mm"],
                    baseline_results["abs_error_y_mm"],
                )

            save_debug_figure(
                vis_baseline,
                comp_dir / f"{image_name}_06_baseline_detection.png",
                "Baseline — Detection Results",
            )

    # 7. 改进算法最终检测结果
    if light_corrected_results.get("board_result") is not None:
        board_r = light_corrected_results["board_result"]
        if board_r.success:
            warped = board_r.data["warped_image"]
            hole_r = light_corrected_results.get("hole_result")
            vis_light = warped.copy()
            if len(vis_light.shape) == 2:
                vis_light = cv2.cvtColor(vis_light, cv2.COLOR_GRAY2BGR)

            if hole_r is not None and hole_r.success:
                vis_light = draw_holes(
                    vis_light,
                    hole_r.data["hole_centers_px"],
                    hole_r.data["hole_radii_px"],
                )

            if not np.isnan(light_corrected_results.get("pitch_x_mm", np.nan)):
                vis_light = draw_measurement_text(
                    vis_light,
                    light_corrected_results["pitch_x_mm"],
                    light_corrected_results["pitch_y_mm"],
                    light_corrected_results["abs_error_x_mm"],
                    light_corrected_results["abs_error_y_mm"],
                )

            save_debug_figure(
                vis_light,
                comp_dir / f"{image_name}_07_light_corrected_detection.png",
                "Light Corrected — Detection Results",
            )

            # 8. 对比拼图
            if baseline_results.get("board_result") is not None \
               and baseline_results["board_result"].success:
                warped_b = baseline_results["board_result"].data["warped_image"]
                vis_b = warped_b.copy()
                if len(vis_b.shape) == 2:
                    vis_b = cv2.cvtColor(vis_b, cv2.COLOR_GRAY2BGR)

                bh_r = baseline_results.get("hole_result")
                if bh_r is not None and bh_r.success:
                    vis_b = draw_holes(vis_b, bh_r.data["hole_centers_px"],
                                       bh_r.data["hole_radii_px"])

                make_side_by_side_comparison(
                    vis_b, vis_light,
                    title_a="Baseline",
                    title_b="Light Corrected",
                    output_path=comp_dir / f"{image_name}_08_comparison.png",
                )

    print(f"  [INFO] 报告图已保存到: {comp_dir}")
