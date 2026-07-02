"""
compare_experiment.py — 对比实验主流程

对每张 PCB 待测图像分别运行：
  A 组 — 普通预处理 (baseline)
  B 组 — 光照矫正预处理 (light_corrected)

两队后续检测流程完全一致：
  预处理 → detect_board → warp_board → detect_holes → measure_geometry

输出：
  - 每张图像的中间处理图及最终检测叠加图
  - measurements_baseline.csv / measurements_light_corrected.csv
  - comparison_summary.csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src import config
from src.utils import get_image_paths, load_image, save_image, DetectionResult
from src.calibrate_camera import load_camera_params, undistort_image
from src.preprocess_baseline import preprocess_baseline, save_baseline_steps
from src.preprocess_light_correct import (
    preprocess_light_corrected,
    save_light_corrected_steps,
)
from src.detect_board import detect_board
from src.detect_holes import detect_all_holes
from src.measure_geometry import run_measurement, summarize_measurements


# ============================================================
# 单张图像处理
# ============================================================

def process_single_image_baseline(
    image_path: Path,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """
    使用普通预处理流程处理单张 PCB 图像（A 组）。

    Args:
        image_path: PCB 图像文件路径
        camera_params: 相机标定参数

    Returns:
        包含测量结果和中间图像的 dict
    """
    image_name = image_path.stem
    print(f"\n{'─'*50}")
    print(f"  [A 组 - 普通预处理] 处理图像: {image_name}")

    # 加载图像
    image = load_image(image_path)
    if image is None:
        return None

    # 普通预处理
    preproc_results = preprocess_baseline(image, camera_params)

    if config.SAVE_INTERMEDIATE_IMAGES:
        save_baseline_steps(preproc_results, image_name)

    # 检测电路板外框
    board_result = detect_board(
        preproc_results["corrected"],
        preproc_results["binary"],
        preproc_results["edge"],
    )

    if not board_result.success:
        print(f"  [FAIL] 电路板检测失败: {board_result.message}")
        return {
            "image_name": image_name,
            "method": "baseline",
            "board_detect_success": False,
            "holes_detect_success": False,
            "pitch_x_mm": np.nan,
            "pitch_y_mm": np.nan,
            "abs_error_x_mm": np.nan,
            "abs_error_y_mm": np.nan,
            "rel_error_x_pct": np.nan,
            "rel_error_y_pct": np.nan,
            "mean_pitch_mm": np.nan,
            "mean_abs_error_mm": np.nan,
            "remark": f"电路板检测失败: {board_result.message}",
            "preproc_results": preproc_results,
            "board_result": None,
            "hole_result": None,
        }

    # 检测安装孔
    hole_result = detect_all_holes(board_result.data["warped_image"])

    # 几何测量
    board_size_px = board_result.data["board_size_px"]
    measurement = run_measurement(hole_result, board_size_px)

    remark = ""
    if not board_result.success:
        remark += f" Board: {board_result.message}"
    if not hole_result.success:
        remark += f" Holes: {hole_result.message}"

    return {
        "image_name": image_name,
        "method": "baseline",
        "board_detect_success": board_result.success,
        "holes_detect_success": hole_result.success,
        **measurement,
        "remark": remark.strip(),
        "preproc_results": preproc_results,
        "board_result": board_result,
        "hole_result": hole_result,
    }


def process_single_image_light_corrected(
    image_path: Path,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """
    使用光照矫正预处理流程处理单张 PCB 图像（B 组）。

    Args:
        image_path: PCB 图像文件路径
        camera_params: 相机标定参数

    Returns:
        包含测量结果和中间图像的 dict
    """
    image_name = image_path.stem
    print(f"\n{'─'*50}")
    print(f"  [B 组 - 光照矫正预处理] 处理图像: {image_name}")

    # 加载图像
    image = load_image(image_path)
    if image is None:
        return None

    # 光照矫正预处理（创新算法）
    preproc_results = preprocess_light_corrected(image, camera_params)

    if config.SAVE_INTERMEDIATE_IMAGES:
        save_light_corrected_steps(preproc_results, image_name)

    # 检测电路板外框（与 A 组使用相同的 detect_board 模块）
    board_result = detect_board(
        preproc_results["corrected"],
        preproc_results["binary"],
        preproc_results["edge"],
    )

    if not board_result.success:
        print(f"  [FAIL] 电路板检测失败: {board_result.message}")
        return {
            "image_name": image_name,
            "method": "light_corrected",
            "board_detect_success": False,
            "holes_detect_success": False,
            "pitch_x_mm": np.nan,
            "pitch_y_mm": np.nan,
            "abs_error_x_mm": np.nan,
            "abs_error_y_mm": np.nan,
            "rel_error_x_pct": np.nan,
            "rel_error_y_pct": np.nan,
            "mean_pitch_mm": np.nan,
            "mean_abs_error_mm": np.nan,
            "remark": f"电路板检测失败: {board_result.message}",
            "preproc_results": preproc_results,
            "board_result": None,
            "hole_result": None,
        }

    # 检测安装孔（与 A 组使用相同的 detect_holes 模块）
    hole_result = detect_all_holes(board_result.data["warped_image"])

    # 几何测量（与 A 组使用相同的 measure_geometry 模块）
    board_size_px = board_result.data["board_size_px"]
    measurement = run_measurement(hole_result, board_size_px)

    remark = ""
    if not board_result.success:
        remark += f" Board: {board_result.message}"
    if not hole_result.success:
        remark += f" Holes: {hole_result.message}"

    return {
        "image_name": image_name,
        "method": "light_corrected",
        "board_detect_success": board_result.success,
        "holes_detect_success": hole_result.success,
        **measurement,
        "remark": remark.strip(),
        "preproc_results": preproc_results,
        "board_result": board_result,
        "hole_result": hole_result,
    }


# ============================================================
# 对比实验主流程
# ============================================================

def run_comparison() -> None:
    """
    执行完整对比实验：
    1. 加载相机标定参数
    2. 对每张 PCB 图像分别运行 A 组和 B 组流程
    3. 保存中间结果和检测可视化
    4. 输出 CSV 汇总
    """
    print("=" * 60)
    print("  对比实验")
    print("  A 组：普通预处理")
    print("  B 组：光照矫正预处理")
    print("=" * 60)

    # 确保输出目录存在
    config.ensure_output_dirs()

    # 加载相机标定参数
    camera_params = load_camera_params()
    if camera_params[0] is not None:
        print("[INFO] 已加载相机标定参数")
    else:
        print("[WARNING] 未找到相机标定参数，将跳过畸变校正")

    # 加载 PCB 图像
    board_paths = get_image_paths(config.BOARD_IMAGE_DIR)
    if len(board_paths) == 0:
        print(f"[ERROR] 未找到 PCB 图像，请将图像放入: {config.BOARD_IMAGE_DIR}")
        return

    print(f"\n[INFO] 找到 {len(board_paths)} 张 PCB 待测图像")

    # 存储所有测量结果
    all_results = []

    for i, path in enumerate(board_paths):
        print(f"\n{'='*50}")
        print(f"  处理图像 {i+1}/{len(board_paths)}: {path.name}")
        print(f"{'='*50}")

        # A 组：普通预处理
        result_a = process_single_image_baseline(path, camera_params)
        if result_a is not None:
            all_results.append(result_a)

        # B 组：光照矫正预处理
        result_b = process_single_image_light_corrected(path, camera_params)
        if result_b is not None:
            all_results.append(result_b)

    # -------------------------------------------------------
    # 输出 CSV
    # -------------------------------------------------------
    if len(all_results) == 0:
        print("[ERROR] 无有效处理结果")
        return

    # 提取用于 CSV 的字段（排除图像数组）
    csv_fields = [
        "image_name", "method", "board_detect_success", "holes_detect_success",
        "pitch_x_mm", "pitch_y_mm",
        "abs_error_x_mm", "abs_error_y_mm",
        "rel_error_x_pct", "rel_error_y_pct",
        "mean_pitch_mm", "mean_abs_error_mm",
        "remark",
    ]

    df_all = pd.DataFrame([{k: r.get(k) for k in csv_fields} for r in all_results])

    # 分别保存两组数据
    df_baseline = df_all[df_all["method"] == "baseline"].copy()
    df_light = df_all[df_all["method"] == "light_corrected"].copy()

    # 基线组
    baseline_csv = config.COMPARISON_OUTPUT_DIR / "measurements_baseline.csv"
    df_baseline.to_csv(baseline_csv, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 基线组测量结果已保存: {baseline_csv}")

    # 改进组
    light_csv = config.COMPARISON_OUTPUT_DIR / "measurements_light_corrected.csv"
    df_light.to_csv(light_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 光照矫正组测量结果已保存: {light_csv}")

    # -------------------------------------------------------
    # 汇总对比表
    # -------------------------------------------------------
    # 分别统计两组的汇总数据
    baseline_measurements = [
        {k: r[k] for k in csv_fields if k in r and r.get("board_detect_success")
         and r.get("holes_detect_success")}
        for r in all_results if r["method"] == "baseline"
    ]
    light_measurements = [
        {k: r[k] for k in csv_fields if k in r and r.get("board_detect_success")
         and r.get("holes_detect_success")}
        for r in all_results if r["method"] == "light_corrected"
    ]

    summary_baseline = summarize_measurements(baseline_measurements)
    summary_light = summarize_measurements(light_measurements)

    # 构造对比 summary
    comparison_rows = []
    for key in sorted(summary_baseline.keys()):
        if key == "num_images":
            continue
        v_base = summary_baseline.get(key, np.nan)
        v_light = summary_light.get(key, np.nan)
        comparison_rows.append({
            "metric": key,
            "baseline": v_base,
            "light_corrected": v_light,
            "improvement": v_base - v_light if not np.isnan(v_base) and not np.isnan(v_light) else np.nan,
        })

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv = config.COMPARISON_OUTPUT_DIR / "comparison_summary.csv"
    df_comp.to_csv(comp_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 对比汇总已保存: {comp_csv}")

    # -------------------------------------------------------
    # 打印简要结果
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  对比实验完成")
    print("=" * 60)

    # 计算有效检测率
    n_baseline = len(df_baseline)
    n_light = len(df_light)
    n_baseline_ok = df_baseline["holes_detect_success"].sum() if n_baseline > 0 else 0
    n_light_ok = df_light["holes_detect_success"].sum() if n_light > 0 else 0

    print(f"\n  基线组 (Baseline):")
    print(f"    总图像数:           {n_baseline}")
    print(f"    板检测成功率:       {df_baseline['board_detect_success'].sum()}/{n_baseline}" if n_baseline > 0 else "    无数据")
    print(f"    孔检测成功率:       {n_baseline_ok}/{n_baseline}" if n_baseline > 0 else "    无数据")

    print(f"\n  光照矫正组 (Light Corrected):")
    print(f"    总图像数:           {n_light}")
    print(f"    板检测成功率:       {df_light['board_detect_success'].sum()}/{n_light}" if n_light > 0 else "    无数据")
    print(f"    孔检测成功率:       {n_light_ok}/{n_light}" if n_light > 0 else "    无数据")

    if "mean_abs_error_mm_mean" in summary_baseline and "mean_abs_error_mm_mean" in summary_light:
        b_err = summary_baseline["mean_abs_error_mm_mean"]
        l_err = summary_light["mean_abs_error_mm_mean"]
        print(f"\n  平均孔距绝对误差:")
        print(f"    基线组:      {b_err:.4f} mm")
        print(f"    光照矫正组:  {l_err:.4f} mm")
        improvement = (b_err - l_err) / b_err * 100 if b_err > 0 else 0
        print(f"    改进幅度:    {improvement:.2f}%")

    print(f"\n  输出文件位于: {config.COMPARISON_OUTPUT_DIR}")
