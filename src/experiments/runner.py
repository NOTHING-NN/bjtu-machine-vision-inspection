"""
runner.py — 统一实验执行引擎

根据 ExperimentConfig 选择预处理策略、mask 生成方式和
是否启用区域分离，执行完整的 PCB 检测 → 孔检测 → 测量流程。

支持：
  - run_experiment()   — 批量运行实验
  - process_single()   — 单张图像处理
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src import config
from src.utils import get_image_paths, load_image
from src.calibration import load_camera_params, undistort_image
from src.preprocessing import preprocess_baseline, preprocess_light_corrected
from src.preprocessing.masks import segment_pcb_green
from src.board_detection import detect_board, separate_highlight_from_mask
from src.measurement import detect_all_holes, run_measurement, summarize_measurements


# ============================================================
# 单张处理
# ============================================================

def process_single(
    image_path: Path,
    exp_config,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """
    按照实验配置处理单张 PCB 图像。

    Args:
        image_path:    PCB 图像文件路径
        exp_config:    ExperimentConfig 实例
        camera_params: (camera_matrix, dist_coeffs) 或 (None, None)

    Returns:
        包含测量结果和中间产物的 dict
    """
    image_name = image_path.stem
    sample_type = config.SAMPLE_TYPE_MAP.get(image_name, "?")
    method = exp_config.name

    print(f"\n{'─'*50}")
    print(f"  [{method}] 处理图像: {image_name}  (类型 {sample_type})")

    # 加载图像
    image = load_image(image_path)
    if image is None:
        return _failure_result(image_name, sample_type, method, "图像加载失败")

    # ---- 预处理 ----
    if exp_config.preprocessing == "baseline":
        preproc_results = preprocess_baseline(image, camera_params)
        corrected = preproc_results["corrected"]
    elif exp_config.preprocessing == "illumination_corrected":
        preproc_results = preprocess_light_corrected(image, camera_params)
        corrected = preproc_results["corrected"]
    else:
        return _failure_result(image_name, sample_type, method,
                               f"未知预处理: {exp_config.preprocessing}")

    # ---- 板检测 ----
    # 生成 green_mask 并可选地应用区域分离
    green_mask = segment_pcb_green(corrected)

    separation_applied = False
    if exp_config.region_separation:
        highlight_mask = preproc_results.get("highlight_mask")
        if highlight_mask is not None and highlight_mask.max() > 0:
            h, w = corrected.shape[:2]
            green_mask = separate_highlight_from_mask(
                green_mask, highlight_mask, (h, w))
            separation_applied = True

    board_result = detect_board(corrected, mask_override=green_mask)

    if not board_result.success:
        return _failure_result(
            image_name, sample_type, method,
            f"电路板检测失败: {board_result.message}",
            preproc_results=preproc_results,
            board_result=board_result,
            separation_applied=separation_applied,
        )

    # ---- 孔检测 ----
    warped_gray = board_result.data["warped_gray"]
    board_size_px = board_result.data["board_size_px"]
    warped_color = board_result.data["warped_image"]

    hole_result = detect_all_holes(warped_gray, board_size_px, warped_color)

    # ---- 几何测量 ----
    measurement = run_measurement(hole_result, board_size_px)

    remark_parts = []
    if not board_result.success:
        remark_parts.append(f"Board: {board_result.message}")
    if not hole_result.success:
        remark_parts.append(f"Holes: {hole_result.message}")
    if separation_applied:
        remark_parts.append("region_separation=on")

    return {
        "image_name": image_name,
        "sample_type": sample_type,
        "method": method,
        "board_detect_success": board_result.success,
        "holes_detect_success": hole_result.success,
        "num_holes": sum(hole_result.data.get("individual_success", [False] * 4)),
        **measurement,
        "remark": " | ".join(remark_parts).strip(),
        "preproc_results": preproc_results,
        "board_result": board_result,
        "hole_result": hole_result,
        "separation_applied": separation_applied,
        "green_mask": green_mask,
    }


def _failure_result(image_name: str, sample_type: str, method: str,
                    message: str,
                    preproc_results: Dict = None,
                    board_result=None,
                    separation_applied: bool = False) -> Dict:
    """构造检测失败时的结果 dict。"""
    remark = message
    if separation_applied:
        remark += " | region_separation=on"

    return {
        "image_name": image_name,
        "sample_type": sample_type,
        "method": method,
        "board_detect_success": False,
        "holes_detect_success": False,
        "num_holes": 0,
        "pitch_x_mm": np.nan,
        "pitch_y_mm": np.nan,
        "abs_error_x_mm": np.nan,
        "abs_error_y_mm": np.nan,
        "rel_error_x_pct": np.nan,
        "rel_error_y_pct": np.nan,
        "mean_pitch_mm": np.nan,
        "mean_abs_error_mm": np.nan,
        "remark": remark,
        "preproc_results": preproc_results or {},
        "board_result": board_result,
        "hole_result": None,
        "separation_applied": separation_applied,
        "green_mask": None,
    }


# ============================================================
# 批量实验
# ============================================================

def run_experiment(
    exp_config,
    image_paths: List[Path],
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]],
) -> List[Dict]:
    """
    按指定配置批量运行实验。

    Args:
        exp_config:    ExperimentConfig 实例
        image_paths:   PCB 图像路径列表
        camera_params: 标定参数

    Returns:
        所有图像的处理结果列表
    """
    print(f"\n{'='*60}")
    print(f"  实验: {exp_config.name}")
    print(f"  预处理: {exp_config.preprocessing}")
    print(f"  Mask:   {exp_config.board_mask}")
    print(f"  区域分离: {'ON' if exp_config.region_separation else 'OFF'}")
    print(f"{'='*60}")

    results = []
    for i, path in enumerate(image_paths):
        print(f"\n  [{i+1}/{len(image_paths)}] {path.name}")
        result = process_single(path, exp_config, camera_params)
        if result is not None:
            results.append(result)

    return results


# ============================================================
# CSV 输出
# ============================================================

def save_results_csv(results: List[Dict], output_path: Path) -> pd.DataFrame:
    """将实验结果保存为 CSV 并返回 DataFrame。"""
    csv_fields = [
        "image_name", "sample_type", "method",
        "board_detect_success", "holes_detect_success", "num_holes",
        "pitch_x_mm", "pitch_y_mm",
        "abs_error_x_mm", "abs_error_y_mm",
        "rel_error_x_pct", "rel_error_y_pct",
        "mean_pitch_mm", "mean_abs_error_mm",
        "remark",
    ]

    df = pd.DataFrame([{k: r.get(k) for k in csv_fields} for r in results])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 结果已保存: {output_path}")
    return df


def print_grouped_summary(all_results: List[Dict]) -> None:
    """按样本类型分组打印汇总统计。"""
    # 分离有效测量
    valid = [r for r in all_results
             if r.get("board_detect_success") and r.get("holes_detect_success")]

    def stats_for(subset):
        if not subset:
            return None
        errors = [r["mean_abs_error_mm"] for r in subset]
        arr = np.array(errors)
        n_total = len(subset)
        n_ok = len([r for r in subset if r.get("board_detect_success")])
        return {
            "n_images": n_total,
            "n_board_ok": n_ok,
            "n_holes_ok": len(errors),
            "mean_error": float(np.mean(arr)),
            "std_error": float(np.std(arr, ddof=1)) if len(errors) > 1 else 0.0,
            "min_error": float(np.min(arr)),
            "max_error": float(np.max(arr)),
        }

    for method in ["baseline", "light_corrected"]:
        method_results = [r for r in all_results if r.get("method") == method]
        print(f"\n{'─'*50}")
        print(f"  [{method}]")
        for stype, label in [("A", "板面反光"), ("B", "邻近强光斑")]:
            subset = [r for r in method_results if r.get("sample_type") == stype]
            s = stats_for(subset)
            if s:
                print(f"  {stype} 类 ({label}): "
                      f"板检测 {s['n_board_ok']}/{s['n_images']}, "
                      f"孔检测 {s['n_holes_ok']}/{s['n_images']}, "
                      f"误差 mean={s['mean_error']:.3f} mm, "
                      f"std={s['std_error']:.3f} mm")
        # 全量
        all_s = stats_for(method_results)
        if all_s:
            print(f"  全部: "
                  f"板检测 {all_s['n_board_ok']}/{all_s['n_images']}, "
                  f"误差 mean={all_s['mean_error']:.3f} mm")
