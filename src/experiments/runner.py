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
from src.utils import load_image
from src.calibration import load_camera_params
from src.preprocessing import preprocess_standard, preprocess_highlight_aware
from src.preprocessing.masks import segment_pcb_green
from src.board_detection import (
    detect_board,
    recover_quad_from_mask,
    separate_highlight_from_mask,
)
from src.measurement import detect_all_holes, run_measurement
from src.experiments.configs import SAMPLE_TYPE_MAP


# ============================================================
# 单张处理
# ============================================================

def process_single(
    image_path: Path,
    exp_config,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]],
    image_to_world: Optional[np.ndarray] = None,
) -> Dict:
    """
    按照实验配置处理单张 PCB 图像。

    Args:
        image_path:    PCB 图像文件路径
        exp_config:    ExperimentConfig 实例
        camera_params: (camera_matrix, dist_coeffs) 或 (None, None)
        image_to_world: 去畸变图像像素到测量平面毫米坐标的单应性

    Returns:
        包含测量结果和中间产物的 dict
    """
    image_name = image_path.stem
    sample_type = SAMPLE_TYPE_MAP.get(image_name, "?")
    method = exp_config.name

    print(f"\n{'─'*50}")
    print(f"  [{method}] 处理图像: {image_name}  (类型 {sample_type})")

    # 加载图像
    image = load_image(image_path)
    if image is None:
        return _failure_result(image_name, sample_type, method, "图像加载失败")

    # ---- 预处理 ----
    if exp_config.preprocessing == "standard":
        preproc_results = preprocess_standard(image, camera_params)
        corrected = preproc_results["corrected"]
    elif exp_config.preprocessing == "highlight_aware":
        preproc_results = preprocess_highlight_aware(image, camera_params)
        corrected = preproc_results["corrected"]
    else:
        return _failure_result(image_name, sample_type, method,
                               f"未知预处理: {exp_config.preprocessing}")

    # ---- 板检测 ----
    # 生成 green_mask 并可选地应用区域分离
    green_mask = segment_pcb_green(corrected)
    measurement_image = preproc_results.get("undistorted", corrected)

    # 记录分离前连通域数量
    n_components_before = _count_connected_components(green_mask)

    separation_applied = False
    separation_modified = False
    if exp_config.region_separation:
        highlight_mask = preproc_results.get("highlight_mask")
        if highlight_mask is not None and highlight_mask.max() > 0:
            h, w = corrected.shape[:2]
            green_mask, separation_modified = separate_highlight_from_mask(
                green_mask, highlight_mask, (h, w))
            separation_applied = True

    # 记录分离后连通域数量
    n_components_after = _count_connected_components(green_mask)

    board_result = detect_board(measurement_image, mask_override=green_mask)
    quad_recovery_used = False

    if (not board_result.success) and exp_config.region_separation:
        recovered = recover_quad_from_mask(green_mask)
        if recovered is not None:
            corners, score, recovery_diag = recovered
            board_result = detect_board(
                measurement_image,
                mask_override=green_mask,
                candidates_override=[(corners, score)],
            )
            if board_result.success:
                quad_recovery_used = True
                board_result.data["quad_recovery"] = recovery_diag

    if not board_result.success:
        return _failure_result(
            image_name, sample_type, method,
            f"电路板检测失败: {board_result.message}",
            preproc_results=preproc_results,
            board_result=board_result,
            separation_applied=separation_applied,
            num_components_before=n_components_before,
            num_components_after=n_components_after,
            separation_modified=separation_modified,
        )

    # ---- 孔检测 ----
    warped_gray = board_result.data["warped_gray"]
    board_size_px = board_result.data["board_size_px"]
    warped_color = board_result.data["warped_image"]

    hole_result = detect_all_holes(warped_gray, board_size_px, warped_color)

    # ---- 几何测量 ----
    measurement = run_measurement(board_result, hole_result, image_to_world)

    remark_parts = []
    if not hole_result.success:
        remark_parts.append(f"Holes: {hole_result.message}")
    if separation_modified:
        remark_parts.append("region_separation=modified")
    elif separation_applied:
        remark_parts.append("region_separation=attempted")
    if quad_recovery_used:
        remark_parts.append("quad_recovery=row_interval")

    return {
        "image_name": image_name,
        "sample_type": sample_type,
        "method": method,
        "board_detect_success": board_result.success,
        "holes_detect_success": hole_result.success,
        "num_holes": sum(hole_result.data.get("individual_success", [False] * 4)),
        **measurement,
        "remark": " | ".join(remark_parts).strip(),
        # 统计字段
        "board_score": board_result.data.get("score") if board_result.success else None,
        "num_components_before": n_components_before,
        "num_components_after": n_components_after,
        "separation_modified": separation_modified,
        "quad_recovery_used": quad_recovery_used,
        # 中间产物
        "preproc_results": preproc_results,
        "board_result": board_result,
        "hole_result": hole_result,
        "separation_applied": separation_applied,
        "green_mask": green_mask,
    }


def _count_connected_components(mask: np.ndarray) -> int:
    """计算二值 mask 中的连通域数量（不含背景）。"""
    if mask is None:
        return 0
    _, labels = cv2.connectedComponents(
        (mask > 0).astype(np.uint8), connectivity=8)
    return max(0, labels.max() - 1)  # 减去背景 label 0


def _failure_result(image_name: str, sample_type: str, method: str,
                    message: str,
                    preproc_results: Dict = None,
                    board_result=None,
                    separation_applied: bool = False,
                    num_components_before: int = 0,
                    num_components_after: int = 0,
                    separation_modified: bool = False) -> Dict:
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
        "board_width_top_mm": np.nan,
        "board_width_bottom_mm": np.nan,
        "board_height_left_mm": np.nan,
        "board_height_right_mm": np.nan,
        "board_width_mm": np.nan,
        "board_height_mm": np.nan,
        "board_width_error_mm": np.nan,
        "board_height_error_mm": np.nan,
        "resolution_x_mm_per_px": np.nan,
        "resolution_y_mm_per_px": np.nan,
        "resolution_mean_mm_per_px": np.nan,
        "resolution_x_px_per_mm": np.nan,
        "resolution_y_px_per_mm": np.nan,
        "resolution_mean_px_per_mm": np.nan,
        "resolution_mean_um_per_px": np.nan,
        "hole_diameters_mm": [np.nan] * 4,
        "hole_centers_world_mm": [(np.nan, np.nan)] * 4,
        "hole1_x_mm": np.nan,
        "hole1_y_mm": np.nan,
        "hole1_diameter_mm": np.nan,
        "hole2_x_mm": np.nan,
        "hole2_y_mm": np.nan,
        "hole2_diameter_mm": np.nan,
        "hole3_x_mm": np.nan,
        "hole3_y_mm": np.nan,
        "hole3_diameter_mm": np.nan,
        "hole4_x_mm": np.nan,
        "hole4_y_mm": np.nan,
        "hole4_diameter_mm": np.nan,
        "measurement_valid": False,
        "measurement_message": message,
        "remark": remark,
        "board_score": None,
        "num_components_before": num_components_before,
        "num_components_after": num_components_after,
        "separation_modified": separation_modified,
        "quad_recovery_used": False,
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
    image_to_world: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    按指定配置批量运行实验。

    Args:
        exp_config:    ExperimentConfig 实例
        image_paths:   PCB 图像路径列表
        camera_params: 标定参数
        image_to_world: 去畸变图像像素到测量平面毫米坐标的单应性

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
        result = process_single(path, exp_config, camera_params, image_to_world)
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
        "measurement_valid", "measurement_message",
        "board_width_mm", "board_height_mm",
        "board_width_error_mm", "board_height_error_mm",
        "resolution_x_mm_per_px", "resolution_y_mm_per_px",
        "resolution_mean_mm_per_px",
        "resolution_x_px_per_mm", "resolution_y_px_per_mm",
        "resolution_mean_px_per_mm", "resolution_mean_um_per_px",
        "board_width_top_mm", "board_width_bottom_mm",
        "board_height_left_mm", "board_height_right_mm",
        "pitch_x_mm", "pitch_y_mm",
        "abs_error_x_mm", "abs_error_y_mm",
        "rel_error_x_pct", "rel_error_y_pct",
        "mean_pitch_mm", "mean_abs_error_mm",
        "hole1_x_mm", "hole1_y_mm", "hole1_diameter_mm",
        "hole2_x_mm", "hole2_y_mm", "hole2_diameter_mm",
        "hole3_x_mm", "hole3_y_mm", "hole3_diameter_mm",
        "hole4_x_mm", "hole4_y_mm", "hole4_diameter_mm",
        "board_score",
        "separation_applied", "separation_modified",
        "quad_recovery_used",
        "num_components_before", "num_components_after",
        "remark",
    ]

    df = pd.DataFrame([{k: r.get(k) for k in csv_fields} for r in results])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 结果已保存: {output_path}")
    return df


def print_grouped_summary(all_results: List[Dict]) -> None:
    """按样本类型分组打印汇总统计。算法 B 仅统计 B 类样本。"""
    method_labels = {
        "algorithm_a": "算法 A (基础测量)",
        "algorithm_b": "算法 B (强光斑感知改进)",
    }

    def stats_for(subset):
        if not subset:
            return None
        n_total = len(subset)
        n_board_ok = len([r for r in subset if r.get("board_detect_success")])
        valid = [
            r for r in subset
            if r.get("board_detect_success")
            and r.get("holes_detect_success")
            and np.isfinite(r.get("mean_abs_error_mm", np.nan))
        ]
        errors = [r["mean_abs_error_mm"] for r in valid]
        arr = np.array(errors, dtype=float)
        return {
            "n_images": n_total,
            "n_board_ok": n_board_ok,
            "n_holes_ok": len(valid),
            "mean_error": float(np.mean(arr)) if len(errors) else np.nan,
            "std_error": float(np.std(arr, ddof=1)) if len(errors) > 1 else 0.0,
            "min_error": float(np.min(arr)) if len(errors) else np.nan,
            "max_error": float(np.max(arr)) if len(errors) else np.nan,
        }

    for method in ["algorithm_a", "algorithm_b"]:
        method_results = [r for r in all_results if r.get("method") == method]
        if not method_results:
            continue

        print(f"\n{'─'*50}")
        print(f"  [{method_labels.get(method, method)}]")

        if method == "algorithm_a":
            # 算法 A：A 类 + B 类
            for stype, label in [("A", "板面反光"), ("B", "邻近强光斑")]:
                subset = [r for r in method_results if r.get("sample_type") == stype]
                s = stats_for(subset)
                if s:
                    print(f"  {stype} 类 ({label}): "
                          f"板检测 {s['n_board_ok']}/{s['n_images']}, "
                          f"孔检测 {s['n_holes_ok']}/{s['n_images']}, "
                          f"误差 mean={s['mean_error']:.3f} mm, "
                          f"std={s['std_error']:.3f} mm")
        else:
            # 算法 B：仅 B 类
            subset = [r for r in method_results if r.get("sample_type") == "B"]
            s = stats_for(subset)
            if s:
                quad_count = sum(1 for r in subset if r.get("quad_recovery_used"))
                print(f"  B 类 (邻近强光斑): "
                      f"板检测 {s['n_board_ok']}/{s['n_images']}, "
                      f"孔检测 {s['n_holes_ok']}/{s['n_images']}")
                if s['n_holes_ok'] > 0:
                    print(f"  误差 mean={s['mean_error']:.3f} mm, "
                          f"std={s['std_error']:.3f} mm")
                if quad_count > 0:
                    print(f"  四边形恢复: {quad_count}/{s['n_images']}")

        # 全量
        all_s = stats_for(method_results)
        if all_s:
            scope = "全部样本" if method == "algorithm_a" else "B 类样本"
            err_part = ""
            if all_s["n_holes_ok"] > 0:
                err_part = f", 误差 mean={all_s['mean_error']:.3f} mm"
            print(f"  [{scope}]: "
                  f"板检测 {all_s['n_board_ok']}/{all_s['n_images']}, "
                  f"孔检测 {all_s['n_holes_ok']}/{all_s['n_images']}"
                  f"{err_part}")
