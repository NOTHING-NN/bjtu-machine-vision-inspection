"""
main.py — 项目命令行入口

用法：
  python main.py calibrate            # 相机标定
  python main.py algorithm-a          # 运行算法 A (基础测量，全部样本)
  python main.py algorithm-b          # 运行算法 B (强光斑感知改进，仅 B 类样本)
  python main.py compare              # 对比实验 + 分组统计报告

说明：
  - calibrate 必须先执行（或确保已有标定参数文件）
  - 算法 A 在全部样本上运行；算法 B 仅在 B 类（强光斑邻接）样本上运行
  - 算法 A 与算法 B 使用完全相同的测量链路（board→holes→measure），
    仅在 PCB mask 生成与区域分离策略上有差异
"""

import sys
import io
from pathlib import Path
from typing import List

# 强制 stdout 使用 UTF-8，避免 Windows GBK 终端下中文乱码
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )


def _filter_b_samples(paths: List[Path]) -> List[Path]:
    """筛选 B 类样本（强光斑邻接），供算法 B 专用。"""
    from src.experiments.configs import SAMPLE_TYPE_MAP
    filtered = [p for p in paths if SAMPLE_TYPE_MAP.get(p.stem, "?") == "B"]
    if not filtered:
        print("[WARNING] 未找到 B 类样本，算法 B 跳过")
    return filtered


def _find_result_by_name(results: List[dict], name: str) -> dict:
    """按 image_name 查找结果 dict。"""
    for r in results:
        if r.get("image_name") == name:
            return r
    return None


def _load_measurement_context():
    """加载相机参数和棋盘格测量平面单应性。"""
    from src.calibration import load_camera_params, load_measurement_plane_homography

    camera_params = load_camera_params()
    image_to_world = load_measurement_plane_homography()
    if image_to_world is None:
        print("[ERROR] 缺少测量平面单应性，不能进行毫米尺寸测量。")
        print("        请先运行: python main.py calibrate")
    return camera_params, image_to_world


def main() -> None:
    if len(sys.argv) < 2:
        print("用法:")
        print("  python main.py calibrate         — 相机标定")
        print("  python main.py algorithm-a       — 运行算法 A (基础测量，全部样本)")
        print("  python main.py algorithm-b       — 运行算法 B (仅 B 类样本)")
        print("  python main.py compare           — 完整对比实验 + 分组报告")
        sys.exit(1)

    command = sys.argv[1]

    if command == "calibrate":
        from src.calibration import run_calibration
        run_calibration()

    elif command in ("algorithm-a", "baseline"):
        from src.experiments import (
            ALGORITHM_A_CONFIG, run_experiment, save_results_csv, print_grouped_summary
        )
        from src.utils import get_image_paths
        from src import config

        config.ensure_output_dirs()
        camera_params, image_to_world = _load_measurement_context()
        if image_to_world is None:
            sys.exit(1)
        paths = get_image_paths(config.PCB_IMAGE_DIR)
        if len(paths) == 0:
            print(f"[ERROR] 未找到 PCB 图像: {config.PCB_IMAGE_DIR}")
            sys.exit(1)

        results = run_experiment(ALGORITHM_A_CONFIG, paths, camera_params, image_to_world)
        save_results_csv(results,
                         config.REPORTS_OUTPUT_DIR / "measurements_algorithm_a.csv")
        print_grouped_summary(results)

    elif command in ("algorithm-b", "light_corrected"):
        from src.experiments import (
            ALGORITHM_B_CONFIG, run_experiment, save_results_csv, print_grouped_summary
        )
        from src.utils import get_image_paths
        from src import config

        config.ensure_output_dirs()
        camera_params, image_to_world = _load_measurement_context()
        if image_to_world is None:
            sys.exit(1)
        all_paths = get_image_paths(config.PCB_IMAGE_DIR)
        paths = _filter_b_samples(all_paths)
        if len(paths) == 0:
            print("[ERROR] 未找到 B 类样本，算法 B 无需运行")
            sys.exit(0)

        print(f"[INFO] 算法 B 仅处理 B 类样本: {[p.name for p in paths]}")
        results = run_experiment(ALGORITHM_B_CONFIG, paths, camera_params, image_to_world)
        save_results_csv(results,
                         config.REPORTS_OUTPUT_DIR / "measurements_algorithm_b.csv")
        print_grouped_summary(results)

    elif command == "compare":
        import numpy as np
        import pandas as pd
        from src.experiments import (
            ALGORITHM_A_CONFIG, ALGORITHM_B_CONFIG,
            run_experiment, save_results_csv,
            generate_report_figures,
        )
        from src.experiments.statistics import format_statistics_table
        from src.utils import get_image_paths, load_image
        from src import config

        config.ensure_output_dirs()
        camera_params, image_to_world = _load_measurement_context()
        if image_to_world is None:
            sys.exit(1)
        all_paths = get_image_paths(config.PCB_IMAGE_DIR)
        if len(all_paths) == 0:
            print(f"[ERROR] 未找到 PCB 图像: {config.PCB_IMAGE_DIR}")
            sys.exit(1)

        paths_b = _filter_b_samples(all_paths)

        print(f"\n[INFO] 全部待测图像: {len(all_paths)} 张")
        print(f"[INFO] 其中 B 类样本: {len(paths_b)} 张 — {[p.name for p in paths_b]}")

        # ---- 算法 A：全部样本 ----
        print(f"\n{'='*60}")
        print("  算法 A (基础测量) — 全部样本")
        print(f"{'='*60}")
        results_a = run_experiment(ALGORITHM_A_CONFIG, all_paths, camera_params, image_to_world)
        save_results_csv(results_a,
                         config.REPORTS_OUTPUT_DIR / "measurements_algorithm_a.csv")

        # ---- 算法 B：仅 B 类样本 ----
        results_b = []
        if paths_b:
            print(f"\n{'='*60}")
            print("  算法 B (强光斑感知改进) — 仅 B 类样本")
            print(f"{'='*60}")
            results_b = run_experiment(ALGORITHM_B_CONFIG, paths_b, camera_params, image_to_world)
            save_results_csv(results_b,
                             config.REPORTS_OUTPUT_DIR / "measurements_algorithm_b.csv")
        else:
            print("[INFO] 无 B 类样本，跳过算法 B")

        # ---- 逐图生成报告 ----
        for path in all_paths:
            name = path.stem
            original = load_image(path)
            r_a = _find_result_by_name(results_a, name)
            r_b = _find_result_by_name(results_b, name)
            if original is not None:
                generate_report_figures(name, original, r_a, r_b)

        # ---- 统计报告 ----
        all_results = results_a + results_b
        report_text = format_statistics_table(all_results)
        print(report_text)

        # ---- 汇总对比 CSV (只在 B 类样本上公平对比) ----
        valid_a_b = [r for r in results_a
                     if r.get("sample_type") == "B"
                     and r.get("board_detect_success")
                     and r.get("holes_detect_success")]
        valid_b = [r for r in results_b
                   if r.get("board_detect_success") and r.get("holes_detect_success")]

        from src.measurement import summarize_measurements
        summary_a_b = summarize_measurements(valid_a_b) if valid_a_b else {}
        summary_b = summarize_measurements(valid_b) if valid_b else {}

        comparison_rows = []
        all_metric_keys = sorted(set(summary_a_b.keys()) | set(summary_b.keys()))
        for key in all_metric_keys:
            if key == "num_images":
                continue
            v_a = summary_a_b.get(key, np.nan)
            v_b = summary_b.get(key, np.nan)
            comparison_rows.append({
                "metric": key,
                "algorithm_a": v_a,
                "algorithm_b": v_b,
            })

        df_comp = pd.DataFrame(comparison_rows)
        comp_csv = config.REPORTS_OUTPUT_DIR / "comparison_summary.csv"
        df_comp.to_csv(comp_csv, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 对比汇总已保存: {comp_csv}")

        # ---- 简要结论 ----
        n_a = len(results_a)
        n_b = len(results_b)
        n_a_ok_all = sum(1 for r in results_a
                         if r.get("board_detect_success") and r.get("holes_detect_success"))
        n_b_ok = len(valid_b)

        print(f"\n  算法 A (基础测量，全部 {n_a} 张):")
        print(f"    板检测成功: {sum(1 for r in results_a if r['board_detect_success'])}/{n_a}")
        print(f"    孔检测成功: {n_a_ok_all}/{n_a}")

        print(f"\n  算法 B (强光斑感知改进，仅 B 类 {n_b} 张):")
        print(f"    板检测成功: {sum(1 for r in results_b if r['board_detect_success'])}/{n_b}")
        print(f"    孔检测成功: {n_b_ok}/{n_b}")

        # 公平对比：仅在 B 类样本上
        if summary_a_b.get("mean_abs_error_mm_mean") and summary_b.get("mean_abs_error_mm_mean"):
            a_err = summary_a_b["mean_abs_error_mm_mean"]
            b_err = summary_b["mean_abs_error_mm_mean"]
            print(f"\n  平均孔距绝对误差 (仅 B 类样本对比):")
            print(f"    算法 A:  {'N/A (B 类全部失败)' if not valid_a_b else f'{a_err:.4f} mm'}")
            print(f"    算法 B:  {b_err:.4f} mm")

        # 板尺寸 (A 类有效数据)
        valid_a_all = [r for r in results_a
                       if r.get("board_detect_success") and r.get("holes_detect_success")]
        if valid_a_all:
            from src.measurement import summarize_measurements
            s_all = summarize_measurements(valid_a_all)
            bw = s_all.get("board_width_mm_mean")
            bh = s_all.get("board_height_mm_mean")
            bwe = s_all.get("board_width_error_mm_mean")
            bhe = s_all.get("board_height_error_mm_mean")
            if bw and bh:
                print(f"\n  板尺寸测量 (算法 A, 棋盘格测量平面):")
                print(f"    宽 (W): {bw:.3f} mm  (误差: {bwe:.3f} mm)")
                print(f"    高 (H): {bh:.3f} mm  (误差: {bhe:.3f} mm)")
            res = s_all.get("resolution_mean_mm_per_px_mean")
            density = s_all.get("resolution_mean_px_per_mm_mean")
            if res is not None and not np.isnan(res):
                print(f"\n  测量分辨率 (算法 A):")
                print(f"    平均: {res:.5f} mm/px  ({res * 1000.0:.2f} um/px)")
                print(f"    采样密度: {density:.2f} px/mm")

        # 孔直径
        if valid_a_all:
            dia_strs = []
            for i in range(1, 5):
                k = f"hole{i}_diameter_mm_mean"
                v = s_all.get(k)
                if v is not None and not np.isnan(v):
                    dia_strs.append(f"H{i}={v:.2f} mm")
            if dia_strs:
                print(f"\n  安装孔直径 (算法 A):")
                print(f"    {',  '.join(dia_strs)}")

        print(f"\n  输出文件位于: {config.REPORTS_OUTPUT_DIR}")

    else:
        print(f"未知命令: {command}")
        print("支持的命令: calibrate, algorithm-a, algorithm-b, compare")
        sys.exit(1)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    main()
