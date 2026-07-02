"""
main.py — 项目命令行入口

用法：
  python main.py calibrate            # 相机标定
  python main.py baseline             # 运行 Baseline 实验
  python main.py light_corrected      # 运行光照矫正实验
  python main.py compare              # 对比实验 + 按 A/B 分组报告

说明：
  - calibrate 必须先执行（或确保已有标定参数文件）
  - compare 自动运行两组实验并对结果进行比较
"""

import sys
import io
from pathlib import Path

# 强制 stdout 使用 UTF-8，避免 Windows GBK 终端下中文乱码
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("用法:")
        print("  python main.py calibrate            — 相机标定")
        print("  python main.py baseline             — 运行 Baseline 实验")
        print("  python main.py light_corrected      — 运行光照矫正实验")
        print("  python main.py compare              — 完整对比实验 + 分组报告")
        sys.exit(1)

    command = sys.argv[1]

    if command == "calibrate":
        from src.calibration import run_calibration
        run_calibration()

    elif command == "baseline":
        from src.calibration import load_camera_params
        from src.experiments import (
            BASELINE_CONFIG, run_experiment, save_results_csv, print_grouped_summary
        )
        from src.utils import get_image_paths
        from src import config

        config.ensure_output_dirs()
        camera_params = load_camera_params()
        paths = get_image_paths(config.BOARD_IMAGE_DIR)
        if len(paths) == 0:
            print(f"[ERROR] 未找到 PCB 图像: {config.BOARD_IMAGE_DIR}")
            sys.exit(1)

        results = run_experiment(BASELINE_CONFIG, paths, camera_params)
        save_results_csv(results,
                         config.COMPARISON_OUTPUT_DIR / "measurements_baseline.csv")
        print_grouped_summary(results)

    elif command == "light_corrected":
        from src.calibration import load_camera_params
        from src.experiments import (
            LIGHT_CORRECTED_CONFIG, run_experiment, save_results_csv, print_grouped_summary
        )
        from src.utils import get_image_paths
        from src import config

        config.ensure_output_dirs()
        camera_params = load_camera_params()
        paths = get_image_paths(config.BOARD_IMAGE_DIR)
        if len(paths) == 0:
            print(f"[ERROR] 未找到 PCB 图像: {config.BOARD_IMAGE_DIR}")
            sys.exit(1)

        results = run_experiment(LIGHT_CORRECTED_CONFIG, paths, camera_params)
        save_results_csv(results,
                         config.COMPARISON_OUTPUT_DIR / "measurements_light_corrected.csv")
        print_grouped_summary(results)

    elif command == "compare":
        import numpy as np
        import pandas as pd
        from src.calibration import load_camera_params
        from src.experiments import (
            BASELINE_CONFIG, LIGHT_CORRECTED_CONFIG,
            run_experiment, save_results_csv, print_grouped_summary,
            generate_report_figures,
        )
        from src.utils import get_image_paths, load_image
        from src import config

        config.ensure_output_dirs()
        camera_params = load_camera_params()
        paths = get_image_paths(config.BOARD_IMAGE_DIR)
        if len(paths) == 0:
            print(f"[ERROR] 未找到 PCB 图像: {config.BOARD_IMAGE_DIR}")
            sys.exit(1)

        print(f"\n[INFO] 找到 {len(paths)} 张 PCB 待测图像")

        # A 组：Baseline
        baseline_results = run_experiment(BASELINE_CONFIG, paths, camera_params)
        save_results_csv(baseline_results,
                         config.COMPARISON_OUTPUT_DIR / "measurements_baseline.csv")

        # B 组：Light Corrected
        light_results = run_experiment(LIGHT_CORRECTED_CONFIG, paths, camera_params)
        save_results_csv(light_results,
                         config.COMPARISON_OUTPUT_DIR / "measurements_light_corrected.csv")

        # 逐图生成报告
        for i, path in enumerate(paths):
            name = path.stem
            original = load_image(path)
            b_result = baseline_results[i] if i < len(baseline_results) else None
            l_result = light_results[i] if i < len(light_results) else None
            if original is not None:
                generate_report_figures(name, original, b_result, l_result)

        # 合并所有结果
        all_results = baseline_results + light_results

        # 分组统计 — A 类 vs B 类
        print("\n" + "=" * 60)
        print("  对比实验完成 — 分组统计")
        print("=" * 60)

        print_grouped_summary(all_results)

        # 汇总对比 CSV
        valid_b = [r for r in baseline_results
                   if r.get("board_detect_success") and r.get("holes_detect_success")]
        valid_l = [r for r in light_results
                   if r.get("board_detect_success") and r.get("holes_detect_success")]

        from src.measurement import summarize_measurements
        summary_b = summarize_measurements(valid_b)
        summary_l = summarize_measurements(valid_l)

        comparison_rows = []
        for key in sorted(summary_b.keys()):
            if key == "num_images":
                continue
            v_base = summary_b.get(key, np.nan)
            v_light = summary_l.get(key, np.nan)
            comparison_rows.append({
                "metric": key,
                "baseline": v_base,
                "light_corrected": v_light,
            })

        df_comp = pd.DataFrame(comparison_rows)
        comp_csv = config.COMPARISON_OUTPUT_DIR / "comparison_summary.csv"
        df_comp.to_csv(comp_csv, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 对比汇总已保存: {comp_csv}")

        # 简要结果
        n_base = len(baseline_results)
        n_light = len(light_results)
        n_base_ok = len(valid_b)
        n_light_ok = len(valid_l)

        print(f"\n  基线组 (Baseline):")
        print(f"    板检测成功: {sum(1 for r in baseline_results if r['board_detect_success'])}/{n_base}")
        print(f"    孔检测成功: {n_base_ok}/{n_base}")

        print(f"\n  光照矫正组 (Light Corrected):")
        print(f"    板检测成功: {sum(1 for r in light_results if r['board_detect_success'])}/{n_light}")
        print(f"    孔检测成功: {n_light_ok}/{n_light}")

        if summary_b.get("mean_abs_error_mm_mean") and summary_l.get("mean_abs_error_mm_mean"):
            b_err = summary_b["mean_abs_error_mm_mean"]
            l_err = summary_l["mean_abs_error_mm_mean"]
            print(f"\n  平均孔距绝对误差:")
            print(f"    基线组:      {b_err:.4f} mm")
            print(f"    光照矫正组:  {l_err:.4f} mm")

        print(f"\n  输出文件位于: {config.COMPARISON_OUTPUT_DIR}")

    else:
        print(f"未知命令: {command}")
        print("支持的命令: calibrate, baseline, light_corrected, compare")
        sys.exit(1)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    main()
