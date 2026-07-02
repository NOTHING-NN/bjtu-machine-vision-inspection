"""
main.py — 项目命令行入口

用法：
  python main.py calibrate            # 相机标定
  python main.py run_baseline         # 仅运行普通预处理（调试验证用）
  python main.py run_light_corrected  # 仅运行光照矫正预处理（调试验证用）
  python main.py compare              # 完整对比实验

说明：
  - calibrate 必须先执行（或确保已有标定参数文件）
  - compare 会自动运行两组预处理并对结果进行比较
"""

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("用法:")
        print("  python main.py calibrate            — 相机标定")
        print("  python main.py run_baseline         — 运行普通预处理")
        print("  python main.py run_light_corrected  — 运行光照矫正预处理")
        print("  python main.py compare              — 完整对比实验")
        sys.exit(1)

    command = sys.argv[1]

    if command == "calibrate":
        from src.calibrate_camera import run_calibration
        run_calibration()

    elif command == "run_baseline":
        from src.calibrate_camera import load_camera_params
        from src.preprocess_baseline import preprocess_baseline, save_baseline_steps
        from src.utils import get_image_paths, load_image
        from src import config

        config.ensure_output_dirs()
        camera_params = load_camera_params()

        board_paths = get_image_paths(config.BOARD_IMAGE_DIR)
        print(f"找到 {len(board_paths)} 张 PCB 图像")

        for path in board_paths:
            print(f"\n处理: {path.name}")
            img = load_image(path)
            if img is None:
                continue
            results = preprocess_baseline(img, camera_params)
            save_baseline_steps(results, path.stem)

    elif command == "run_light_corrected":
        from src.calibrate_camera import load_camera_params
        from src.preprocess_light_correct import (
            preprocess_light_corrected,
            save_light_corrected_steps,
        )
        from src.utils import get_image_paths, load_image
        from src import config

        config.ensure_output_dirs()
        camera_params = load_camera_params()

        board_paths = get_image_paths(config.BOARD_IMAGE_DIR)
        print(f"找到 {len(board_paths)} 张 PCB 图像")

        for path in board_paths:
            print(f"\n处理: {path.name}")
            img = load_image(path)
            if img is None:
                continue
            results = preprocess_light_corrected(img, camera_params)
            save_light_corrected_steps(results, path.stem)

    elif command == "compare":
        from src.compare_experiment import run_comparison
        run_comparison()

    else:
        print(f"未知命令: {command}")
        print("支持的命令: calibrate, run_baseline, run_light_corrected, compare")
        sys.exit(1)


if __name__ == "__main__":
    # 将项目根目录添加到 sys.path，确保 src 包可导入
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    main()
