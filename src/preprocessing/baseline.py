"""
baseline.py — 普通预处理流程（对照组）

流水线：
  原图 → 畸变校正 → 灰度化/G通道 → 高斯滤波 → 自适应阈值 → Canny 边缘检测

所有步骤均为常规图像处理算法，不含光照矫正，
用于对照实验。
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.calibration import undistort_image
from src.utils import save_image, ensure_grayscale, extract_green_channel


def preprocess_baseline(
    image: np.ndarray,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    执行普通预处理全流程（对照组）。

    Args:
        image: 输入 BGR 图像
        camera_params: (camera_matrix, dist_coeffs) 元组，None 则跳过畸变校正

    Returns:
        包含各阶段结果图像的 dict:
            - "corrected":   畸变校正后的 BGR 图
            - "gray":        灰度图 / G 通道图
            - "blurred":     高斯滤波图
            - "binary":      二值化图
            - "edge":        Canny 边缘图
    """
    results = {}

    # 1. 畸变校正
    if camera_params is not None and camera_params[0] is not None:
        camera_matrix, dist_coeffs = camera_params
        corrected = undistort_image(image, camera_matrix, dist_coeffs)
    else:
        corrected = image.copy()
    results["corrected"] = corrected

    # 2. 灰度化 — 优先使用 G 通道（PCB 基板为绿色）
    try:
        gray = extract_green_channel(corrected)
    except ValueError:
        gray = ensure_grayscale(corrected)
    results["gray"] = gray

    # 3. 高斯滤波去噪
    blurred = cv2.GaussianBlur(
        gray, config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_SIGMA,
    )
    results["blurred"] = blurred

    # 4. 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESH_BLOCK_SIZE, config.ADAPTIVE_THRESH_C,
    )
    results["binary"] = binary

    # 5. Canny 边缘检测
    edges = cv2.Canny(blurred, config.CANNY_LOW_THRESH, config.CANNY_HIGH_THRESH)
    results["edge"] = edges

    return results


def save_baseline_steps(
    results: Dict[str, np.ndarray],
    image_name: str,
    output_dir: Optional[Path] = None,
) -> None:
    """保存普通预处理各步骤的中间结果图像。"""
    if output_dir is None:
        output_dir = config.EXPERIMENTS_OUTPUT_DIR / "algorithm_a"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_map = {
        "corrected": "01_corrected",
        "gray":      "02_gray",
        "blurred":   "03_blurred",
        "binary":    "04_binary",
        "edge":      "05_edge",
    }

    for key, suffix in suffix_map.items():
        if key in results:
            out_path = output_dir / f"{image_name}_{suffix}{config.OUTPUT_IMAGE_EXT}"
            save_image(results[key], out_path)

    print(f"  [INFO] 算法A预处理中间结果已保存到: {output_dir}")
