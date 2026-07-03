"""
src.preprocessing — 预处理层

提供两种可替换预处理策略：
  - standard (原 baseline):    普通灰度化/G通道 + 高斯滤波 + 自适应阈值 + Canny
  - highlight_aware (原 light_corrected): 光照场估计 + 高光抑制 + 暗区提升 + CLAHE

两种策略输出统一格式的 dict，后续检测模块可选择使用其中的
mask、corrected_image 等中间结果。
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.calibration import undistort_image
from src.preprocessing.baseline import preprocess_baseline, save_baseline_steps
from src.preprocessing.illumination import compute_illumination_field, boost_shadows
from src.preprocessing.highlight import detect_highlights, suppress_highlights

# 别名：保持向后兼容的同时提供新命名
preprocess_standard = preprocess_baseline
save_standard_steps = save_baseline_steps


def preprocess_highlight_aware(
    image: np.ndarray,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    强光斑感知预处理全流程（算法 B 专用）。

    流程：
      1. 畸变校正
      2. BGR → HSV
      3. 光照场估计
      4. 高光检测 + 抑制
      5. 暗区提升
      6. CLAHE 局部对比度增强
      7. HSV → BGR，输出矫正彩色图
      8. 灰度化 + 二值化 + Canny

    Args:
        image:        输入 BGR 图像
        camera_params: (camera_matrix, dist_coeffs) 或 None

    Returns:
        {
            "corrected":       光照矫正后的 BGR 彩色图
            "gray":            灰度图
            "illumination":    光照场可视化
            "highlight_mask":  高光区域 mask
            "binary":          二值化图
            "edge":            Canny 边缘图
        }
    """
    results = {}

    # 1. 畸变校正
    if camera_params is not None and camera_params[0] is not None:
        cm, dc = camera_params
        corrected_bgr = undistort_image(image, cm, dc)
    else:
        corrected_bgr = image.copy()
    results["undistorted"] = corrected_bgr

    # 2. BGR → HSV
    hsv = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 3. 光照场估计
    illumination = compute_illumination_field(v)
    results["illumination"] = illumination

    # 4. 高光检测 + 抑制
    highlight_mask = detect_highlights(h, s, v)
    results["highlight_mask"] = (highlight_mask * 255).astype(np.uint8) \
        if highlight_mask.max() <= 1 else highlight_mask

    v = suppress_highlights(v, highlight_mask, illumination.astype(np.float32))

    # 5. 暗区提升
    v = boost_shadows(v, illumination.astype(np.float32))

    # 6. CLAHE 局部对比度增强（仅 V 通道）
    v = np.clip(v, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                             tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    v = clahe.apply(v)

    # 7. HSV → BGR
    hsv_out = np.dstack([h.astype(np.uint8), s.astype(np.uint8), v])
    corrected_final_bgr = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
    results["corrected"] = corrected_final_bgr

    gray = cv2.cvtColor(corrected_final_bgr, cv2.COLOR_BGR2GRAY)
    results["gray"] = gray

    # 8. 二值化 + 边缘（用于报告对比，不影响检测流程）
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL_SIZE,
                               config.GAUSSIAN_SIGMA)
    results["binary"] = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, config.ADAPTIVE_THRESH_BLOCK_SIZE,
        config.ADAPTIVE_THRESH_C,
    )
    results["edge"] = cv2.Canny(blurred, config.CANNY_LOW_THRESH,
                                config.CANNY_HIGH_THRESH)

    return results


def save_highlight_aware_steps(
    results: Dict[str, np.ndarray],
    image_name: str,
    output_dir=None,
) -> None:
    """保存强光斑感知预处理各步骤的中间结果图像。"""
    from pathlib import Path
    from src.utils import save_image as _save_image

    if output_dir is None:
        output_dir = config.EXPERIMENTS_OUTPUT_DIR / "algorithm_b"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_map = {
        "corrected":       "01_corrected_bgr",
        "illumination":    "02_illumination",
        "highlight_mask":  "03_highlight_mask",
        "gray":            "04_gray",
        "binary":          "05_binary",
        "edge":            "06_edge",
    }

    for key, suffix in suffix_map.items():
        if key in results:
            out_path = output_dir / f"{image_name}_{suffix}{config.OUTPUT_IMAGE_EXT}"
            _save_image(results[key], out_path)

    print(f"  [INFO] 算法B预处理中间结果已保存到: {output_dir}")


# 向后兼容别名
preprocess_light_corrected = preprocess_highlight_aware
save_light_corrected_steps = save_highlight_aware_steps
