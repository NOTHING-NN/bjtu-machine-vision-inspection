"""
preprocess_light_correct.py — 光照矫正预处理流程（改进组）

流水线：
  原图 → 畸变校正 → ExG 绿色增强 → 大尺度高斯模糊估计光照背景
  → 平场校正/背景扣除 → 高亮抑制 → CLAHE 局部对比度增强
  → 自适应阈值 / Canny 边缘检测

本模块为本项目的核心创新点：
  针对非均匀光照条件下 PCB 图像的预处理，
  通过估计低频光照分量并进行背景扣除，
  实现对光照不均匀的矫正，提高后续检测精度。

每步均有详细注释，便于撰写报告时说明各步骤的原理和目的。
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.calibrate_camera import undistort_image
from src.utils import save_image


# ============================================================
# 子步骤函数
# ============================================================

def compute_exg(image: np.ndarray) -> np.ndarray:
    """
    计算 ExG（Excess Green）绿色增强指数。

    公式：ExG = 2G - R - B

    目的：
      PCB 基板通常为绿色，ExG 可增强绿色基板与铜箔走线（红/黄色调）
      之间的对比度，使后续光照估计更准确。
      同时 ExG 对阴影和高光的敏感度低于单独的 G 通道。

    Args:
        image: BGR 彩色图像（uint8）

    Returns:
        ExG 特征图像（float32，可能含负值，后续归一化）
    """
    # 分离 BGR 三个通道，转为 float32 避免溢出
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    # ExG = 2G - R - B
    exg = 2.0 * g - r - b

    return exg


def estimate_illumination(feature_image: np.ndarray) -> np.ndarray:
    """
    通过大尺度高斯模糊估计低频光照背景。

    原理：
      一幅图像 I(x,y) 可以建模为：
        I(x,y) = R(x,y) × L(x,y)   （Retinex 理论）
      其中 R(x,y) 为反射分量（高频，物体本身属性），
      L(x,y) 为光照分量（低频，光照不均匀的来源）。

      通过大尺度高斯低通滤波，提取低频成分作为光照估计 L*(x,y)，
      后续通过 I/L* 或 I - L* 来消除光照不均匀。

    Args:
        feature_image: 特征图像（如 ExG 特征图）

    Returns:
        低频光照估计图像（与输入同尺寸，float32）
    """
    illumination = cv2.GaussianBlur(
        feature_image,
        config.LARGE_GAUSSIAN_KERNEL_SIZE,
        config.LARGE_GAUSSIAN_SIGMA,
    )
    return illumination


def correct_illumination(feature_image: np.ndarray,
                         illumination: np.ndarray) -> np.ndarray:
    """
    平场校正 / 背景扣除。

    使用"减去低频分量"模型进行光照矫正：
      I_corrected = I - L* + mean(L*)

    这样可以：
      1. 扣除不均匀的光照背景。
      2. 加回均值以保持整体亮度水平不变。

    同时也返回"除法型"矫正结果（对比用）：
      I_retinex = I / (L* + eps)

    Args:
        feature_image: 原始特征图像（float32）
        illumination: 估计的光照分量（float32）

    Returns:
        光照矫正后的图像（float32）
    """
    # 背景减法矫正（Retinex 减法模型）
    mean_illum = illumination.mean()
    corrected = feature_image - illumination + mean_illum

    # 归一化到合理范围
    corrected = np.clip(corrected, 0, 255)

    return corrected


def suppress_highlight(image: np.ndarray) -> np.ndarray:
    """
    高亮区域抑制。

    目的：
      PCB 图像中金属焊盘和锡层可能产生强烈反光，
      这些高亮区域的像素值远高于正常区域，会干扰后续的边缘检测。
      通过对高亮区域进行限制，减少其对后续算法的干扰。

    方法：
      将超过指定分位数的像素值截断到该分位数。

    Args:
        image: 输入图像（uint8）

    Returns:
        高亮抑制后的图像（uint8）
    """
    # 计算高亮度阈值（指定分位数）
    thresh = np.percentile(image, config.HIGHLIGHT_PERCENTILE)
    # 将超过阈值的像素截断
    suppressed = np.clip(image, 0, thresh)
    # 重新拉伸到 [0, 255]
    if suppressed.max() > 0:
        suppressed = (suppressed / suppressed.max() * 255).astype(np.uint8)
    else:
        suppressed = suppressed.astype(np.uint8)
    return suppressed


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    CLAHE（Contrast Limited Adaptive Histogram Equalization）
    自适应直方图均衡化。

    目的：
      经过光照矫正后，部分区域的对比度可能仍然偏低。
      CLAHE 在局部区域内进行直方图均衡，并限制对比度放大倍数，
      能够有效增强局部细节，同时避免噪声过度放大。

    Args:
        image: 输入图像（uint8 灰度图）

    Returns:
        CLAHE 增强后的图像（uint8）
    """
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    enhanced = clahe.apply(image)
    return enhanced


# ============================================================
# 完整流水线
# ============================================================

def preprocess_light_corrected(
    image: np.ndarray,
    camera_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    执行完整的「光照矫正」预处理流程（改进组）。

    流程：
      1. 畸变校正
      2. ExG 绿色增强 — 增强 PCB 基板特征
      3. 大尺度高斯模糊 — 估计低频光照背景
      4. 背景减法矫正 — 消除光照不均匀
      5. 归一化到 uint8
      6. 高亮区域抑制 — 减少金属反光干扰
      7. CLAHE 增强 — 提升局部对比度
      8. 高斯滤波去噪
      9. 自适应阈值二值化
      10. Canny 边缘检测

    Args:
        image: 输入 BGR 图像
        camera_params: (camera_matrix, dist_coeffs) 元组，None 则跳过畸变校正

    Returns:
        包含各阶段结果图像的 dict:
            - "corrected":      畸变校正后的 BGR 图
            - "exg":            ExG 特征图（归一化 uint8）
            - "illumination":   估计的光照分量（归一化 uint8）
            - "illum_corrected": 光照矫正后的图像（uint8）
            - "highlight_supp": 高亮抑制后的图像（uint8）
            - "enhanced":       CLAHE 增强后的图像（uint8）
            - "binary":         二值化图
            - "edge":           Canny 边缘图
    """
    results = {}

    # -------------------------------------------------------
    # 1. 畸变校正
    # -------------------------------------------------------
    if camera_params is not None and camera_params[0] is not None:
        camera_matrix, dist_coeffs = camera_params
        corrected = undistort_image(image, camera_matrix, dist_coeffs)
    else:
        corrected = image.copy()
    results["corrected"] = corrected

    # -------------------------------------------------------
    # 2. ExG 绿色增强
    #    目的：利用 PCB 绿色基板特征，增强基板与铜箔的对比度
    # -------------------------------------------------------
    exg_float = compute_exg(corrected)

    # 归一化 ExG 到 [0, 255] 以便可视化和后续处理
    exg_min = exg_float.min()
    exg_max = exg_float.max()
    if exg_max - exg_min > 1e-6:
        exg_uint8 = ((exg_float - exg_min) / (exg_max - exg_min) * 255).astype(np.uint8)
    else:
        exg_uint8 = np.zeros_like(exg_float, dtype=np.uint8)
    results["exg"] = exg_uint8

    # -------------------------------------------------------
    # 3. 大尺度高斯模糊 — 估计低频光照背景
    #    原理：大核低通滤波保留光照的宏观分布，过滤掉
    #          电路板细节纹理等高频信息
    # -------------------------------------------------------
    illumination_float = estimate_illumination(exg_float)
    # 归一化用于可视化
    ill_min = illumination_float.min()
    ill_max = illumination_float.max()
    if ill_max - ill_min > 1e-6:
        illum_uint8 = ((illumination_float - ill_min) / (ill_max - ill_min) * 255).astype(np.uint8)
    else:
        illum_uint8 = np.zeros_like(illumination_float, dtype=np.uint8)
    results["illumination"] = illum_uint8

    # -------------------------------------------------------
    # 4. 背景减法矫正（光照矫正核心步骤）
    #    目的：去除由非均匀光照造成的低频亮度变化
    # -------------------------------------------------------
    corrected_float = correct_illumination(exg_float, illumination_float)
    illum_corrected_uint8 = np.clip(corrected_float, 0, 255).astype(np.uint8)
    results["illum_corrected"] = illum_corrected_uint8

    # -------------------------------------------------------
    # 5. 高亮区域抑制
    #    目的：限制金属焊盘/锡层反光造成的过亮像素
    # -------------------------------------------------------
    highlight_supp = suppress_highlight(illum_corrected_uint8)
    results["highlight_supp"] = highlight_supp

    # -------------------------------------------------------
    # 6. CLAHE 局部对比度增强
    #    目的：提升局部区域的细节可见度
    # -------------------------------------------------------
    enhanced = apply_clahe(highlight_supp)
    results["enhanced"] = enhanced

    # -------------------------------------------------------
    # 7. 高斯滤波去噪（为后续边缘检测准备）
    # -------------------------------------------------------
    blurred = cv2.GaussianBlur(
        enhanced,
        config.GAUSSIAN_KERNEL_SIZE,
        config.GAUSSIAN_SIGMA,
    )

    # -------------------------------------------------------
    # 8. 自适应阈值二值化
    # -------------------------------------------------------
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESH_BLOCK_SIZE,
        config.ADAPTIVE_THRESH_C,
    )
    results["binary"] = binary

    # -------------------------------------------------------
    # 9. Canny 边缘检测
    # -------------------------------------------------------
    edges = cv2.Canny(
        blurred,
        config.CANNY_LOW_THRESH,
        config.CANNY_HIGH_THRESH,
    )
    results["edge"] = edges

    return results


def save_light_corrected_steps(
    results: Dict[str, np.ndarray],
    image_name: str,
    output_dir: Optional[Path] = None,
) -> None:
    """
    保存光照矫正预处理各步骤的中间结果图像。

    Args:
        results: preprocess_light_corrected 返回的 dict
        image_name: 原始图像文件名（不含扩展名）
        output_dir: 输出目录，默认为 outputs/light_corrected/
    """
    if output_dir is None:
        output_dir = config.LIGHT_CORRECTED_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_map = {
        "corrected":       "01_corrected",
        "exg":             "02_exg",
        "illumination":    "03_illumination",
        "illum_corrected": "04_illum_corrected",
        "highlight_supp":  "05_highlight_supp",
        "enhanced":        "06_enhanced",
        "binary":          "07_binary",
        "edge":            "08_edge",
    }

    for key, suffix in suffix_map.items():
        if key in results:
            out_path = output_dir / f"{image_name}_{suffix}{config.OUTPUT_IMAGE_EXT}"
            save_image(results[key], out_path)

    print(f"  [INFO] 光照矫正预处理中间结果已保存到: {output_dir}")
