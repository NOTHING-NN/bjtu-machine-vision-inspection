"""
utils.py — 通用工具函数

提供图像 I/O、路径处理、结果记录等辅助功能，
其他模块按需引用。
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 图像 I/O
# ============================================================

def get_image_paths(directory: Path) -> List[Path]:
    """
    获取指定目录下所有图像文件的路径（按文件名排序）。

    Args:
        directory: 图像目录路径

    Returns:
        排序后的图像文件路径列表
    """
    from src.config import IMAGE_EXTENSIONS

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    paths = sorted(
        [p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )
    return paths


def load_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    加载单张图像。

    Args:
        path: 图像文件路径
        flags: cv2.imread 的读取标志

    Returns:
        图像数组，加载失败返回 None
    """
    if not path.exists():
        print(f"[WARNING] 文件不存在: {path}")
        return None

    img = cv2.imread(str(path), flags)
    if img is None:
        print(f"[WARNING] 无法读取图像: {path}")
    return img


def save_image(image: np.ndarray, path: Path) -> bool:
    """
    保存图像到指定路径，自动创建父目录。

    Args:
        image: 图像数组
        path: 保存路径

    Returns:
        是否成功保存
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        print(f"[ERROR] 图像保存失败: {path}")
    return success


# ============================================================
# 图像变换
# ============================================================

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """如果输入为彩色图，转为灰度图。"""
    if image is None:
        raise ValueError("输入图像为空")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    提取彩色图像的 G（绿）通道。

    PCB 基板通常为绿色，G 通道对铜箔走线与基板的区分度较好。

    Args:
        image: BGR 彩色图像

    Returns:
        G 通道灰度图像
    """
    if image is None or len(image.shape) < 3:
        raise ValueError("输入必须为彩色图像")
    return image[:, :, 1]  # OpenCV 为 BGR 顺序，索引 1 为 G


# ============================================================
# 异常处理辅助
# ============================================================

class DetectionResult:
    """
    检测结果容器。

    统一封装各类检测结果，包含成功标志和结构化的返回值，
    避免在检测失败时抛出异常导致程序崩溃。

    Attributes:
        success: 检测是否成功
        data: 检测结果数据（dict）
        message: 描述信息（成功或失败原因）
    """

    def __init__(self, success: bool, data: dict = None,
                 message: str = ""):
        self.success = success
        self.data = data if data is not None else {}
        self.message = message

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"DetectionResult({status}) {self.message}"

    def __bool__(self) -> bool:
        return self.success


# ============================================================
# 辅助函数
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    """将值限制在 [low, high] 区间内。"""
    return max(low, min(high, value))


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """安全除法，分母为 0 时返回默认值。"""
    if denominator == 0:
        return default
    return numerator / denominator
