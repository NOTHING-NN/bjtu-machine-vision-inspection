"""
configs.py — 实验配置与样本分类

定义：
  - ExperimentConfig — 实验配置数据类
  - SAMPLE_TYPE_MAP  — 样本光照类型映射
  - 预设实验配置
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExperimentConfig:
    """一次实验的完整配置。"""
    name: str                       # "baseline" | "light_corrected"
    preprocessing: str              # "baseline" | "illumination_corrected"
    board_mask: str                 # "hsv_green" | "illumination_aware_green"
    region_separation: bool         # 是否启用光斑分离


# ============================================================
# 预设实验配置
# ============================================================

BASELINE_CONFIG = ExperimentConfig(
    name="baseline",
    preprocessing="baseline",
    board_mask="hsv_green",
    region_separation=False,
)

LIGHT_CORRECTED_CONFIG = ExperimentConfig(
    name="light_corrected",
    preprocessing="illumination_corrected",
    board_mask="illumination_aware_green",
    region_separation=True,
)

# ============================================================
# 样本光照类型映射
# ============================================================

SAMPLE_TYPE_MAP: Dict[str, str] = {
    "01": "A",   # 板面反光 — PCB 本身局部发亮，背景光斑不严重
    "02": "A",
    "03": "A",
    "04": "A",
    "05": "B",   # 邻近强光斑 — 巨大光斑紧贴 PCB
    "06": "B",
    "07": "B",
}
