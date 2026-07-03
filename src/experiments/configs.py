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
    name: str                       # "algorithm_a" | "algorithm_b"
    preprocessing: str              # "standard" | "highlight_aware"
    board_mask: str                 # "hsv_green" | "highlight_aware_green"
    region_separation: bool         # 是否启用光斑分离


# ============================================================
# 预设实验配置
# ============================================================

ALGORITHM_A_CONFIG = ExperimentConfig(
    name="algorithm_a",
    preprocessing="standard",
    board_mask="hsv_green",
    region_separation=False,
)

ALGORITHM_B_CONFIG = ExperimentConfig(
    name="algorithm_b",
    preprocessing="highlight_aware",
    board_mask="highlight_aware_green",
    region_separation=True,
)

# ============================================================
# 样本光照类型映射
# ============================================================

SAMPLE_TYPE_MAP: Dict[str, str] = {
    "01": "A",   # 正常反光 — PCB 本身局部发亮
    "02": "A",
    "03": "A",
    "04": "A",
    "05": "B",   # 强光斑 — 巨大光斑紧贴 PCB
    "06": "B",
}
