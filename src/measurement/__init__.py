"""
src.measurement — 几何测量层

提供：
  - detect_all_holes()   — 俯视图安装孔检测
  - run_measurement()    — 孔距/孔径/误差计算
  - summarize_measurements() — 多图汇总统计
"""

from src.measurement.hole_detector import detect_all_holes, detect_hole_in_roi
from src.measurement.geometry import (
    compute_errors,
    compute_hole_diameter,
    compute_hole_pitch,
    pixel_to_mm_after_warp,
    run_measurement,
    summarize_measurements,
)
