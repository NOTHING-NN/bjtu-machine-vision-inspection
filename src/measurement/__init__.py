"""
src.measurement — 几何测量层

提供：
  - detect_all_holes()   — 无 PCB 尺寸先验的俯视图安装孔检测
  - run_measurement()    — 基于棋盘格测量平面的尺寸/误差计算
  - summarize_measurements() — 多图汇总统计
"""

from src.measurement.hole_detector import detect_all_holes, detect_hole_in_roi
from src.measurement.geometry import (
    compute_board_dimensions,
    compute_measurement_resolution,
    compute_errors,
    compute_hole_pitch,
    run_measurement,
    summarize_measurements,
)
