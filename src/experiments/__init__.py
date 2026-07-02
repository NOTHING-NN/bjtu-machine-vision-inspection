"""
src.experiments — 实验与报告层

提供：
  - ExperimentConfig / BASELINE_CONFIG / LIGHT_CORRECTED_CONFIG — 实验配置
  - run_experiment() / process_single() — 统一实验执行
  - generate_report_figures() — 报告图生成
"""

from src.experiments.configs import (
    ExperimentConfig,
    BASELINE_CONFIG,
    LIGHT_CORRECTED_CONFIG,
    SAMPLE_TYPE_MAP,
)
from src.experiments.runner import (
    process_single,
    run_experiment,
    save_results_csv,
    print_grouped_summary,
)
from src.experiments.report import (
    draw_board_corners,
    draw_holes,
    draw_measurement_text,
    make_side_by_side_comparison,
    save_debug_figure,
    generate_report_figures,
)
