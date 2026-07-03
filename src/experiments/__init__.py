"""
src.experiments — 实验与报告层

提供：
  - ExperimentConfig / ALGORITHM_A_CONFIG / ALGORITHM_B_CONFIG — 实验配置
  - run_experiment() / process_single() — 统一实验执行
  - generate_report_figures() — 报告图生成
  - compute_detection_rates() / compute_measurement_stats() — 统计评估
"""

from src.experiments.configs import (
    ExperimentConfig,
    ALGORITHM_A_CONFIG,
    ALGORITHM_B_CONFIG,
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
from src.experiments.statistics import (
    compute_detection_rates,
    compute_error_statistics,
    compute_separation_impact,
    format_statistics_table,
)
