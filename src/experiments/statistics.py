"""
statistics.py — 实验统计与性能评估

提供：
  - 板检测 / 四角点定位 / 孔检测成功率统计
  - X/Y 孔距误差分布与重复性标准差
  - 强光斑分离效果评估（分离前/后连通域数量对比）
  - 三类分组对比报告格式化输出
"""

from typing import Dict, List, Optional

import numpy as np


# ============================================================
# 1. 检测成功率统计
# ============================================================

def compute_detection_rates(results: List[Dict]) -> Dict:
    """
    计算板检测 / 孔检测的综合成功率。

    Args:
        results: process_single() 返回的结果 dict 列表

    Returns:
        {
            "total_images": int,
            "board_detect_ok": int,
            "board_detect_rate": float,
            "holes_detect_ok": int,
            "holes_detect_rate": float,
            "end_to_end_ok": int,
            "end_to_end_rate": float,
        }
    """
    if not results:
        return {
            "total_images": 0,
            "board_detect_ok": 0, "board_detect_rate": 0.0,
            "holes_detect_ok": 0, "holes_detect_rate": 0.0,
            "end_to_end_ok": 0, "end_to_end_rate": 0.0,
        }

    n = len(results)
    board_ok = sum(1 for r in results if r.get("board_detect_success"))
    holes_ok = sum(1 for r in results if r.get("holes_detect_success"))
    e2e_ok = sum(1 for r in results
                 if r.get("board_detect_success") and r.get("holes_detect_success"))

    return {
        "total_images": n,
        "board_detect_ok": board_ok,
        "board_detect_rate": board_ok / n,
        "holes_detect_ok": holes_ok,
        "holes_detect_rate": holes_ok / n,
        "end_to_end_ok": e2e_ok,
        "end_to_end_rate": e2e_ok / n,
    }


# ============================================================
# 2. 误差分布统计
# ============================================================

def compute_error_statistics(results: List[Dict]) -> Dict:
    """
    统计有效测量结果的 X/Y 孔距误差分布。

    只使用 board_detect_success 且 holes_detect_success 均为 True 的结果。

    Returns:
        {
            "n_valid": int,
            "pitch_x_mean": float, "pitch_x_std": float,
            "pitch_y_mean": float, "pitch_y_std": float,
            "abs_error_x_mean": float, "abs_error_y_mean": float,
            "mean_abs_error_mean": float,     # 平均绝对误差
            "mean_abs_error_std": float,      # 重复性标准差
            "mean_abs_error_min": float,
            "mean_abs_error_max": float,
        }
    """
    valid = [r for r in results
             if r.get("board_detect_success") and r.get("holes_detect_success")]

    if not valid:
        return {"n_valid": 0}

    keys = [
        "pitch_x_mm", "pitch_y_mm",
        "abs_error_x_mm", "abs_error_y_mm",
        "mean_abs_error_mm",
    ]

    stats = {"n_valid": len(valid)}
    for key in keys:
        values = [r[key] for r in valid if not np.isnan(r.get(key, np.nan))]
        if values:
            arr = np.array(values)
            stats[f"{key}_mean"] = float(np.mean(arr))
            stats[f"{key}_std"] = float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0
            stats[f"{key}_min"] = float(np.min(arr))
            stats[f"{key}_max"] = float(np.max(arr))

    return stats


# ============================================================
# 3. 分组统计
# ============================================================

def compute_grouped_stats(results: List[Dict]) -> Dict[str, Dict]:
    """
    按 (method, sample_type) 分组计算检测率和误差统计。

    Returns:
        {
            "algorithm_a|A": {...detection rates...},
            "algorithm_a|B": {...},
            "algorithm_b|A": {...},
            "algorithm_b|B": {...},
        }
    """
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        method = r.get("method", "unknown")
        stype = r.get("sample_type", "?")
        key = f"{method}|{stype}"
        groups.setdefault(key, []).append(r)

    output = {}
    for key, items in groups.items():
        detection = compute_detection_rates(items)
        error = compute_error_statistics(items)
        output[key] = {**detection, **error}

    return output


# ============================================================
# 4. 区域分离效果评估
# ============================================================

def compute_separation_impact(results: List[Dict]) -> Dict:
    """
    评估强光斑分离是否生效及其影响。

    仅考虑 method 中包含 region_separation 的实验结果。

    Returns:
        {
            "total": int,
            "separation_triggered": int,     # 实际触发了分离的图像数
            "separation_recovery": int,      # 触发分离且检测成功的图像数
            "separation_fallback": int,      # 检测仍失败的图像数
            "n_before_mean": float,          # 分离前平均连通域数
            "n_after_mean": float,           # 分离后平均连通域数
        }
    """
    separated = [r for r in results if r.get("separation_applied")]

    if not separated:
        return {
            "total": 0, "separation_triggered": 0,
            "separation_recovery": 0, "separation_fallback": 0,
            "n_before_mean": 0.0, "n_after_mean": 0.0,
        }

    triggered = [r for r in separated if r.get("separation_modified")]
    recovery = [r for r in triggered if r.get("board_detect_success")]
    fallback = [r for r in triggered if not r.get("board_detect_success")]

    n_before_vals = [r.get("num_components_before", 0) for r in separated
                     if r.get("num_components_before") is not None]
    n_after_vals = [r.get("num_components_after", 0) for r in separated
                    if r.get("num_components_after") is not None]

    return {
        "total": len(separated),
        "separation_triggered": len(triggered),
        "separation_recovery": len(recovery),
        "separation_fallback": len(fallback),
        "n_before_mean": float(np.mean(n_before_vals)) if n_before_vals else 0.0,
        "n_after_mean": float(np.mean(n_after_vals)) if n_after_vals else 0.0,
    }


# ============================================================
# 5. 格式化输出
# ============================================================

SAMPLE_LABELS = {
    "A": "A 类 (正常/轻反光)",
    "B": "B 类 (强光斑邻接)",
}

METHOD_LABELS = {
    "algorithm_a": "算法 A (基础测量)",
    "algorithm_b": "算法 B (强光斑感知改进)",
}


def _fmt_pct(value: float) -> str:
    """格式化百分数。"""
    return f"{value * 100:.1f}%"


def _fmt_mm(value: float) -> str:
    """格式化毫米值。"""
    if np.isnan(value):
        return "N/A"
    return f"{value:.4f} mm"


def _print_section_header(title: str) -> str:
    """生成分节标题。"""
    bar = "=" * 64
    return f"\n{bar}\n  {title}\n{bar}"


def _print_sub_header(title: str) -> str:
    """生成子标题。"""
    bar = "-" * 48
    return f"\n{bar}\n  {title}\n{bar}"


def format_statistics_table(
    all_results: List[Dict],
) -> str:
    """
    生成两类对比统计报告。

    输出：
      1. 算法 A (基础测量) 在 A/B 两类样本上的运行结果
      2. 算法 A vs 算法 B 在 B 类样本上的对比（含分离效果与四边形恢复）

    说明：算法 B 仅运行在 B 类样本上，不参与 A 类样本对比。

    Args:
        all_results: 所有实验结果列表（含 method + sample_type 字段）

    Returns:
        格式化后的多行字符串，可直接 print
    """
    grouped = compute_grouped_stats(all_results)
    lines = []

    # ─── 类别 1：基础算法在 A/B 两类样本上的运行结果 ───
    lines.append(_print_section_header(
        "1. 算法 A (基础测量) 在 A/B 两类样本上的运行结果"))
    lines.append("  说明：强光斑邻接会导致常规流程失效。")

    for stype in ["A", "B"]:
        key = f"algorithm_a|{stype}"
        stats = grouped.get(key)
        if stats is None:
            lines.append(f"\n  {SAMPLE_LABELS.get(stype, stype)}: 无数据")
            continue

        s = stats
        lines.append(_print_sub_header(SAMPLE_LABELS.get(stype, stype)))
        lines.append(f"  图像数:        {s['total_images']}")
        lines.append(f"  板检测成功:    {s['board_detect_ok']}/{s['total_images']}"
                     f"  ({_fmt_pct(s['board_detect_rate'])})")
        lines.append(f"  孔检测成功:    {s['holes_detect_ok']}/{s['total_images']}"
                     f"  ({_fmt_pct(s['holes_detect_rate'])})")
        lines.append(f"  端到端成功:    {s['end_to_end_ok']}/{s['total_images']}"
                     f"  ({_fmt_pct(s['end_to_end_rate'])})")
        if s.get("n_valid", 0) > 0:
            lines.append(f"  平均孔距误差:  {_fmt_mm(s.get('mean_abs_error_mm_mean', np.nan))}"
                         f" ± {_fmt_mm(s.get('mean_abs_error_mm_std', np.nan))}")
            lines.append(f"  X 方向误差:    {_fmt_mm(s.get('abs_error_x_mean', np.nan))}"
                         f" ± {_fmt_mm(s.get('abs_error_x_std', np.nan))}")
            lines.append(f"  Y 方向误差:    {_fmt_mm(s.get('abs_error_y_mean', np.nan))}"
                         f" ± {_fmt_mm(s.get('abs_error_y_std', np.nan))}")

    # ─── 类别 2：算法 A vs 算法 B 在 B 类样本上的对比 ───
    lines.append(_print_section_header(
        "2. 算法 A vs 算法 B 在 B 类样本上的对比"))
    lines.append("  说明：验证区域分离算法能否恢复外框检测、孔检测和尺寸测量。")

    key_a_b = "algorithm_a|B"
    key_b_b = "algorithm_b|B"
    stats_a = grouped.get(key_a_b)
    stats_b = grouped.get(key_b_b)

    if stats_a and stats_b:
        lines.append(_print_sub_header("板检测成功率"))
        lines.append(f"  算法 A:  {stats_a['board_detect_ok']}/{stats_a['total_images']}"
                     f"  ({_fmt_pct(stats_a['board_detect_rate'])})")
        lines.append(f"  算法 B:  {stats_b['board_detect_ok']}/{stats_b['total_images']}"
                     f"  ({_fmt_pct(stats_b['board_detect_rate'])})")

        lines.append(_print_sub_header("孔检测成功率"))
        lines.append(f"  算法 A:  {stats_a['holes_detect_ok']}/{stats_a['total_images']}"
                     f"  ({_fmt_pct(stats_a['holes_detect_rate'])})")
        lines.append(f"  算法 B:  {stats_b['holes_detect_ok']}/{stats_b['total_images']}"
                     f"  ({_fmt_pct(stats_b['holes_detect_rate'])})")

        lines.append(_print_sub_header("端到端成功率"))
        lines.append(f"  算法 A:  {stats_a['end_to_end_ok']}/{stats_a['total_images']}"
                     f"  ({_fmt_pct(stats_a['end_to_end_rate'])})")
        lines.append(f"  算法 B:  {stats_b['end_to_end_ok']}/{stats_b['total_images']}"
                     f"  ({_fmt_pct(stats_b['end_to_end_rate'])})")

        if stats_a.get("n_valid", 0) > 0 or stats_b.get("n_valid", 0) > 0:
            lines.append(_print_sub_header("孔距测量误差"))
            lines.append(f"  算法 A:  mean={_fmt_mm(stats_a.get('mean_abs_error_mm_mean', np.nan))}"
                         f"  std={_fmt_mm(stats_a.get('mean_abs_error_mm_std', np.nan))}")
            lines.append(f"  算法 B:  mean={_fmt_mm(stats_b.get('mean_abs_error_mm_mean', np.nan))}"
                         f"  std={_fmt_mm(stats_b.get('mean_abs_error_mm_std', np.nan))}")

    # 区域分离 + 四边形恢复效果
    algo_b_results = [r for r in all_results if r.get("method") == "algorithm_b"]
    sep_impact = compute_separation_impact(algo_b_results)
    if sep_impact["total"] > 0:
        lines.append(_print_sub_header("强光斑分离效果 (算法 B)"))
        lines.append(f"  分离触发:  {sep_impact['separation_triggered']}/{sep_impact['total']}")
        lines.append(f"  分离后成功检测: {sep_impact['separation_recovery']}")
        lines.append(f"  分离后仍失败:   {sep_impact['separation_fallback']}")
        lines.append(f"  分离前连通域平均数: {sep_impact['n_before_mean']:.1f}")
        lines.append(f"  分离后连通域平均数: {sep_impact['n_after_mean']:.1f}")

    # 四边形恢复统计
    quad_recovered = [r for r in algo_b_results if r.get("quad_recovery_used")]
    if quad_recovered:
        lines.append(_print_sub_header("四边形恢复效果 (算法 B)"))
        lines.append(f"  四边形恢复触发: {len(quad_recovered)}/{len(algo_b_results)}")
        for r in quad_recovered:
            name = r.get("image_name", "?")
            err = r.get("mean_abs_error_mm", np.nan)
            lines.append(f"    {name}: 孔距误差={_fmt_mm(err)}")

    # 汇总
    lines.append(_print_section_header("汇总"))
    lines.append(f"  总处理次数: {len(all_results)}")
    lines.append(f"  算法 A 样本范围: 全部 (A 类 + B 类)")
    lines.append(f"  算法 B 样本范围: 仅 B 类")
    for method in ["algorithm_a", "algorithm_b"]:
        subset = [r for r in all_results if r.get("method") == method]
        if subset:
            rates = compute_detection_rates(subset)
            label = METHOD_LABELS.get(method, method)
            scope = "全部" if method == "algorithm_a" else "仅 B 类"
            lines.append(f"  {label} ({scope}): "
                         f"板检测 {rates['board_detect_ok']}/{rates['total_images']}"
                         f" ({_fmt_pct(rates['board_detect_rate'])}), "
                         f"孔检测 {rates['holes_detect_ok']}/{rates['total_images']}"
                         f" ({_fmt_pct(rates['holes_detect_rate'])})")

    return "\n".join(lines)
