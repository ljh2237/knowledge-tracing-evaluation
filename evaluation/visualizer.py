"""模块五：实验结果可视化（matplotlib）。

生成八类图表：AUC / Accuracy / Log Loss 对比柱状图、ROC 曲线、遗忘时间响应曲线、
不同模型的时间响应对比图、推理延迟对比图、不同批量大小下的吞吐量图。

设计约束：

- **不写死任何实验数值**，所有数据都来自对应模块的计算结果。
- 无法计算的图表（例如单一类别无法画 ROC、只有一个模型无法做模型间对比）会被跳过，
  并在 ``plots_manifest.json`` 中写明原因，而不是画一张空图。
- 只要数据来自 ``is_mock=True`` 的模拟模型，图中就会带上明显的 ``DEMO · SYNTHETIC DATA``
  水印，并用斜纹柱区分模拟模型。
- 优先使用中文字体；环境缺少中文字体时自动退回英文标签，不会画出方框或报错。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在导入 pyplot 之前设置，保证无显示环境也能出图。

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.metrics import auc as sk_auc, roc_curve  # noqa: E402

from evaluation.common import (  # noqa: E402
    EVALUATION_VERSION,
    EvaluationError,
    PredictionData,
    ensure_output_dir,
    read_json_document,
    utc_now_iso,
    write_document,
)
from evaluation.evaluator import GROUP_ALL  # noqa: E402

PLOT_DPI = 200
FIGURE_SIZE = (9.0, 5.2)
FIGURE_SIZE_WIDE = (11.0, 5.6)
FIGURE_SIZE_DOUBLE = (12.0, 5.2)

PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
)
DEMO_HATCH = "///"
DEMO_TEXT = "DEMO · SYNTHETIC DATA"

#: 按优先级尝试的中文字体；全部缺失时改用英文标签。
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "PingFang SC", "Hiragino Sans GB", "WenQuanYi Zen Hei", "DengXian",
    "Arial Unicode MS", "Microsoft JhengHei",
)

_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "auc_title": "模型 ROC-AUC 对比",
        "accuracy_title": "模型 Accuracy 对比",
        "logloss_title": "模型 Log Loss 对比",
        "roc_title": "ROC 曲线",
        "time_title": "时间间隔与平均预测概率（观察性分析）",
        "time_compare_title": "不同模型的时间响应对比（受控固定历史）",
        "latency_title": "推理延迟对比",
        "throughput_title": "不同批量大小下的吞吐量",
        "x_model": "模型",
        "x_bucket": "时间间隔区间",
        "x_interval": "目标题距最后一次历史作答的时间间隔（天）",
        "x_batch": "批量大小",
        "y_auc": "ROC-AUC（越高越好）",
        "y_accuracy": "Accuracy（越高越好）",
        "y_logloss": "Log Loss（越低越好）",
        "y_prob_observed": "平均预测概率",
        "y_prob_controlled": "平均预测概率",
        "distribution_title": "预测概率分布（各模型对比）",
        "distribution_by_label_title": "预测概率分布（按真实标签分组）",
        "x_y_pred": "预测答对概率 y_pred",
        "y_density": "概率密度",
        "label_correct": "真实答对",
        "label_wrong": "真实答错",
        "y_latency": "延迟（毫秒）",
        "y_latency_per_sample": "单样本延迟（毫秒）",
        "y_throughput": "吞吐量（样本/秒）",
        "y_tpr": "真正率 TPR",
        "x_fpr": "假正率 FPR",
        "series_mean": "平均批次延迟",
        "series_p50": "P50 批次延迟",
        "series_p95": "P95 批次延迟",
        "series_per_sample": "单样本延迟",
        "chance": "随机猜测基线 (AUC=0.5)",
        "sample_count": "样本数",
        "no_value": "不可计算",
        "mock_marker": "斜纹 = 模拟模型",
        "demo": "DEMO · 合成数据",
    },
    "en": {
        "auc_title": "ROC-AUC by model",
        "accuracy_title": "Accuracy by model",
        "logloss_title": "Log Loss by model",
        "roc_title": "ROC curves",
        "time_title": "Time interval vs mean predicted probability (observational)",
        "time_compare_title": "Time response across models (controlled, fixed history)",
        "latency_title": "Inference latency",
        "throughput_title": "Throughput by batch size",
        "x_model": "Model",
        "x_bucket": "Time interval bucket",
        "x_interval": "Interval since last history answer (days)",
        "x_batch": "Batch size",
        "y_auc": "ROC-AUC (higher is better)",
        "y_accuracy": "Accuracy (higher is better)",
        "y_logloss": "Log Loss (lower is better)",
        "y_prob_observed": "Mean predicted probability",
        "y_prob_controlled": "Mean predicted probability",
        "distribution_title": "Predicted probability distribution by model",
        "distribution_by_label_title": "Predicted probability distribution by true label",
        "x_y_pred": "Predicted probability y_pred",
        "y_density": "Density",
        "label_correct": "Correct",
        "label_wrong": "Incorrect",
        "y_latency": "Latency (ms)",
        "y_latency_per_sample": "Per-sample latency (ms)",
        "y_throughput": "Throughput (samples/s)",
        "y_tpr": "True positive rate",
        "x_fpr": "False positive rate",
        "series_mean": "Mean batch latency",
        "series_p50": "P50 batch latency",
        "series_p95": "P95 batch latency",
        "series_per_sample": "Per-sample latency",
        "chance": "Chance level (AUC=0.5)",
        "sample_count": "Samples",
        "no_value": "n/a",
        "mock_marker": "Hatched = mock model",
        "demo": "DEMO · SYNTHETIC DATA",
    },
}

_STATE: dict[str, str] = {"lang": "en"}


def T(key: str) -> str:
    """取当前语言下的图表文案。"""
    return _LABELS[_STATE["lang"]].get(key, key)


@lru_cache(maxsize=1)
def configure_style() -> str:
    """统一图表风格并选择可用字体，返回实际使用的语言代码 ('zh' 或 'en')。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_FONT_CANDIDATES if name in available), None)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": PLOT_DPI,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.unicode_minus": False,
    })
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen, *CJK_FONT_CANDIDATES, "DejaVu Sans"]
        _STATE["lang"] = "zh"
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        _STATE["lang"] = "en"
    return _STATE["lang"]


@dataclass(frozen=True)
class PlotResult:
    name: str
    file: str | None
    status: str
    reason: str | None
    source: str
    demo: bool

    def as_record(self) -> dict:
        return {
            "name": self.name, "file": self.file, "status": self.status,
            "reason": self.reason, "source": self.source, "demo_watermark": self.demo,
        }


@dataclass
class PlotInputs:
    """绘图所需的全部数据，全部来自对应模块的计算结果。

    ``prediction_sources`` 是逐样本的原始预测，ROC 曲线与预测分布图都由它计算，
    因此必须直接来自 ``predictions.csv`` 而不是汇总结果。
    """

    metrics: dict | None = None
    forgetting: dict | None = None
    comparison: dict | None = None
    performance: dict | None = None
    prediction_sources: Sequence[PredictionData] = field(default_factory=tuple)


def _skip(name: str, source: str, reason: str) -> PlotResult:
    return PlotResult(name=name, file=None, status="skipped", reason=reason, source=source, demo=False)


def _watermark(figure) -> None:
    """在图形右下角标注模拟数据，避免图表被误当作真实实验结果。"""
    figure.text(
        0.99, 0.01, DEMO_TEXT, ha="right", va="bottom", fontsize=11, color="#B23A48",
        alpha=0.85, fontweight="bold", rotation=0,
    )


def _save(figure, path: Path, demo: bool) -> None:
    if demo:
        _watermark(figure)
    figure.savefig(path)
    plt.close(figure)


def _annotate_bars(
    axes, bars, values: Sequence[float | None], digits: int = 4, *, rotation: int = 0
) -> None:
    for bar, value in zip(bars, values, strict=True):
        if value is None:
            continue
        axes.annotate(
            f"{value:.{digits}f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9,
            rotation=rotation,
        )


def _metric_rows(metrics: dict | None, metric: str) -> list[dict]:
    """从 metrics.json 中取出给定指标在每个模型 all 组上的取值。"""
    if not metrics:
        return []
    rows = [
        {
            "label": row["label"], "value": row.get(metric), "is_mock": row.get("is_mock"),
            "auc_status": row.get("auc_status"), "auc_reason": row.get("auc_reason"),
        }
        for row in metrics.get("rows", []) if row.get("group") == GROUP_ALL
    ]
    return rows


def _any_mock(rows: Sequence[dict]) -> bool:
    return any(row.get("is_mock") for row in rows)


def plot_metric_comparison(
    rows: Sequence[dict], metric: str, output_path: Path, *, title_key: str, y_key: str,
    source: str, digits: int = 4,
) -> PlotResult:
    """绘制单指标的多模型对比柱状图。"""
    plottable = [row for row in rows if row["value"] is not None]
    if not plottable:
        reasons = {
            row["label"]: row.get("auc_reason") for row in rows if row.get("auc_reason")
        }
        detail = f"；原因：{reasons}" if reasons else ""
        return _skip(f"metric_{metric}", source, f"没有任何模型能计算 {metric}{detail}")
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    labels = [row["label"] for row in plottable]
    values = [float(row["value"]) for row in plottable]
    positions = np.arange(len(labels))
    demo = _any_mock(plottable)
    bars = axes.bar(
        positions, values, width=0.55,
        color=[PALETTE[index % len(PALETTE)] for index in range(len(labels))],
        hatch=[DEMO_HATCH if row["is_mock"] else "" for row in plottable],
        edgecolor="white", linewidth=0.8,
    )
    _annotate_bars(axes, bars, values, digits=digits)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=12 if max(len(x) for x in labels) > 10 else 0)
    axes.set_xlabel(T("x_model"))
    axes.set_ylabel(T(y_key))
    axes.set_title(T(title_key))
    if metric == "roc_auc":
        axes.set_ylim(0.0, 1.05)
        axes.axhline(0.5, color="#999999", linestyle="--", linewidth=1.0)
        axes.annotate(T("chance"), xy=(len(labels) - 0.5, 0.5), xytext=(0, 5),
                      textcoords="offset points", ha="right", fontsize=9, color="#666666")
    skipped = [row["label"] for row in rows if row["value"] is None]
    if skipped:
        axes.annotate(
            f"{T('no_value')}: {', '.join(skipped)}", xy=(0.01, 0.02), xycoords="axes fraction",
            fontsize=9, color="#B23A48",
        )
    if demo:
        axes.annotate(T("mock_marker"), xy=(1.0, 1.02), xycoords="axes fraction",
                      ha="right", fontsize=9, color="#B23A48")
    _save(figure, output_path, demo)
    return PlotResult(f"metric_{metric}", output_path.name, "created", None, source, demo)


def plot_roc_curves(sources: Sequence[PredictionData], output_path: Path) -> PlotResult:
    """绘制多模型 ROC 曲线；单一类别的模型会被跳过并说明原因。"""
    source = "metrics/metrics.json + predictions.csv"
    if not sources:
        return _skip("roc_curve", source, "未提供预测文件，无法计算 ROC 曲线")
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    drawn, skipped, demo = 0, [], False
    for index, data in enumerate(sources):
        labels = data.frame["y_true"].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2:
            skipped.append(f"{data.label}（单一类别，ROC 无定义）")
            continue
        probabilities = data.frame["y_pred"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        axes.plot(
            fpr, tpr, color=PALETTE[index % len(PALETTE)], linewidth=2.0,
            label=f"{data.label} (AUC={sk_auc(fpr, tpr):.4f})",
        )
        drawn += 1
        demo = demo or bool(data.is_mock)
    if drawn == 0:
        plt.close(figure)
        return _skip("roc_curve", source, f"所有模型都无法绘制 ROC 曲线：{skipped}")
    axes.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0, label=T("chance"))
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.set_xlabel(T("x_fpr"))
    axes.set_ylabel(T("y_tpr"))
    axes.set_title(T("roc_title"))
    axes.legend(loc="lower right")
    if skipped:
        axes.annotate(f"{T('no_value')}: {', '.join(skipped)}", xy=(0.02, 0.96),
                      xycoords="axes fraction", fontsize=9, color="#B23A48", va="top")
    _save(figure, output_path, demo)
    return PlotResult("roc_curve", output_path.name, "created", None, source, demo)


def plot_prediction_distribution(
    sources: Sequence[PredictionData], output_path: Path, *, threshold: float = 0.5
) -> PlotResult:
    """绘制预测概率分布图。

    左图：各模型 ``y_pred`` 的概率分布叠加对比，并标出分类阈值。
    右图：第一个模型按真实标签分组的 ``y_pred`` 分布，用于观察模型是否把
    答对与答错的样本分开——两组分布重叠越多，说明区分能力越弱。
    """
    source = "predictions.csv"
    if not sources:
        return _skip("prediction_distribution", source, "未提供预测文件，无法绘制预测分布图")
    bins = np.linspace(0.0, 1.0, 21)
    figure, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)
    demo = any(bool(item.is_mock) for item in sources)

    for index, data in enumerate(sources):
        probabilities = data.frame["y_pred"].to_numpy(dtype=float)
        left.hist(
            probabilities, bins=bins, density=True, histtype="stepfilled", alpha=0.45,
            color=PALETTE[index % len(PALETTE)], edgecolor=PALETTE[index % len(PALETTE)],
            linewidth=1.4, label=f"{data.label}（n={data.sample_count}）",
        )
    left.axvline(threshold, color="#B23A48", linestyle="--", linewidth=1.4)
    # 标注放在坐标区底部，避免与上方的图例文字重叠。
    left.annotate(
        f"threshold={threshold:g}", xy=(threshold, 0.0), xycoords=("data", "axes fraction"),
        xytext=(4, 4), textcoords="offset points", fontsize=9, color="#B23A48", va="bottom",
    )
    left.set_xlim(0.0, 1.0)
    left.set_xlabel(T("x_y_pred"))
    left.set_ylabel(T("y_density"))
    left.set_title(T("distribution_title"))
    left.legend(loc="upper center")

    primary = sources[0]
    for label_value, colour, name_key in ((1, PALETTE[2], "label_correct"), (0, PALETTE[3], "label_wrong")):
        subset = primary.frame.loc[primary.frame["y_true"] == label_value, "y_pred"]
        if subset.empty:
            continue
        right.hist(
            subset.to_numpy(dtype=float), bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, color=colour, edgecolor=colour, linewidth=1.4,
            label=f"{T(name_key)}（n={len(subset)}）",
        )
    right.axvline(threshold, color="#B23A48", linestyle="--", linewidth=1.4)
    right.set_xlim(0.0, 1.0)
    right.set_xlabel(T("x_y_pred"))
    right.set_ylabel(T("y_density"))
    right.set_title(f"{T('distribution_by_label_title')} — {primary.label}")
    right.legend(loc="upper center")

    _save(figure, output_path, demo)
    return PlotResult("prediction_distribution", output_path.name, "created", None, source, demo)


def plot_time_response(bucket_rows: Sequence[dict], output_path: Path) -> PlotResult:
    """绘制观察性时间间隔—预测概率曲线（来自遗忘测试 A 部分）。"""
    source = "forgetting/interval_buckets.csv"
    if not bucket_rows:
        return _skip("forgetting_observed", source, "没有可用的时间间隔分桶结果")
    figure, axes = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    positions = np.arange(len(bucket_rows))
    means = np.array([row["mean_y_pred"] for row in bucket_rows], dtype=float)
    errors = np.array([row["std_y_pred"] for row in bucket_rows], dtype=float)
    counts = [int(row["sample_count"]) for row in bucket_rows]
    demo = _any_mock(bucket_rows)
    axes.bar(positions, means, width=0.55, color=PALETTE[0], yerr=errors, capsize=4,
             error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
             hatch=DEMO_HATCH if demo else "", edgecolor="white")
    axes.set_xticks(positions)
    axes.set_xticklabels([row["bucket"] for row in bucket_rows], rotation=15, ha="right")
    axes.set_xlabel(T("x_bucket"))
    axes.set_ylabel(T("y_prob_observed"))
    axes.set_title(T("time_title"))
    for position, mean, count in zip(positions, means, counts, strict=True):
        axes.annotate(f"{mean:.3f}\n{count} {T('sample_count')}",
                      xy=(position, mean), xytext=(0, 10), textcoords="offset points",
                      ha="center", fontsize=9)
    # 留出空间放置柱顶标注；全为 0 时也不会产生退化坐标轴。
    top = float(np.max(means + errors)) if len(means) else 0.0
    axes.set_ylim(0.0, max(0.1, min(1.0, top * 1.45)))
    _save(figure, output_path, demo)
    return PlotResult("forgetting_observed", output_path.name, "created", None, source, demo)


def plot_model_time_response(entries: Sequence[dict], output_path: Path) -> PlotResult:
    """绘制多模型受控时间响应对比曲线（来自遗忘测试 B 部分）。"""
    source = "forgetting/forgetting_summary.json"
    available = [entry for entry in entries if entry.get("available")]
    if len(available) < 2:
        reason = (
            "只有 1 个模型具备受控时间响应数据，无法绘制模型间对比图；"
            "请通过 --controlled-model 提供至少两个模型"
            if len(available) == 1 else "没有任何模型具备受控时间响应数据"
        )
        return _skip("forgetting_model_compare", source, reason)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    demo = False
    for index, entry in enumerate(available):
        intervals = [item["interval_days"] for item in entry["per_interval"]]
        means = [item["mean_probability"] for item in entry["per_interval"]]
        errors = [item["std_probability"] for item in entry["per_interval"]]
        is_mock = bool(entry["model"]["is_mock"])
        demo = demo or is_mock
        axes.errorbar(
            intervals, means, yerr=errors, marker="o", markersize=6, linewidth=2.0, capsize=4,
            color=PALETTE[index % len(PALETTE)],
            linestyle="--" if is_mock else "-",
            label=f"{entry['label']}（{'模拟' if is_mock else '外部'}模型）",
        )
    axes.set_xlabel(T("x_interval"))
    axes.set_ylabel(T("y_prob_controlled"))
    axes.set_title(T("time_compare_title"))
    axes.set_ylim(-0.02, 1.02)
    axes.legend(loc="best")
    if demo:
        axes.annotate(T("mock_marker"), xy=(1.0, 1.02), xycoords="axes fraction",
                      ha="right", fontsize=9, color="#B23A48")
    _save(figure, output_path, demo)
    return PlotResult("forgetting_model_compare", output_path.name, "created", None, source, demo)


def plot_latency(rows: Sequence[dict], output_path: Path) -> PlotResult:
    """绘制批次延迟与单样本延迟对比图（来自性能测试）。"""
    source = "performance/performance.csv"
    if not rows:
        return _skip("latency", source, "没有可用的推理性能结果")
    figure, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE_DOUBLE)
    positions = np.arange(len(rows))
    width = 0.26
    demo = _any_mock(rows)
    series = (
        ("mean_batch_latency_ms", "series_mean", -width, PALETTE[0]),
        ("p50_batch_latency_ms", "series_p50", 0.0, PALETTE[2]),
        ("p95_batch_latency_ms", "series_p95", width, PALETTE[3]),
    )
    for key, label_key, offset, color in series:
        values = [float(row[key]) for row in rows]
        bars = left.bar(positions + offset, values, width=width, color=color,
                        label=T(label_key), hatch=DEMO_HATCH if demo else "",
                        edgecolor="white", linewidth=0.6)
        # 同组三根柱较窄，竖排标注避免数字互相覆盖。
        _annotate_bars(left, bars, values, digits=3, rotation=90)
    left.set_xticks(positions)
    left.set_xticklabels([str(row["batch_size"]) for row in rows])
    left.set_xlabel(T("x_batch"))
    left.set_ylabel(T("y_latency"))
    left.set_title(f"{T('latency_title')} — {T('series_mean')} / P50 / P95")
    left.legend(loc="upper left")

    per_sample = [float(row["mean_per_sample_latency_ms"]) for row in rows]
    bars = right.bar(positions, per_sample, width=0.5, color=PALETTE[1],
                     hatch=DEMO_HATCH if demo else "", edgecolor="white")
    _annotate_bars(right, bars, per_sample, digits=4)
    right.set_xticks(positions)
    right.set_xticklabels([str(row["batch_size"]) for row in rows])
    right.set_xlabel(T("x_batch"))
    right.set_ylabel(T("y_latency_per_sample"))
    right.set_title(f"{T('latency_title')} — {T('series_per_sample')}")
    _save(figure, output_path, demo)
    return PlotResult("latency", output_path.name, "created", None, source, demo)


def plot_throughput(rows: Sequence[dict], output_path: Path) -> PlotResult:
    """绘制不同批量大小下的吞吐量曲线（来自性能测试）。"""
    source = "performance/performance.csv"
    if not rows:
        return _skip("throughput", source, "没有可用的推理性能结果")
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    batch_sizes = [row["batch_size"] for row in rows]
    throughput = [float(row["throughput_samples_per_second"]) for row in rows]
    demo = _any_mock(rows)
    axes.plot(batch_sizes, throughput, marker="o", markersize=7, linewidth=2.2, color=PALETTE[0])
    for size, value in zip(batch_sizes, throughput, strict=True):
        axes.annotate(f"{value:.1f}", xy=(size, value), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=9)
    axes.set_xlabel(T("x_batch"))
    axes.set_ylabel(T("y_throughput"))
    axes.set_title(T("throughput_title"))
    axes.set_xticks(sorted(set(batch_sizes)))
    axes.set_ylim(0.0, float(max(throughput)) * 1.2 if throughput else 1.0)
    _save(figure, output_path, demo)
    return PlotResult("throughput", output_path.name, "created", None, source, demo)


def generate_plots(
    inputs: PlotInputs, output_dir: str | Path, *, overwrite: bool = False
) -> dict:
    """生成全部可计算的图表，写出 plots_manifest.json 记录成功与跳过原因。"""
    configure_style()
    plots_dir = ensure_output_dir(Path(output_dir).expanduser().resolve() / "plots",
                                  overwrite=overwrite)
    results: list[PlotResult] = []

    for metric, name, title, y_key in (
        ("roc_auc", "auc", "auc_title", "y_auc"),
        ("accuracy", "accuracy", "accuracy_title", "y_accuracy"),
        ("log_loss", "log_loss", "logloss_title", "y_logloss"),
    ):
        if not inputs.metrics:
            results.append(_skip(f"metric_{name}", "metrics/metrics.json",
                                 "未提供 metrics.json，请先运行 metrics 子命令"))
            continue
        results.append(plot_metric_comparison(
            _metric_rows(inputs.metrics, metric), metric, plots_dir / f"{name}_comparison.png",
            title_key=title, y_key=y_key, source="metrics/metrics.json", digits=4,
        ))

    results.append(plot_roc_curves(inputs.prediction_sources, plots_dir / "roc_curves.png"))
    results.append(plot_prediction_distribution(
        inputs.prediction_sources, plots_dir / "prediction_distribution.png"
    ))

    descriptive = (inputs.forgetting or {}).get("descriptive") or {}
    if descriptive.get("available"):
        results.append(plot_time_response(descriptive["bucket_rows"],
                                          plots_dir / "forgetting_observed.png"))
    else:
        results.append(_skip(
            "forgetting_observed", "forgetting/interval_buckets.csv",
            descriptive.get("reason", "未提供遗忘测试的描述性分析结果"),
        ))

    controlled = (inputs.forgetting or {}).get("controlled_models") or []
    results.append(plot_model_time_response(controlled, plots_dir / "forgetting_model_compare.png"))

    performance_rows = (inputs.performance or {}).get("rows") or []
    results.append(plot_latency(performance_rows, plots_dir / "inference_latency.png"))
    results.append(plot_throughput(performance_rows, plots_dir / "throughput_by_batch.png"))

    payload = {
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": utc_now_iso(),
        "font_language": _STATE["lang"],
        "dpi": PLOT_DPI,
        "plots": [result.as_record() for result in results],
        "created": [result.name for result in results if result.status == "created"],
        "skipped": [
            {"name": result.name, "reason": result.reason}
            for result in results if result.status == "skipped"
        ],
    }
    write_document(plots_dir / "plots_manifest.json", payload)
    return payload


def load_artifacts(evaluation_dir: str | Path) -> dict[str, dict | None]:
    """从评估输出目录读取各模块已写出的 JSON 结果，供绘图与报告复用。"""
    root = Path(evaluation_dir).expanduser().resolve()
    mapping = {
        "metrics": root / "metrics" / "metrics.json",
        "forgetting": root / "forgetting" / "forgetting_summary.json",
        "comparison": root / "comparison" / "model_comparison.json",
        "performance": root / "performance" / "performance.json",
    }
    loaded: dict[str, dict | None] = {}
    for key, path in mapping.items():
        if path.is_file():
            loaded[key] = read_json_document(path, description=f"{key} 结果")
        else:
            loaded[key] = None
    return loaded


def require_plot_inputs(inputs: PlotInputs) -> None:
    """在没有提供任何绘图数据时给出明确错误，而不是静默生成空目录。"""
    if not any((inputs.metrics, inputs.forgetting, inputs.comparison,
                inputs.performance, inputs.prediction_sources)):
        raise EvaluationError("没有可供绘图的数据，请先运行 metrics / forgetting / performance")
