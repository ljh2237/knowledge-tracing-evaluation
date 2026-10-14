"""模块二：遗忘机制专项测试。

包含两种**性质完全不同**、不可互相替代的测试：

A. 现有预测结果的描述性分析（``describe_time_response``）
   读取 ``predictions.csv`` 中已有的 ``delta_time_seconds`` 与 ``y_pred``，按时间
   间隔分组统计。这是**观察性分析**：没有控制学生、知识点、题目难度与历史长度，
   因此不能作为遗忘的因果证据。只有 ``event_order`` 的数据会因缺少真实时间间隔
   而被整体跳过，绝不会把事件顺序伪造成天数。

B. 固定学习历史的受控时间测试（``controlled_time_response``）
   复用负责人 A 的 ``PredictionRequest`` / ``ModelAdapter`` 接口，对同一份固定的
   历史学习状态与同一道目标题目，仅改变目标时点与最后一次历史作答之间的时间间隔
   （例如 0、1、3、7、14 天），观察预测概率如何变化。因果顺序始终成立：目标时点
   永远不早于任何历史答案的可见时点。

``MockModel`` 没有显式遗忘机制，其概率不随时间变化是**合理的测试结果**，不得为了
得到下降曲线而伪造数据。真实遗忘机制的有效性需要训练组提供相应模型与数据。
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Interval

from contracts import PredictionRequest
from data_loader import SkillVocabulary, load_data
from evaluation.common import (
    EVALUATION_VERSION,
    EvaluationError,
    PredictionData,
    ensure_output_dir,
    load_predictions,
    resolve_predictions,
    utc_now_iso,
    write_document,
    write_table,
)
from model_adapter import ModelAdapter
from pipeline import iter_requests

SECONDS_PER_DAY = 86400.0

#: 默认时间间隔分桶边界（秒），最后一桶为开区间上界。
DEFAULT_BUCKET_EDGES: tuple[float, ...] = (
    0.0, 60.0, 5 * 60.0, 60 * 60.0, 6 * 60 * 60.0,
    24 * 60 * 60.0, 7 * 24 * 60 * 60.0, 30 * 24 * 60 * 60.0,
)

#: 默认受控测试的时间间隔，单位为天。
DEFAULT_INTERVAL_DAYS: tuple[float, ...] = (0.0, 1.0, 3.0, 7.0, 14.0)

UNIT_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": SECONDS_PER_DAY}

#: 概率变化小于该值就视为「无时间响应」，避免浮点误差被当成机制。
RESPONSE_TOLERANCE = 1e-9

INTERVAL_COLUMNS = (
    "bucket", "bucket_lower_seconds", "bucket_upper_seconds", "sample_count", "sample_share",
    "min_delta_seconds", "max_delta_seconds", "mean_delta_seconds", "median_delta_seconds",
    "mean_y_pred", "median_y_pred", "std_y_pred", "mean_y_true", "accuracy", "label",
    "model_name", "is_mock",
)

CONTROLLED_COLUMNS = (
    "label", "candidate_record_id", "student_id", "question_id", "skill_id", "interval_seconds",
    "interval_days", "repetition", "probability", "history_length", "base_delta_seconds",
    "model_name", "is_mock",
)

DESCRIPTIVE_CAVEAT = (
    "观察性分析：仅按已有预测结果的时间间隔分组统计，未控制学生、知识点、题目难度与历史长度，"
    "因此不能据此断言遗忘的因果效应。仅使用 event_order 的数据不含真实时间，已整体排除，未伪造天数。"
)

CONTROLLED_CAVEAT = (
    "受控性说明：同一候选样本的历史学习状态、目标题目与知识点完全固定，仅改变目标时点与最后一次"
    "历史作答之间的时间间隔；目标时点始终不早于历史答案可见时点，因果顺序成立。该测试检验的是"
    "「模型是否对时间间隔作出响应」，不是「模型在真实数据上的遗忘效果」。"
)


def parse_intervals(text: str, unit: str = "days") -> tuple[float, ...]:
    """解析 ``--intervals`` 文本，例如 ``0,1,3,7,14``，返回升序去重的秒数元组。"""
    if unit not in UNIT_SECONDS:
        raise EvaluationError(f"时间单位必须是 {sorted(UNIT_SECONDS)} 之一，实际为 {unit!r}")
    if not isinstance(text, str) or not text.strip():
        raise EvaluationError("--intervals 不能为空，例如 --intervals 0,1,3,7,14 --unit days")
    values: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            raise EvaluationError(f"--intervals 含空项: {text!r}")
        try:
            value = float(chunk)
        except ValueError as exc:
            raise EvaluationError(f"--intervals 含非数值项 {chunk!r}") from exc
        if not math.isfinite(value) or value < 0:
            raise EvaluationError(f"时间间隔必须是非负有限数，实际为 {chunk!r}")
        values.append(value * UNIT_SECONDS[unit])
    if len(set(values)) != len(values):
        raise EvaluationError(f"--intervals 存在重复项: {text!r}")
    return tuple(sorted(values))


def format_duration(seconds: float) -> str:
    """把秒数格式化为适合图表与表格的紧凑中文单位。"""
    if seconds < 60:
        return f"{seconds:g}秒"
    if seconds < 3600:
        return f"{seconds / 60:g}分"
    if seconds < SECONDS_PER_DAY:
        return f"{seconds / 3600:g}小时"
    return f"{seconds / SECONDS_PER_DAY:g}天"


def _interval_label(interval: Interval) -> str:
    upper = "∞" if math.isinf(interval.right) else format_duration(float(interval.right))
    return f"[{format_duration(float(interval.left))}, {upper})"


def _safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    """相关系数在样本过少或某一维恒定时无定义，返回 None 而不是编造数值。"""
    if left.size < 3 or right.size < 3:
        return None
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(pd.Series(left).corr(pd.Series(right), method=method))
    return value if math.isfinite(value) else None


def describe_time_response(
    data: PredictionData,
    *,
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    threshold: float = 0.5,
) -> dict:
    """A 部分：对已有预测结果做时间间隔的描述性分组统计。"""
    if not data.has_delta_time:
        return {
            "available": False,
            "reason": "predictions.csv 中不存在 delta_time_seconds 列，无法进行时间间隔分析",
            "caveat": DESCRIPTIVE_CAVEAT,
        }
    edges = tuple(float(edge) for edge in bucket_edges)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=False)):
        raise EvaluationError("分桶边界必须严格递增且至少包含两个边界")

    frame = data.frame
    total = int(len(frame))
    usable = frame[frame["delta_time_seconds"].notna()]
    if usable.empty:
        return {
            "available": False,
            "reason": (
                "没有任何样本带有有效的真实时间间隔；该数据集可能仅有 event_order，"
                "按要求不使用事件顺序伪造天数"
            ),
            "caveat": DESCRIPTIVE_CAVEAT,
            "total_samples": total,
            "usable_samples": 0,
            "excluded_samples": total,
        }

    buckets = pd.cut(usable["delta_time_seconds"], bins=[*edges, math.inf], right=False)
    rows: list[dict] = []
    for interval in buckets.cat.categories:
        subset = usable[buckets == interval]
        if subset.empty:
            continue
        probabilities = subset["y_pred"].to_numpy(dtype=float)
        labels_true = subset["y_true"].to_numpy(dtype=int)
        rows.append({
            "bucket": _interval_label(interval),
            "bucket_lower_seconds": float(interval.left),
            "bucket_upper_seconds": None if math.isinf(interval.right) else float(interval.right),
            "sample_count": int(len(subset)),
            "sample_share": len(subset) / len(usable),
            "min_delta_seconds": float(subset["delta_time_seconds"].min()),
            "max_delta_seconds": float(subset["delta_time_seconds"].max()),
            "mean_delta_seconds": float(subset["delta_time_seconds"].mean()),
            "median_delta_seconds": float(subset["delta_time_seconds"].median()),
            "mean_y_pred": float(probabilities.mean()),
            "median_y_pred": float(np.median(probabilities)),
            "std_y_pred": float(probabilities.std(ddof=0)),
            "mean_y_true": float(labels_true.mean()),
            "accuracy": float(((probabilities >= threshold).astype(int) == labels_true).mean()),
            "label": data.label,
            "model_name": data.model_names[0] if data.model_names else "",
            "is_mock": data.is_mock,
        })

    deltas = usable["delta_time_seconds"].to_numpy(dtype=float)
    probabilities = usable["y_pred"].to_numpy(dtype=float)
    warm = usable[~usable["cold_start"]] if data.has_cold_start else usable
    return {
        "available": True,
        "total_samples": total,
        "usable_samples": int(len(usable)),
        "excluded_samples": total - int(len(usable)),
        "excluded_reason": (
            "缺少有效真实时间间隔（冷启动或仅有 event_order）；这些样本不参与时间分组"
        ),
        "time_source": ",".join(data.time_sources) if data.time_sources else "unknown",
        "threshold": float(threshold),
        "correlations": {
            "pearson_delta_vs_y_pred": _safe_correlation(deltas, probabilities, "pearson"),
            "spearman_delta_vs_y_pred": _safe_correlation(deltas, probabilities, "spearman"),
            "spearman_delta_vs_y_pred_non_cold_start": _safe_correlation(
                warm["delta_time_seconds"].to_numpy(dtype=float),
                warm["y_pred"].to_numpy(dtype=float), "spearman",
            ),
        },
        "bucket_rows": rows,
        "model": {
            "label": data.label,
            "model_name": data.model_names[0] if data.model_names else "",
            "is_mock": data.is_mock,
            "data_nature": data.data_nature,
        },
        "caveat": DESCRIPTIVE_CAVEAT,
    }


def select_candidates(
    requests: Sequence[PredictionRequest], *, max_requests: int | None, seed: int
) -> list[PredictionRequest]:
    """挑选具备固定历史与真实时间间隔的受控测试对象，结果可复现。"""
    if max_requests is not None and max_requests < 1:
        raise EvaluationError("max_requests 必须为正整数或不限制")
    candidates = [
        request for request in requests
        if request.target.timestamp is not None
        and request.history
        and request.delta_time_seconds is not None
        and request.history[-1].available_at is not None
    ]
    candidates.sort(key=lambda request: request.target.record_id)
    if max_requests is not None and len(candidates) > max_requests:
        # 用固定种子抽样，保证同一 seed 下复现同一批候选样本。
        chosen = sorted(random.Random(seed).sample(range(len(candidates)), max_requests))
        candidates = [candidates[index] for index in chosen]
    return candidates


def controlled_time_response(
    input_path: str | Path,
    adapter: ModelAdapter,
    *,
    intervals_seconds: Sequence[float] = tuple(value * SECONDS_PER_DAY for value in DEFAULT_INTERVAL_DAYS),
    repeats: int = 3,
    max_requests: int | None = 24,
    max_history: int = 256,
    batch_size: int = 64,
    invalid_policy: str = "error",
    timestamp_unit: str = "iso",
    vocabulary: SkillVocabulary | None = None,
    unknown_skill_policy: str = "error",
    seed: int = 20261013,
    label: str = "",
) -> dict:
    """B 部分：固定学习历史的受控时间响应测试。"""
    if repeats < 1:
        raise EvaluationError("repeats 必须为正整数")
    if batch_size < 1:
        raise EvaluationError("batch_size 必须为正整数")
    intervals = tuple(float(value) for value in intervals_seconds)
    if not intervals or any(value < 0 or not math.isfinite(value) for value in intervals):
        raise EvaluationError("时间间隔必须是非负有限数序列")

    loaded = load_data(
        input_path, invalid_policy=invalid_policy, timestamp_unit=timestamp_unit,
        vocabulary=vocabulary, unknown_skill_policy=unknown_skill_policy,
    )
    candidates = select_candidates(
        list(iter_requests(loaded.records, max_history)), max_requests=max_requests, seed=seed
    )
    resolved_label = label or adapter.model.name
    model_info = {
        "model_name": adapter.model.name,
        "is_mock": adapter.model.is_mock,
        "data_nature": (
            "模拟模型输出（仅用于程序功能验证）" if adapter.model.is_mock else "外部模型输出"
        ),
    }
    input_info = {
        "file": Path(input_path).name,
        "accepted_rows": loaded.summary["accepted_rows"],
        "students": loaded.summary["students"],
        "time_source": loaded.summary["time_source"],
    }
    if not candidates:
        return {
            "label": resolved_label,
            "available": False,
            "reason": (
                "没有同时具备「非空可见历史」「真实时间戳」「有效时间间隔」的请求，"
                "无法构造固定历史的受控测试；仅有 event_order 的数据不适用本测试"
            ),
            "model": model_info,
            "input": input_info,
            "caveat": CONTROLLED_CAVEAT,
        }

    conditions: list[dict] = []
    for repetition in range(1, repeats + 1):
        for candidate in candidates:
            last_available = float(candidate.history[-1].available_at)
            for interval in intervals:
                unique_id = f"{candidate.target.record_id}#t{interval:g}s#r{repetition}"
                target = replace(
                    candidate.target, record_id=unique_id, timestamp=last_available + interval
                )
                conditions.append({
                    "candidate": candidate,
                    "repetition": repetition,
                    "interval_seconds": interval,
                    "request": PredictionRequest(
                        target=target, history=candidate.history, delta_time_seconds=float(interval)
                    ),
                })

    probabilities: dict[str, float] = {}
    for start in range(0, len(conditions), batch_size):
        batch = conditions[start:start + batch_size]
        predictions = adapter.predict([item["request"] for item in batch])
        for item, prediction in zip(batch, predictions, strict=True):
            probabilities[item["request"].target.record_id] = prediction.probability

    detail_rows: list[dict] = []
    for item in conditions:
        candidate = item["candidate"]
        detail_rows.append({
            "label": resolved_label,
            "candidate_record_id": candidate.target.record_id,
            "student_id": candidate.target.student_id,
            "question_id": candidate.target.question_id,
            "skill_id": ";".join(candidate.target.skill_tokens),
            "interval_seconds": item["interval_seconds"],
            "interval_days": item["interval_seconds"] / SECONDS_PER_DAY,
            "repetition": item["repetition"],
            "probability": probabilities[item["request"].target.record_id],
            "history_length": len(candidate.history),
            "base_delta_seconds": candidate.delta_time_seconds,
            "model_name": adapter.model.name,
            "is_mock": adapter.model.is_mock,
        })

    frame = pd.DataFrame(detail_rows)
    spread = frame.groupby(["candidate_record_id", "interval_seconds"])["probability"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    interval_index = spread.index.get_level_values("interval_seconds")

    per_interval: list[dict] = []
    for interval in intervals:
        subset = frame[frame["interval_seconds"] == interval]
        values = subset["probability"].to_numpy(dtype=float)
        spread_subset = spread[interval_index == interval]
        per_interval.append({
            "interval_seconds": interval,
            "interval_days": interval / SECONDS_PER_DAY,
            "interval_label": format_duration(interval),
            "candidate_count": int(subset["candidate_record_id"].nunique()),
            "observation_count": int(len(subset)),
            "mean_probability": float(values.mean()),
            "median_probability": float(np.median(values)),
            "std_probability": float(values.std(ddof=0)),
            "min_probability": float(values.min()),
            "max_probability": float(values.max()),
            "max_repeat_spread": float(spread_subset.max()) if len(spread_subset) else 0.0,
        })

    baseline_value = per_interval[0]["mean_probability"]
    max_change = 0.0
    for entry in per_interval:
        change = entry["mean_probability"] - baseline_value
        entry["change_vs_baseline"] = change
        max_change = max(max_change, abs(change))
    means = [entry["mean_probability"] for entry in per_interval]
    monotonic_decreasing = all(
        later <= earlier + RESPONSE_TOLERANCE for earlier, later in zip(means[:-1], means[1:], strict=False)
    )

    caveats = [CONTROLLED_CAVEAT]
    if max_change <= RESPONSE_TOLERANCE:
        caveats.append(
            f"在 {len(candidates)} 个候选样本、{len(intervals)} 个时间间隔、{repeats} 次重复下，"
            f"预测概率相对基线的最大变化为 {max_change:.3g}，未检测到时间响应。"
            "这是合理结果：负责人 A 的 MockModel 没有遗忘机制，本测试不得据此制造下降曲线。"
        )
    if adapter.model.is_mock:
        caveats.append(
            "模型声明 is_mock=True，本次结果只能证明受控测试程序可运行，"
            "不能证明任何遗忘机制的有效性。"
        )

    return {
        "label": resolved_label,
        "available": True,
        "model": model_info,
        "input": input_info,
        "design": {
            "description": (
                "固定同一学生的历史学习状态、目标题目与知识点，只改变目标时点与最后一次"
                "历史作答的时间间隔"
            ),
            "intervals_seconds": list(intervals),
            "intervals_days": [value / SECONDS_PER_DAY for value in intervals],
            "repeats": repeats,
            "sampling_seed": seed,
            "candidate_count": len(candidates),
            "predictions_per_candidate": len(intervals) * repeats,
            "total_inference_calls": len(conditions),
            "history_fixed": True,
            "causal_order_preserved": True,
            "unique_record_id_note": (
                "受控请求使用 <原 record_id>#t<间隔秒数>s#r<重复序号> 生成唯一编号，"
                "避免同一批次内 record_id 冲突；模型不应依赖 record_id 本身。"
            ),
        },
        "baseline_interval_seconds": intervals[0],
        "per_interval": per_interval,
        "response": {
            "detected": bool(max_change > RESPONSE_TOLERANCE),
            "tolerance": RESPONSE_TOLERANCE,
            "max_abs_change_vs_baseline": float(max_change),
            "monotonic_decreasing": bool(monotonic_decreasing and max_change > RESPONSE_TOLERANCE),
            "spearman_interval_vs_mean_probability": _safe_correlation(
                np.array([entry["interval_seconds"] for entry in per_interval], dtype=float),
                np.array(means, dtype=float), "spearman",
            ),
        },
        "caveats": caveats,
        "detail_rows": detail_rows,
    }


def run_forgetting(
    output_dir: str | Path,
    *,
    predictions: str | Path | None = None,
    input_path: str | Path | None = None,
    controlled_models: Sequence[tuple[str, ModelAdapter]] = (),
    intervals_seconds: Sequence[float] | None = None,
    repeats: int = 3,
    max_requests: int | None = 24,
    max_history: int = 256,
    batch_size: int = 64,
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    threshold: float = 0.5,
    invalid_policy: str = "error",
    timestamp_unit: str = "iso",
    vocabulary: SkillVocabulary | None = None,
    unknown_skill_policy: str = "error",
    seed: int = 20261013,
    overwrite: bool = False,
) -> dict:
    """执行遗忘机制专项测试的全部可用部分，写出 CSV 与 JSON 结果。

    A 部分需要 ``predictions``；B 部分需要 ``input_path`` 与一组 ``controlled_models``
    （标签与适配器）。缺少任一部分输入时该部分标记为不可用并说明原因，不生成替代数据。
    多个受控模型共用同一批候选样本与同一组时间间隔，因此可以直接绘制时间响应对比图。

    先完成全部分析再创建输出目录：输入非法时不会留下任何产物目录。
    """
    if intervals_seconds is None:
        intervals_seconds = tuple(value * SECONDS_PER_DAY for value in DEFAULT_INTERVAL_DAYS)

    if predictions is None:
        descriptive: dict = {
            "available": False,
            "reason": "未提供预测文件（--predictions），跳过描述性时间间隔分析",
            "caveat": DESCRIPTIVE_CAVEAT,
        }
    else:
        descriptive = describe_time_response(
            load_predictions(resolve_predictions(predictions)),
            bucket_edges=bucket_edges, threshold=threshold,
        )

    controlled_entries: list[dict] = []
    if input_path is None or not controlled_models:
        controlled_entries.append({
            "label": "",
            "available": False,
            "reason": "未提供原始输入 CSV 与模型适配器，跳过固定历史受控时间测试",
            "caveat": CONTROLLED_CAVEAT,
        })
    else:
        for label, adapter in controlled_models:
            controlled_entries.append(controlled_time_response(
                input_path, adapter, intervals_seconds=intervals_seconds, repeats=repeats,
                max_requests=max_requests, max_history=max_history, batch_size=batch_size,
                invalid_policy=invalid_policy, timestamp_unit=timestamp_unit,
                vocabulary=vocabulary, unknown_skill_policy=unknown_skill_policy, seed=seed,
                label=label,
            ))

    forgetting_dir = ensure_output_dir(
        Path(output_dir).expanduser().resolve() / "forgetting", overwrite=overwrite
    )
    if descriptive.get("available"):
        write_table(
            forgetting_dir / "interval_buckets.csv", descriptive["bucket_rows"], INTERVAL_COLUMNS
        )
    detail_rows = [
        row for entry in controlled_entries for row in entry.get("detail_rows", [])
    ]
    if detail_rows:
        write_table(
            forgetting_dir / "controlled_time_response.csv", detail_rows, CONTROLLED_COLUMNS
        )
    # 明细行只写入 CSV，JSON 保留可读的汇总结构，避免重复与体积膨胀。
    payload = {
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": utc_now_iso(),
        "descriptive": descriptive,
        "controlled_models": [
            {key: value for key, value in entry.items() if key != "detail_rows"}
            for entry in controlled_entries
        ],
    }
    write_document(forgetting_dir / "forgetting_summary.json", payload)
    return payload


def _format_optional(value: float | None) -> str:
    return "不可计算" if value is None else f"{value:.4f}"


def build_descriptive_markdown(descriptive: dict) -> str:
    """把 A 部分结果渲染为 Markdown 片段，供报告生成器复用。"""
    if not descriptive.get("available"):
        return f"不适用：{descriptive.get('reason', '缺少数据')}\n"
    lines = [
        f"- 有效样本：{descriptive['usable_samples']} / {descriptive['total_samples']}"
        f"（排除 {descriptive['excluded_samples']} 条，{descriptive['excluded_reason']}）",
        f"- 时间来源：`{descriptive['time_source']}`；分类阈值：{descriptive['threshold']}",
        "",
        "| 时间间隔区间 | 样本数 | 占比 | 平均预测概率 | 中位预测概率 | 标准差 | 真实通过率 | 准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in descriptive["bucket_rows"]:
        lines.append(
            f"| {row['bucket']} | {row['sample_count']} | {row['sample_share']:.1%} | "
            f"{row['mean_y_pred']:.4f} | {row['median_y_pred']:.4f} | {row['std_y_pred']:.4f} | "
            f"{row['mean_y_true']:.4f} | {row['accuracy']:.4f} |"
        )
    correlations = descriptive["correlations"]
    lines += [
        "",
        "相关系数（观察性，仅供参考）："
        f"Pearson(delta, y_pred) = {_format_optional(correlations['pearson_delta_vs_y_pred'])}，"
        f"Spearman(delta, y_pred) = {_format_optional(correlations['spearman_delta_vs_y_pred'])}，"
        "非冷启动样本 Spearman = "
        f"{_format_optional(correlations['spearman_delta_vs_y_pred_non_cold_start'])}。",
        "",
        f"> {descriptive['caveat']}",
        "",
    ]
    return "\n".join(lines)


def build_controlled_markdown(controlled: dict) -> str:
    """把 B 部分结果渲染为 Markdown 片段，供报告生成器复用。"""
    if not controlled.get("available"):
        return f"不适用：{controlled.get('reason', '缺少数据')}\n"
    design = controlled["design"]
    response = controlled["response"]
    model = controlled["model"]
    lines = [
        f"- 测试模型：`{model['model_name']}`（is_mock={model['is_mock']}，{model['data_nature']}）",
        f"- 输入数据：`{controlled['input']['file']}`，"
        f"时间来源 `{controlled['input']['time_source']}`，"
        f"接受记录 {controlled['input']['accepted_rows']} 条",
        f"- 候选样本：{design['candidate_count']} 个；每个样本"
        f" {len(design['intervals_seconds'])} 个时间间隔 × {design['repeats']} 次重复；"
        f"总推理次数 {design['total_inference_calls']}",
        f"- 时间间隔（秒）：{', '.join(f'{value:g}' for value in design['intervals_seconds'])}",
        f"- 检测到时间响应：**{'是' if response['detected'] else '否'}**；"
        f"相对基线的最大绝对变化 {response['max_abs_change_vs_baseline']:.3g}；"
        f"单调下降：{'是' if response['monotonic_decreasing'] else '否'}",
        "",
        "| 时间间隔 | 候选样本数 | 观测数 | 平均预测概率 | 中位数 | 标准差 | 相对基线变化 | 重复间最大差异 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in controlled["per_interval"]:
        lines.append(
            f"| {entry['interval_label']} | {entry['candidate_count']} | "
            f"{entry['observation_count']} | {entry['mean_probability']:.4f} | "
            f"{entry['median_probability']:.4f} | {entry['std_probability']:.4f} | "
            f"{entry['change_vs_baseline']:+.4f} | {entry['max_repeat_spread']:.3g} |"
        )
    lines.append("")
    for caveat in controlled["caveats"]:
        lines.append(f"> {caveat}")
        lines.append("")
    return "\n".join(lines)
