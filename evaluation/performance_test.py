"""模块四：模型推理性能测试。

复用负责人 A 的 ``ModelAdapter`` 与 ``PredictionRequest``，因此 Mock 模型与将来的
真实模型走同一条测量路径。计时口径：

- **批次延迟**：单次 ``adapter.predict(batch)`` 的墙钟耗时，由 ``time.perf_counter()``
  测量。为避免最后一批样本数不同影响可比性，每个 batch size 只使用能整除的完整批次。
- **单样本延迟**：批次延迟 ÷ 该批次样本数。
- **P50 / P95**：对全部被计时的批次延迟序列取百分位（``numpy.percentile`` 线性插值）。
- **吞吐量**：被计时的总样本数 ÷ 被计时的总耗时，单位「样本/秒」。

**不得混淆的边界**：``run_summary.json`` 中的 ``model_seconds`` 是整条流水线的累计
耗时，不是模型性能；开发电脑上的数字也不能当作鸿蒙端侧性能。真实性能测试需要训练组
的模型权重与目标运行环境。
"""
from __future__ import annotations

import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn

from contracts import PredictionRequest
from data_loader import SkillVocabulary, load_data
from evaluation.common import (
    EVALUATION_VERSION,
    EvaluationError,
    ensure_output_dir,
    utc_now_iso,
    write_document,
    write_table,
)
from model_adapter import ModelAdapter
from pipeline import iter_requests

#: 默认比较的批量大小。
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 8, 32, 64)

PERFORMANCE_COLUMNS = (
    "batch_size", "batch_count", "sample_count", "repeats", "warmup_batches",
    "mean_batch_latency_ms", "p50_batch_latency_ms", "p95_batch_latency_ms",
    "mean_per_sample_latency_ms", "p50_per_sample_latency_ms", "p95_per_sample_latency_ms",
    "throughput_samples_per_second", "total_seconds", "pool_size",
    "model_name", "is_mock",
)

TIMING_CAVEAT = (
    "计时口径：批次延迟为单次 adapter.predict(batch) 的墙钟耗时（time.perf_counter）；"
    "单样本延迟为批次延迟除以该批次样本数；P50/P95 对批次延迟序列取线性插值百分位；"
    "吞吐量为被计时样本数除以被计时的总耗时。"
)

SCOPE_CAVEAT = (
    "性能数字只代表本次运行所在机器的本次测量：它既不等同于 run_summary.json 中的流水线"
    "累计耗时 (model_seconds)，也不等同于鸿蒙端侧性能。模拟模型（is_mock=True）的性能"
    "不能代表真实 DKT/AKT 的推理开销。"
)


@dataclass(frozen=True)
class BatchObservation:
    """一次被计时的批次调用。"""

    batch_size: int
    sample_count: int
    repetition: int
    seconds: float


@dataclass(frozen=True)
class PerformanceRow:
    batch_size: int
    batch_count: int
    sample_count: int
    repeats: int
    warmup_batches: int
    mean_batch_latency_ms: float
    p50_batch_latency_ms: float
    p95_batch_latency_ms: float
    mean_per_sample_latency_ms: float
    p50_per_sample_latency_ms: float
    p95_per_sample_latency_ms: float
    throughput_samples_per_second: float
    total_seconds: float
    pool_size: int
    model_name: str
    is_mock: bool | None

    def as_record(self) -> dict:
        return asdict(self)


def parse_batch_sizes(text: str) -> tuple[int, ...]:
    """解析 ``--batch-sizes 1,8,32,64``，返回升序去重的正整数元组。"""
    if not isinstance(text, str) or not text.strip():
        raise EvaluationError("--batch-sizes 不能为空，例如 --batch-sizes 1,8,32,64")
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            raise EvaluationError(f"--batch-sizes 含空项: {text!r}")
        try:
            value = int(chunk)
        except ValueError as exc:
            raise EvaluationError(f"--batch-sizes 含非整数项 {chunk!r}") from exc
        if value < 1:
            raise EvaluationError(f"批量大小必须是正整数，实际为 {chunk!r}")
        values.append(value)
    if len(set(values)) != len(values):
        raise EvaluationError(f"--batch-sizes 存在重复项: {text!r}")
    return tuple(sorted(values))


def environment_info() -> dict:
    """记录测量环境，便于判断数字是否可迁移。"""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "timer": "time.perf_counter",
    }


def measure_batches(
    adapter: ModelAdapter,
    requests: Sequence[PredictionRequest],
    *,
    batch_size: int,
    repeats: int,
    warmup: int,
) -> list[BatchObservation]:
    """对固定请求池重复计时；只使用完整批次以保证单样本延迟可比。"""
    if batch_size < 1:
        raise EvaluationError(f"batch_size 必须是正整数: {batch_size}")
    if repeats < 1:
        raise EvaluationError(f"repeats 必须是正整数: {repeats}")
    if warmup < 0:
        raise EvaluationError(f"warmup 不能为负数: {warmup}")
    usable = (len(requests) // batch_size) * batch_size
    if usable == 0:
        raise EvaluationError(
            f"请求数量 {len(requests)} 少于一个完整批次（batch_size={batch_size}），无法测量"
        )
    pool = list(requests[:usable])
    batches = [pool[start:start + batch_size] for start in range(0, usable, batch_size)]

    for _ in range(warmup):
        for batch in batches:
            adapter.predict(batch)

    observations: list[BatchObservation] = []
    for repetition in range(1, repeats + 1):
        for batch in batches:
            start = time.perf_counter()
            adapter.predict(batch)
            elapsed = time.perf_counter() - start
            observations.append(BatchObservation(
                batch_size=batch_size, sample_count=len(batch),
                repetition=repetition, seconds=elapsed,
            ))
    return observations


def summarize_batches(
    observations: Sequence[BatchObservation], *, model_name: str, is_mock: bool | None,
    warmup: int, pool_size: int,
) -> PerformanceRow:
    """把一个 batch size 的计时观测汇总为一行统计结果。"""
    if not observations:
        raise EvaluationError("没有计时观测，无法统计性能")
    batch_seconds = np.array([item.seconds for item in observations], dtype=float)
    batch_sizes = {item.batch_size for item in observations}
    if len(batch_sizes) != 1:
        raise EvaluationError("同一批观测的 batch_size 必须一致")
    batch_size = batch_sizes.pop()
    if not np.all(np.isfinite(batch_seconds)) or np.any(batch_seconds < 0):
        raise EvaluationError("计时结果必须是非负有限数")

    per_sample_seconds = np.array(
        [item.seconds / item.sample_count for item in observations], dtype=float
    )
    total_seconds = float(batch_seconds.sum())
    total_samples = int(sum(item.sample_count for item in observations))
    return PerformanceRow(
        batch_size=batch_size,
        batch_count=len(observations),
        sample_count=total_samples,
        repeats=max(item.repetition for item in observations),
        warmup_batches=warmup,
        mean_batch_latency_ms=float(batch_seconds.mean() * 1000.0),
        p50_batch_latency_ms=float(np.percentile(batch_seconds, 50) * 1000.0),
        p95_batch_latency_ms=float(np.percentile(batch_seconds, 95) * 1000.0),
        mean_per_sample_latency_ms=float(per_sample_seconds.mean() * 1000.0),
        p50_per_sample_latency_ms=float(np.percentile(per_sample_seconds, 50) * 1000.0),
        p95_per_sample_latency_ms=float(np.percentile(per_sample_seconds, 95) * 1000.0),
        throughput_samples_per_second=(total_samples / total_seconds) if total_seconds > 0 else math.inf,
        total_seconds=total_seconds,
        pool_size=pool_size,
        model_name=model_name,
        is_mock=is_mock,
    )


def run_performance(
    input_path: str | Path,
    adapter: ModelAdapter,
    output_dir: str | Path,
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    repeats: int = 3,
    warmup: int = 1,
    max_requests: int | None = 1024,
    max_history: int = 256,
    invalid_policy: str = "error",
    timestamp_unit: str = "iso",
    vocabulary: SkillVocabulary | None = None,
    unknown_skill_policy: str = "error",
    overwrite: bool = False,
) -> dict:
    """对给定输入与模型执行多批量大小的推理性能测试并写出结果。"""
    sizes = tuple(int(size) for size in batch_sizes)
    if not sizes or any(size < 1 for size in sizes):
        raise EvaluationError("batch_sizes 必须是非空的正整数序列")
    if max_requests is not None and max_requests < 1:
        raise EvaluationError("max_requests 必须为正整数或不限制")

    loaded = load_data(
        input_path, invalid_policy=invalid_policy, timestamp_unit=timestamp_unit,
        vocabulary=vocabulary, unknown_skill_policy=unknown_skill_policy,
    )
    pool = list(iter_requests(loaded.records, max_history))
    if max_requests is not None:
        # 所有 batch size 共用同一请求池，保证不同批量之间的数字可比。
        pool = pool[:max_requests]
    if not pool:
        raise EvaluationError("没有可用的预测请求，无法进行性能测试")
    if len(pool) < max(sizes):
        raise EvaluationError(
            f"请求数量 {len(pool)} 小于最大批量 {max(sizes)}，无法构成完整批次；"
            "请增大 --max-requests 或减小 --batch-sizes"
        )

    rows: list[PerformanceRow] = []
    for size in sizes:
        observations = measure_batches(
            adapter, pool, batch_size=size, repeats=repeats, warmup=warmup
        )
        rows.append(summarize_batches(
            observations, model_name=adapter.model.name, is_mock=adapter.model.is_mock,
            warmup=warmup, pool_size=len(pool),
        ))

    payload = {
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": utc_now_iso(),
        "environment": environment_info(),
        "settings": {
            "batch_sizes": list(sizes), "repeats": repeats, "warmup_batches": warmup,
            "max_requests": max_requests, "pool_size": len(pool), "max_history": max_history,
            "invalid_policy": invalid_policy, "timestamp_unit": timestamp_unit,
            "unknown_skill_policy": unknown_skill_policy,
        },
        "model": {
            "model_name": adapter.model.name,
            "is_mock": adapter.model.is_mock,
            "data_nature": (
                "模拟模型输出（仅用于程序功能验证）" if adapter.model.is_mock else "外部模型输出"
            ),
        },
        "input": {
            "file": Path(input_path).name,
            "accepted_rows": loaded.summary["accepted_rows"],
            "students": loaded.summary["students"],
            "time_source": loaded.summary["time_source"],
        },
        "rows": [row.as_record() for row in rows],
        "caveats": [TIMING_CAVEAT, SCOPE_CAVEAT],
    }
    performance_dir = ensure_output_dir(
        Path(output_dir).expanduser().resolve() / "performance", overwrite=overwrite
    )
    write_table(
        performance_dir / "performance.csv", [row.as_record() for row in rows], PERFORMANCE_COLUMNS
    )
    write_document(performance_dir / "performance.json", payload)
    return payload


def build_performance_markdown(payload: dict) -> str:
    """把性能测试结果渲染为 Markdown 片段，供报告生成器复用。"""
    model = payload["model"]
    settings = payload["settings"]
    environment = payload["environment"]
    lines = [
        f"- 测试模型：`{model['model_name']}`（is_mock={model['is_mock']}，{model['data_nature']}）",
        f"- 请求池：`{payload['input']['file']}` 的 {settings['pool_size']} 条请求；"
        f"每个批量重复 {settings['repeats']} 轮，预热 {settings['warmup_batches']} 个批次",
        f"- 运行环境：{environment['python_implementation']} {environment['python']}，"
        f"{environment['platform']}，CPU 逻辑核 {environment['cpu_count']}",
        f"- 计时方式：`{environment['timer']}`",
        "",
        "| 批量大小 | 批次数 | 样本数 | 平均批次延迟(ms) | P50(ms) | P95(ms) | "
        "单样本平均(ms) | 吞吐量(样本/秒) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['batch_size']} | {row['batch_count']} | {row['sample_count']} | "
            f"{row['mean_batch_latency_ms']:.3f} | {row['p50_batch_latency_ms']:.3f} | "
            f"{row['p95_batch_latency_ms']:.3f} | {row['mean_per_sample_latency_ms']:.3f} | "
            f"{row['throughput_samples_per_second']:.1f} |"
        )
    lines.append("")
    for caveat in payload["caveats"]:
        lines.append(f"> {caveat}")
        lines.append("")
    return "\n".join(lines)
