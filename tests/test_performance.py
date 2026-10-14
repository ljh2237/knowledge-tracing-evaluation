"""推理性能测试：统计口径、计时口径与异常接口。"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from contracts import (
    DataValidationError,
    ModelValidationError,
    Prediction,
    PredictionRequest,
    Target,
)
from evaluation.common import EvaluationError
from evaluation.performance_test import (
    BatchObservation,
    environment_info,
    measure_batches,
    parse_batch_sizes,
    run_performance,
    summarize_batches,
)
from mock_model import MockModel
from model_adapter import ModelAdapter


def write_input(path, events=16, students=2):
    rows = []
    for student in range(students):
        for index in range(events // students):
            rows.append({
                "record_id": f"s{student}-{index}", "student_id": f"S{student}",
                "question_id": f"Q{index}", "skill_id": "K01", "correct": index % 2,
                "timestamp": f"2026-10-01T{index:02d}:00:00+00:00",
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_requests(count):
    targets = [
        Target(f"r{index}", "S1", "Q1", ("K01",), (1,), None, index, "order")
        for index in range(count)
    ]
    return [PredictionRequest(target, (), None) for target in targets]


class CountingModel:
    """记录调用次数的确定性模型，用于核对预热与重复轮数。"""

    name = "counting_view"
    is_mock = True

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def predict(self, requests):
        self.calls += 1
        self.batch_sizes.append(len(requests))
        return [Prediction(request.target.record_id, 0.5) for request in requests]


# --------------------------------------------------------------------- 参数解析


def test_parse_batch_sizes_deduplicates_and_sorts():
    assert parse_batch_sizes("64,1,8") == (1, 8, 64)


@pytest.mark.parametrize("text,message", [
    ("", "不能为空"), ("1,,2", "空项"), ("a", "非整数"), ("0", "正整数"), ("-4", "正整数"),
    ("1,1", "重复"),
])
def test_parse_batch_sizes_rejects_bad_input(text, message):
    with pytest.raises(EvaluationError, match=message):
        parse_batch_sizes(text)


def test_environment_info_records_runtime_and_timer():
    environment = environment_info()
    for key in ("python", "platform", "cpu_count", "pandas", "numpy", "scikit_learn", "timer"):
        assert key in environment
    assert environment["timer"] == "time.perf_counter"


# ----------------------------------------------------------------------- 计时


def test_measure_batches_counts_warmup_and_repeats():
    model = CountingModel()
    adapter = ModelAdapter(model)
    observations = measure_batches(
        adapter, make_requests(8), batch_size=4, repeats=3, warmup=2
    )
    # 2 个批次 × (2 次预热 + 3 轮重复) = 10 次调用。
    assert model.calls == 10
    assert len(observations) == 6
    assert {item.sample_count for item in observations} == {4}
    assert {item.repetition for item in observations} == {1, 2, 3}


def test_measure_batches_drops_incomplete_tail_batch():
    model = CountingModel()
    observations = measure_batches(
        ModelAdapter(model), make_requests(7), batch_size=4, repeats=1, warmup=0
    )
    # 7 条请求只能组成 1 个完整批次，剩余 3 条被排除以保证单样本延迟可比。
    assert len(observations) == 1
    assert observations[0].sample_count == 4


def test_measure_batches_rejects_pool_smaller_than_batch():
    with pytest.raises(EvaluationError, match="少于一个完整批次"):
        measure_batches(ModelAdapter(CountingModel()), make_requests(3), batch_size=8,
                        repeats=1, warmup=0)


@pytest.mark.parametrize("kwargs,message", [
    ({"batch_size": 0, "repeats": 1, "warmup": 0}, "batch_size"),
    ({"batch_size": 1, "repeats": 0, "warmup": 0}, "repeats"),
    ({"batch_size": 1, "repeats": 1, "warmup": -1}, "warmup"),
])
def test_measure_batches_validates_arguments(kwargs, message):
    with pytest.raises(EvaluationError, match=message):
        measure_batches(ModelAdapter(CountingModel()), make_requests(4), **kwargs)


def test_measure_batches_propagates_model_failure():
    class BrokenModel:
        name = "broken"
        is_mock = True

        def predict(self, requests):
            raise RuntimeError("推理崩溃")

    # 模型异常必须上抛为 ModelValidationError，而不是被吞掉当作成功。
    with pytest.raises(ModelValidationError, match="推理崩溃"):
        measure_batches(ModelAdapter(BrokenModel()), make_requests(4), batch_size=2,
                        repeats=1, warmup=0)


# ----------------------------------------------------------------------- 统计


def test_summarize_batches_computes_latency_and_throughput():
    observations = [BatchObservation(2, 2, index, 0.5) for index in (1, 2, 3)]
    row = summarize_batches(observations, model_name="m", is_mock=True, warmup=0, pool_size=6)
    assert row.mean_batch_latency_ms == pytest.approx(500.0)
    assert row.p50_batch_latency_ms == pytest.approx(500.0)
    assert row.p95_batch_latency_ms == pytest.approx(500.0)
    assert row.mean_per_sample_latency_ms == pytest.approx(250.0)
    # 3 批 × 0.5 秒 = 1.5 秒，共 6 个样本 → 4 样本/秒。
    assert row.throughput_samples_per_second == pytest.approx(4.0)
    assert row.batch_count == 3
    assert row.sample_count == 6
    assert row.pool_size == 6


def test_summarize_batches_percentiles_track_the_distribution():
    seconds = [0.1, 0.2, 0.3, 0.4, 1.0]
    observations = [
        BatchObservation(1, 1, index + 1, value) for index, value in enumerate(seconds)
    ]
    row = summarize_batches(observations, model_name="m", is_mock=True, warmup=0, pool_size=5)
    assert row.p50_batch_latency_ms == pytest.approx(300.0)
    assert row.p95_batch_latency_ms == pytest.approx(880.0)
    assert row.mean_batch_latency_ms == pytest.approx(400.0)
    assert row.p50_batch_latency_ms <= row.mean_batch_latency_ms <= row.p95_batch_latency_ms


def test_summarize_batches_rejects_mixed_batch_sizes_and_empty_input():
    with pytest.raises(EvaluationError, match="没有计时观测"):
        summarize_batches([], model_name="m", is_mock=True, warmup=0, pool_size=0)
    mixed = [BatchObservation(1, 1, 1, 0.1), BatchObservation(2, 2, 2, 0.1)]
    with pytest.raises(EvaluationError, match="batch_size 必须一致"):
        summarize_batches(mixed, model_name="m", is_mock=True, warmup=0, pool_size=3)


def test_summarize_batches_rejects_negative_timing():
    bad = [BatchObservation(1, 1, 1, -0.1)]
    with pytest.raises(EvaluationError, match="非负有限数"):
        summarize_batches(bad, model_name="m", is_mock=True, warmup=0, pool_size=1)


# ------------------------------------------------------------------- 编排入口


def test_run_performance_writes_csv_and_json(tmp_path):
    path = write_input(tmp_path / "input.csv", events=16)
    payload = run_performance(
        path, ModelAdapter(MockModel()), tmp_path / "out",
        batch_sizes=(1, 4), repeats=2, warmup=1, max_requests=16, overwrite=True,
    )
    out = tmp_path / "out" / "performance"
    assert (out / "performance.csv").is_file()
    stored = json.loads((out / "performance.json").read_text(encoding="utf-8"))
    assert [row["batch_size"] for row in stored["rows"]] == [1, 4]
    assert stored["settings"]["pool_size"] == 16
    assert stored["model"]["is_mock"] is True
    assert any("run_summary.json" in caveat for caveat in stored["caveats"])
    for row in payload["rows"]:
        assert row["p50_batch_latency_ms"] <= row["p95_batch_latency_ms"]
        assert row["throughput_samples_per_second"] > 0


def test_run_performance_uses_one_pool_for_all_batch_sizes(tmp_path):
    path = write_input(tmp_path / "input.csv", events=16)
    payload = run_performance(
        path, ModelAdapter(MockModel()), tmp_path / "out",
        batch_sizes=(1, 2, 4), repeats=1, warmup=0, max_requests=8, overwrite=True,
    )
    assert {row["pool_size"] for row in payload["rows"]} == {8}


def test_run_performance_rejects_pool_smaller_than_largest_batch(tmp_path):
    path = write_input(tmp_path / "input.csv", events=8)
    with pytest.raises(EvaluationError, match="小于最大批量"):
        run_performance(path, ModelAdapter(MockModel()), tmp_path / "out",
                        batch_sizes=(64,), max_requests=8)


def test_run_performance_validates_sizes_and_limits(tmp_path):
    path = write_input(tmp_path / "input.csv", events=8)
    with pytest.raises(EvaluationError, match="正整数序列"):
        run_performance(path, ModelAdapter(MockModel()), tmp_path / "out", batch_sizes=(0,))
    with pytest.raises(EvaluationError, match="max_requests"):
        run_performance(path, ModelAdapter(MockModel()), tmp_path / "out", max_requests=0)


def test_run_performance_fails_on_missing_input(tmp_path):
    # 输入文件不存在时必须明确失败，且不留下任何输出目录。
    with pytest.raises(DataValidationError, match="无法读取 CSV"):
        run_performance(tmp_path / "missing.csv", ModelAdapter(MockModel()), tmp_path / "out")
    assert not (tmp_path / "out").exists()
