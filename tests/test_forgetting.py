"""遗忘机制专项测试：描述性分析、受控固定历史测试及边界情况。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.common import EvaluationError, load_predictions
from evaluation.forgetting_test import (
    DEFAULT_INTERVAL_DAYS,
    SECONDS_PER_DAY,
    controlled_time_response,
    describe_time_response,
    format_duration,
    parse_intervals,
    run_forgetting,
    select_candidates,
)
from evaluation.synthetic_time_model import SyntheticTimeSensitiveModel, create_model
from mock_model import MockModel
from model_adapter import ModelAdapter
from pipeline import iter_requests
from data_loader import load_data

ROOT = Path(__file__).resolve().parents[1]


def write_predictions(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def row(record_id, delta, y_pred, *, time_source="timestamp", cold_start="False"):
    delta_cell = "" if delta is None else delta
    return {
        "record_id": record_id, "student_id": "S1", "question_id": "Q1", "skill_id": "K01",
        "time_source": time_source, "y_true": 1, "y_pred": y_pred, "predicted_label": 1,
        "history_length": 1, "delta_time_seconds": delta_cell, "cold_start": cold_start,
        "model_name": "unit_model", "is_mock": "True",
    }


def write_input(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------- 参数解析


def test_parse_intervals_converts_units_and_sorts():
    assert parse_intervals("0,1,3", "days") == (0.0, SECONDS_PER_DAY, 3 * SECONDS_PER_DAY)
    assert parse_intervals("0,30", "minutes") == (0.0, 1800.0)
    assert parse_intervals("7,1,3", "hours") == (3600.0, 3 * 3600.0, 7 * 3600.0)


@pytest.mark.parametrize("text,unit,message", [
    ("", "days", "不能为空"),
    ("1,,2", "days", "空项"),
    ("a,1", "days", "非数值"),
    ("-1,0", "days", "非负"),
    ("1,1", "days", "重复"),
    ("1,2", "weeks", "时间单位"),
])
def test_parse_intervals_rejects_bad_input(text, unit, message):
    with pytest.raises(EvaluationError, match=message):
        parse_intervals(text, unit)


def test_format_duration_uses_readable_units():
    assert format_duration(0) == "0秒"
    assert format_duration(90) == "1.5分"
    assert format_duration(7200) == "2小时"
    assert format_duration(SECONDS_PER_DAY) == "1天"


# --------------------------------------------------------------- A 描述性分析


def test_describe_groups_by_interval(tmp_path):
    rows = [
        row("a", 10.0, 0.9), row("b", 30.0, 0.7),
        row("c", 3600.0, 0.5), row("d", 7200.0, 0.3),
        row("e", 10 * SECONDS_PER_DAY, 0.1),
    ]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    result = describe_time_response(data)
    assert result["available"] is True
    assert result["usable_samples"] == 5
    assert result["excluded_samples"] == 0
    buckets = {item["bucket"]: item for item in result["bucket_rows"]}
    assert buckets["[0秒, 1分)"]["sample_count"] == 2
    assert buckets["[0秒, 1分)"]["mean_y_pred"] == pytest.approx(0.8)
    assert "[1小时, 6小时)" in buckets
    assert "观察性分析" in result["caveat"]


def test_describe_excludes_cold_start_rows_without_delta(tmp_path):
    rows = [row("a", 10.0, 0.9), row("b", None, 0.5, cold_start="True")]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    result = describe_time_response(data)
    assert result["usable_samples"] == 1
    assert result["excluded_samples"] == 1
    assert "冷启动" in result["excluded_reason"]


def test_describe_is_unavailable_for_order_only_data(tmp_path):
    rows = [row(letter, None, 0.5, time_source="order") for letter in "abc"]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    result = describe_time_response(data)
    assert result["available"] is False
    assert "event_order" in result["reason"]
    assert "伪造" in result["reason"]


def test_describe_is_unavailable_when_column_missing(tmp_path):
    rows = [{k: v for k, v in row("a", 10.0, 0.5).items() if k != "delta_time_seconds"}]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    result = describe_time_response(data)
    assert result["available"] is False
    assert "delta_time_seconds" in result["reason"]


def test_describe_rejects_non_increasing_bucket_edges(tmp_path):
    rows = [row("a", 10.0, 0.5)]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    with pytest.raises(EvaluationError, match="分桶边界"):
        describe_time_response(data, bucket_edges=(0.0, 60.0, 30.0))


def test_correlation_is_none_when_probability_is_constant(tmp_path):
    rows = [row(letter, 10.0 * index, 0.5) for index, letter in enumerate("abcd")]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    result = describe_time_response(data)
    assert result["correlations"]["spearman_delta_vs_y_pred"] is None


# ------------------------------------------------------------- B 受控时间测试


@pytest.fixture
def controlled_input(tmp_path):
    """两位学生、同一知识点、带真实时间戳，可用于构造固定历史受控请求。"""
    rows = [
        {"record_id": "s1-1", "student_id": "S1", "question_id": "Q1", "skill_id": "K01",
         "correct": 1, "timestamp": "2026-10-01T00:00:00+00:00"},
        {"record_id": "s1-2", "student_id": "S1", "question_id": "Q2", "skill_id": "K01",
         "correct": 0, "timestamp": "2026-10-01T01:00:00+00:00"},
        {"record_id": "s1-3", "student_id": "S1", "question_id": "Q3", "skill_id": "K01",
         "correct": 1, "timestamp": "2026-10-01T02:00:00+00:00"},
        {"record_id": "s2-1", "student_id": "S2", "question_id": "Q1", "skill_id": "K01",
         "correct": 0, "timestamp": "2026-10-02T00:00:00+00:00"},
        {"record_id": "s2-2", "student_id": "S2", "question_id": "Q2", "skill_id": "K01",
         "correct": 1, "timestamp": "2026-10-02T03:00:00+00:00"},
    ]
    return write_input(tmp_path / "input.csv", rows)


def candidates_for(path):
    return select_candidates(list(iter_requests(load_data(path).records)), max_requests=None, seed=0)


def test_candidates_require_history_and_real_time(controlled_input):
    candidates = candidates_for(controlled_input)
    assert {request.target.record_id for request in candidates} == {"s1-2", "s1-3", "s2-2"}
    assert all(request.history for request in candidates)
    assert all(request.delta_time_seconds is not None for request in candidates)


def test_candidate_selection_is_reproducible(controlled_input):
    """同一种子必须复现同一批候选样本，且抽样结果是全集的子集。"""
    requests = list(iter_requests(load_data(controlled_input).records))
    everything = select_candidates(requests, max_requests=None, seed=0)
    first = select_candidates(requests, max_requests=2, seed=11)
    second = select_candidates(requests, max_requests=2, seed=11)
    assert [r.target.record_id for r in first] == [r.target.record_id for r in second]
    assert len(first) == 2
    assert set(r.target.record_id for r in first) <= set(r.target.record_id for r in everything)
    # 不限制数量时不做抽样。
    assert len(everything) == 3


def test_mock_model_shows_no_time_response(controlled_input):
    """MockModel 没有遗忘机制，概率不随时间变化是合理结果，不得强制制造下降曲线。"""
    result = controlled_time_response(
        controlled_input, ModelAdapter(MockModel()),
        intervals_seconds=(0.0, SECONDS_PER_DAY, 14 * SECONDS_PER_DAY), repeats=2,
    )
    assert result["available"] is True
    assert result["response"]["detected"] is False
    assert result["response"]["max_abs_change_vs_baseline"] == pytest.approx(0.0)
    assert result["response"]["monotonic_decreasing"] is False
    assert all(item["max_repeat_spread"] == pytest.approx(0.0) for item in result["per_interval"])
    assert any("没有遗忘机制" in caveat for caveat in result["caveats"])


def test_synthetic_fixture_shows_monotonic_decreasing_response(controlled_input):
    """明确标注的合成夹具必须能被测试程序识别出时间响应，证明测试本身有效。"""
    result = controlled_time_response(
        controlled_input, ModelAdapter(SyntheticTimeSensitiveModel()),
        intervals_seconds=(0.0, SECONDS_PER_DAY, 7 * SECONDS_PER_DAY, 14 * SECONDS_PER_DAY),
        repeats=2,
    )
    assert result["response"]["detected"] is True
    assert result["response"]["monotonic_decreasing"] is True
    means = [item["mean_probability"] for item in result["per_interval"]]
    assert means == sorted(means, reverse=True)
    assert result["model"]["is_mock"] is True
    assert result["model"]["model_name"] == "synthetic_time_sensitive_fixture_v1"


def test_controlled_test_fixes_history_and_preserves_causal_order(controlled_input):
    result = controlled_time_response(
        controlled_input, ModelAdapter(MockModel()),
        intervals_seconds=(0.0, SECONDS_PER_DAY), repeats=2,
    )
    assert result["design"]["history_fixed"] is True
    assert result["design"]["causal_order_preserved"] is True
    frame = pd.DataFrame(result["detail_rows"])
    # 同一候选样本在不同时间间隔下的历史长度必须完全一致。
    grouped = frame.groupby("candidate_record_id")["history_length"].nunique()
    assert (grouped == 1).all()
    # 每个（候选, 间隔, 重复）组合恰好一次调用。
    assert len(frame) == result["design"]["total_inference_calls"]
    assert frame["interval_seconds"].nunique() == 2


def test_controlled_zero_interval_is_accepted(controlled_input):
    result = controlled_time_response(
        controlled_input, ModelAdapter(MockModel()), intervals_seconds=(0.0,), repeats=1,
    )
    assert result["available"] is True
    assert result["baseline_interval_seconds"] == 0.0
    assert result["per_interval"][0]["interval_seconds"] == 0.0


def test_controlled_rejects_negative_interval(controlled_input):
    with pytest.raises(EvaluationError, match="非负有限数"):
        controlled_time_response(
            controlled_input, ModelAdapter(MockModel()), intervals_seconds=(-1.0, 0.0)
        )


def test_controlled_rejects_invalid_repeats_and_batch_size(controlled_input):
    with pytest.raises(EvaluationError, match="repeats"):
        controlled_time_response(controlled_input, ModelAdapter(MockModel()), repeats=0)
    with pytest.raises(EvaluationError, match="batch_size"):
        controlled_time_response(controlled_input, ModelAdapter(MockModel()), batch_size=0)


def test_controlled_skips_order_only_input(tmp_path):
    rows = [
        {"record_id": "a", "student_id": "S1", "question_id": "Q1", "skill_id": "K01",
         "correct": 1, "event_order": 1},
        {"record_id": "b", "student_id": "S1", "question_id": "Q2", "skill_id": "K01",
         "correct": 0, "event_order": 2},
    ]
    result = controlled_time_response(
        write_input(tmp_path / "order.csv", rows), ModelAdapter(MockModel())
    )
    assert result["available"] is False
    assert "event_order" in result["reason"]


def test_controlled_skips_cold_start_only_input(tmp_path):
    rows = [
        {"record_id": f"r{index}", "student_id": f"S{index}", "question_id": "Q1", "skill_id": "K01",
         "correct": 1, "timestamp": "2026-10-01T00:00:00+00:00"}
        for index in range(3)
    ]
    result = controlled_time_response(
        write_input(tmp_path / "cold.csv", rows), ModelAdapter(MockModel())
    )
    assert result["available"] is False
    assert "非空可见历史" in result["reason"]


def test_synthetic_factory_reads_decay_config():
    model = create_model({"decay_per_day": 1.0, "floor": 0.0})
    assert model.decay_per_day == 1.0
    with pytest.raises(ValueError, match="decay_per_day"):
        create_model({"decay_per_day": 0})


# ------------------------------------------------------------------- 编排入口


def test_run_forgetting_reports_both_parts(tmp_path, controlled_input):
    predictions = write_predictions(tmp_path / "p.csv", [
        row("a", 10.0, 0.9), row("b", 100.0, 0.8), row("c", 1000.0, 0.7),
    ])
    payload = run_forgetting(
        tmp_path / "out", predictions=predictions, input_path=controlled_input,
        controlled_models=[("mock", ModelAdapter(MockModel()))],
        intervals_seconds=(0.0, SECONDS_PER_DAY), repeats=1, overwrite=True,
    )
    out = tmp_path / "out" / "forgetting"
    assert (out / "interval_buckets.csv").is_file()
    assert (out / "controlled_time_response.csv").is_file()
    stored = json.loads((out / "forgetting_summary.json").read_text(encoding="utf-8"))
    assert stored["descriptive"]["available"] is True
    assert stored["controlled_models"][0]["available"] is True
    assert "detail_rows" not in stored["controlled_models"][0]
    assert payload["descriptive"]["usable_samples"] == 3


def test_run_forgetting_skips_parts_without_input(tmp_path):
    payload = run_forgetting(tmp_path / "out", overwrite=True)
    assert payload["descriptive"]["available"] is False
    assert payload["controlled_models"][0]["available"] is False
    assert "未提供预测文件" in payload["descriptive"]["reason"]
    assert "未提供原始输入" in payload["controlled_models"][0]["reason"]
    assert not (tmp_path / "out" / "forgetting" / "interval_buckets.csv").exists()


def test_run_forgetting_uses_default_day_intervals(tmp_path, controlled_input):
    payload = run_forgetting(
        tmp_path / "out", input_path=controlled_input,
        controlled_models=[("mock", ModelAdapter(MockModel()))], repeats=1, overwrite=True,
    )
    intervals = payload["controlled_models"][0]["design"]["intervals_seconds"]
    assert intervals == [value * SECONDS_PER_DAY for value in DEFAULT_INTERVAL_DAYS]
