"""模型评价指标模块测试：含可人工核对的标准数值用例。"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from evaluation.common import EvaluationError, load_predictions
from evaluation.evaluator import (
    AUC_COMPUTED,
    AUC_SINGLE_CLASS,
    GROUP_ALL,
    GROUP_COLD,
    GROUP_NON_COLD,
    compute_core_metrics,
    evaluate_frame,
    run_metrics,
)

# 人工可核对的标准数据：AUC = 3/4 = 0.75，Accuracy = 3/4 = 0.75，
# LogLoss = -(ln0.9 + ln0.6 + ln0.35 + ln0.8)/4 ≈ 0.4722879538，Brier = 0.158125。
REFERENCE_LABELS = [0, 0, 1, 1]
REFERENCE_PROBABILITIES = [0.1, 0.4, 0.35, 0.8]


def write_predictions(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def prediction_row(record_id, y_true, y_pred, **extra):
    try:
        label = int(float(y_pred) >= 0.5)
    except (TypeError, ValueError):
        label = ""  # 非法概率原样保留，用于验证加载器会拒绝而不是纠正它
    row = {
        "record_id": record_id, "student_id": "S1", "question_id": "Q1", "skill_id": "K01",
        "time_source": "timestamp", "y_true": y_true, "y_pred": y_pred,
        "predicted_label": label, "history_length": 1,
        "cold_start": "False", "model_name": "unit_model", "is_mock": "True",
    }
    row.update(extra)
    return row


def test_reference_metrics_match_hand_computation():
    metrics = compute_core_metrics(REFERENCE_LABELS, REFERENCE_PROBABILITIES, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["log_loss"] == pytest.approx(0.47228795380917615, rel=1e-12)
    assert metrics["brier_score"] == pytest.approx(0.158125)
    assert metrics["auc_status"] == AUC_COMPUTED
    assert metrics["sample_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2


def test_threshold_changes_accuracy_but_not_auc_or_log_loss():
    strict = compute_core_metrics(REFERENCE_LABELS, REFERENCE_PROBABILITIES, threshold=0.9)
    lenient = compute_core_metrics(REFERENCE_LABELS, REFERENCE_PROBABILITIES, threshold=0.2)
    # threshold=0.9 全部判负 → 2/4；threshold=0.2 只有 0.1 判负 → 3/4。
    assert strict["accuracy"] == pytest.approx(0.5)
    assert lenient["accuracy"] == pytest.approx(0.75)
    assert strict["roc_auc"] == lenient["roc_auc"] == pytest.approx(0.75)
    assert strict["log_loss"] == pytest.approx(lenient["log_loss"])


def test_single_class_auc_is_marked_undefined_not_invented():
    metrics = compute_core_metrics([1, 1, 1], [0.2, 0.6, 0.9])
    assert metrics["roc_auc"] is None
    assert metrics["auc_status"] == AUC_SINGLE_CLASS
    assert "ROC-AUC 无定义" in metrics["auc_reason"]
    # 其余指标仍然可算，不应因为 AUC 无定义而整体失败：0.2 判负，其余两个判正 → 2/3。
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["log_loss"] > 0


def test_perfect_and_constant_predictions():
    perfect = compute_core_metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["accuracy"] == pytest.approx(1.0)
    assert perfect["log_loss"] < 1e-10
    constant = compute_core_metrics([0, 0, 1, 1], [0.6, 0.6, 0.6, 0.6])
    assert constant["roc_auc"] == pytest.approx(0.5)


def test_compute_core_metrics_rejects_invalid_input():
    with pytest.raises(EvaluationError, match="长度|不一致"):
        compute_core_metrics([0, 1], [0.5])
    with pytest.raises(EvaluationError, match="阈值"):
        compute_core_metrics([0, 1], [0.5, 0.5], threshold=1.5)
    with pytest.raises(EvaluationError, match="y_true"):
        compute_core_metrics([0, 2], [0.5, 0.5])
    with pytest.raises(EvaluationError, match="y_pred"):
        compute_core_metrics([0, 1], [0.5, math.nan])
    with pytest.raises(EvaluationError, match="y_pred"):
        compute_core_metrics([0, 1], [-0.1, 0.5])


def test_load_predictions_rejects_missing_file(tmp_path):
    with pytest.raises(EvaluationError, match="不存在"):
        load_predictions(tmp_path / "nope.csv")


def test_load_predictions_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("record_id,y_true,y_pred\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="没有数据行"):
        load_predictions(path)


def test_load_predictions_rejects_missing_required_column(tmp_path):
    path = write_predictions(tmp_path / "m.csv", [{"record_id": "a", "y_true": 1}])
    with pytest.raises(EvaluationError, match="缺少必需列"):
        load_predictions(path)


def test_load_predictions_rejects_duplicate_record_id(tmp_path):
    path = write_predictions(tmp_path / "d.csv", [
        prediction_row("dup", 1, 0.6), prediction_row("dup", 0, 0.4),
    ])
    with pytest.raises(EvaluationError, match="record_id 必须唯一"):
        load_predictions(path)


def test_load_predictions_rejects_illegal_label_and_probability(tmp_path):
    bad_label = write_predictions(tmp_path / "l.csv", [prediction_row("a", 2, 0.5)])
    with pytest.raises(EvaluationError, match="y_true 必须是 0 或 1"):
        load_predictions(bad_label)
    blank_label = write_predictions(tmp_path / "lb.csv", [prediction_row("a", "", 0.5)])
    with pytest.raises(EvaluationError, match="y_true 必须是 0 或 1"):
        load_predictions(blank_label)

    for bad in ("1.5", "-0.2", "nan", "inf", ""):
        path = write_predictions(tmp_path / f"p{abs(hash(bad))}.csv", [prediction_row("a", 1, bad)])
        with pytest.raises(EvaluationError, match="y_pred 必须是"):
            load_predictions(path)


def test_load_predictions_rejects_inconsistent_mock_flag_and_negative_delta(tmp_path):
    mixed = write_predictions(tmp_path / "mixed.csv", [
        prediction_row("a", 1, 0.5, is_mock="True"),
        prediction_row("b", 0, 0.5, is_mock="False"),
    ])
    with pytest.raises(EvaluationError, match="is_mock 在同一文件中不一致"):
        load_predictions(mixed)

    negative = write_predictions(tmp_path / "neg.csv", [
        prediction_row("a", 1, 0.5, delta_time_seconds=-1.0),
    ])
    with pytest.raises(EvaluationError, match="负的 delta_time_seconds"):
        load_predictions(negative)


def test_load_predictions_rejects_unknown_time_source(tmp_path):
    path = write_predictions(tmp_path / "ts.csv", [prediction_row("a", 1, 0.5, time_source="clock")])
    with pytest.raises(EvaluationError, match="time_source"):
        load_predictions(path)


def test_load_predictions_accepts_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_predictions(run_dir / "predictions.csv", [prediction_row("a", 1, 0.7)])
    data = load_predictions(run_dir)
    assert data.path.name == "predictions.csv"
    assert data.sample_count == 1
    assert data.is_mock is True
    assert data.model_names == ("unit_model",)


def test_evaluate_frame_splits_cold_start_groups(tmp_path):
    rows = [
        prediction_row("a", 1, 0.9, cold_start="True", history_length=0),
        prediction_row("b", 0, 0.1, cold_start="True", history_length=0),
        prediction_row("c", 1, 0.8),
        prediction_row("d", 0, 0.2),
    ]
    path = write_predictions(tmp_path / "p.csv", rows)
    data = load_predictions(path)
    groups = {row.group: row for row in evaluate_frame(data)}
    assert set(groups) == {GROUP_ALL, GROUP_NON_COLD, GROUP_COLD}
    assert groups[GROUP_ALL].sample_count == 4
    assert groups[GROUP_NON_COLD].sample_count == 2
    assert groups[GROUP_COLD].sample_count == 2
    assert groups[GROUP_NON_COLD].roc_auc == pytest.approx(1.0)
    assert groups[GROUP_COLD].roc_auc == pytest.approx(1.0)


def test_evaluate_frame_excludes_cold_start_from_all_group(tmp_path):
    rows = [
        prediction_row("a", 1, 0.9, cold_start="True", history_length=0),
        prediction_row("b", 0, 0.1),
    ]
    data = load_predictions(write_predictions(tmp_path / "p.csv", rows))
    rows_out = {row.group: row for row in evaluate_frame(data, include_cold_start=False)}
    assert rows_out[GROUP_ALL].sample_count == 1
    assert "已排除冷启动" in rows_out[GROUP_ALL].group_description


def test_run_metrics_writes_csv_and_json(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [
        prediction_row("a", 0, 0.1), prediction_row("b", 1, 0.9),
    ])
    payload = run_metrics([path], tmp_path / "out", overwrite=True)
    metrics_json = tmp_path / "out" / "metrics" / "metrics.json"
    metrics_csv = tmp_path / "out" / "metrics" / "metrics.csv"
    assert metrics_json.is_file() and metrics_csv.is_file()
    assert json.loads(metrics_json.read_text(encoding="utf-8")) == payload
    assert payload["inputs"][0]["sample_count"] == 2
    assert payload["configuration"]["threshold"] == 0.5
    frame = pd.read_csv(metrics_csv)
    assert set(frame["group"]) == {GROUP_ALL, GROUP_NON_COLD, GROUP_COLD}


def test_run_metrics_refuses_overwrite_without_flag(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [prediction_row("a", 1, 0.6)])
    run_metrics([path], tmp_path / "out", overwrite=True)
    with pytest.raises(EvaluationError, match="--overwrite"):
        run_metrics([path], tmp_path / "out", overwrite=False)


def test_run_metrics_records_undefined_auc_reason(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [
        prediction_row("a", 1, 0.9), prediction_row("b", 1, 0.2),
    ])
    payload = run_metrics([path], tmp_path / "out", overwrite=True)
    all_row = next(row for row in payload["rows"] if row["group"] == GROUP_ALL)
    assert all_row["roc_auc"] is None
    assert all_row["auc_status"] == AUC_SINGLE_CLASS
    assert any("AUC 不可计算" in note for note in payload["notes"])
