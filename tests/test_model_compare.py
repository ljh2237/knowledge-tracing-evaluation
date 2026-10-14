"""多模型对比测试：样本对齐、标签冲突、覆盖不一致与排序不变性。"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from evaluation.common import EvaluationError
from evaluation.evaluator import GROUP_ALL, GROUP_COLD, GROUP_NON_COLD
from evaluation.model_compare import (
    build_comparison_markdown,
    compare_models,
    run_comparison,
)


def write(path, rows, name="predictions.csv"):
    target = path / name if path.is_dir() else path
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def row(record_id, y_true, y_pred, *, model="m", is_mock="True", student="S1",
        question="Q1", skill="K01", cold_start="False", delta=10.0):
    return {
        "record_id": record_id, "student_id": student, "question_id": question,
        "skill_id": skill, "time_source": "timestamp", "y_true": y_true, "y_pred": y_pred,
        "predicted_label": int(float(y_pred) >= 0.5), "history_length": 1,
        "delta_time_seconds": delta, "cold_start": cold_start, "model_name": model,
        "is_mock": is_mock,
    }


ALIGNED_LABELS = {f"r{index}": index % 2 for index in range(8)}


def aligned_rows(model, *, good=True):
    """good=True 时概率与标签一致（AUC=1.0），good=False 时完全相反（AUC=0.0）。"""
    return [
        row(rid, label, (0.9 if label else 0.1) if good else (0.1 if label else 0.9), model=model)
        for rid, label in ALIGNED_LABELS.items()
    ]


def test_compare_aligns_by_record_id_and_computes_deltas(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    result = compare_models([first, second], labels=["a", "b"], baseline=0)
    assert result["aligned_sample_count"] == 8
    assert result["baseline"] == {"index": 0, "label": "a"}
    rows = {item["label"]: item for item in result["rows"] if item["group"] == GROUP_ALL}
    assert rows["a"]["roc_auc"] == pytest.approx(1.0)
    assert rows["b"]["roc_auc"] == pytest.approx(0.0)
    assert rows["a"]["delta_roc_auc"] == pytest.approx(0.0)
    assert rows["b"]["delta_roc_auc"] == pytest.approx(-1.0)


def test_compare_is_invariant_to_model_order(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    forward = compare_models([first, second], labels=["a", "b"], baseline="a")
    backward = compare_models([second, first], labels=["b", "a"], baseline="a")
    forward_rows = {item["label"]: item for item in forward["rows"] if item["group"] == GROUP_ALL}
    backward_rows = {item["label"]: item for item in backward["rows"] if item["group"] == GROUP_ALL}
    assert forward_rows["a"]["roc_auc"] == pytest.approx(backward_rows["a"]["roc_auc"])
    assert forward_rows["b"]["delta_accuracy"] == pytest.approx(backward_rows["b"]["delta_accuracy"])


def test_compare_rejects_missing_samples_by_default(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False)[:-1])
    with pytest.raises(EvaluationError, match="样本覆盖不一致"):
        compare_models([first, second])


def test_compare_intersect_policy_records_dropped_samples(tmp_path):
    # model_b 多出 1 个样本，被丢弃的样本必须记在 model_b 名下。
    first = write(tmp_path / "a.csv", aligned_rows("model_a")[:-1])
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    result = compare_models([first, second], missing_policy="intersect")
    assert result["aligned_sample_count"] == 7
    assert result["union_sample_count"] == 8
    assert result["coverage"]["model_a"]["dropped"] == 0
    assert result["coverage"]["model_b"]["dropped"] == 1
    assert result["coverage"]["model_b"]["dropped_examples"] == ["r7"]
    assert any("intersect" in note for note in result["notes"])


def test_compare_rejects_conflicting_labels(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    rows = aligned_rows("model_b", good=False)
    rows[0]["y_true"] = 1 - rows[0]["y_true"]
    second = write(tmp_path / "b.csv", rows)
    with pytest.raises(EvaluationError, match="真实标签 y_true .*冲突"):
        compare_models([first, second])


def test_compare_rejects_conflicting_sample_metadata(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    rows = aligned_rows("model_b", good=False)
    rows[0]["question_id"] = "DIFFERENT"
    second = write(tmp_path / "b.csv", rows)
    with pytest.raises(EvaluationError, match="question_id .*不一致"):
        compare_models([first, second])


def test_compare_notes_when_identity_cannot_be_cross_checked(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    rows = aligned_rows("model_b", good=False)
    for item in rows:
        item["student_id"] = ""
    second = write(tmp_path / "b.csv", rows)
    result = compare_models([first, second])
    assert any("无法交叉核对" in note for note in result["notes"])


def test_compare_requires_at_least_two_models(tmp_path):
    single = write(tmp_path / "a.csv", aligned_rows("model_a"))
    with pytest.raises(EvaluationError, match="至少需要两个"):
        compare_models([single])


def test_compare_rejects_duplicate_labels_and_bad_baseline(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    with pytest.raises(EvaluationError, match="--labels 存在重复名称"):
        compare_models([first, second], labels=["x", "x"])
    with pytest.raises(EvaluationError, match="--labels 数量"):
        compare_models([first, second], labels=["only-one"])
    with pytest.raises(EvaluationError, match="基准模型名称不存在"):
        compare_models([first, second], baseline="nope")
    with pytest.raises(EvaluationError, match="基准模型序号超出范围"):
        compare_models([first, second], baseline=5)


def test_compare_reports_cold_start_subgroups(tmp_path):
    rows_a = [row(f"r{index}", index % 2, 0.9, model="a",
                  cold_start=str(index < 2).capitalize()) for index in range(6)]
    rows_b = [row(f"r{index}", index % 2, 0.2, model="b",
                  cold_start=str(index < 2).capitalize()) for index in range(6)]
    result = compare_models([write(tmp_path / "a.csv", rows_a), write(tmp_path / "b.csv", rows_b)])
    groups = {item["group"] for item in result["rows"]}
    assert groups == {GROUP_ALL, GROUP_NON_COLD, GROUP_COLD}
    cold = next(item for item in result["rows"]
                if item["group"] == GROUP_COLD and item["label"] == "a")
    assert cold["sample_count"] == 2


def test_compare_warns_when_all_models_are_mock(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    result = compare_models([first, second])
    assert any("全部为模拟模型" in caveat for caveat in result["scope_caveats"])
    assert any("不能证明" in caveat for caveat in result["scope_caveats"])


def test_compare_notes_differing_run_summaries(tmp_path):
    """两个运行目录的输入指纹不同时必须给出提示，而不是静默通过。"""
    first_dir, second_dir = tmp_path / "run_a", tmp_path / "run_b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = write(first_dir / "predictions.csv", aligned_rows("model_a"))
    second = write(second_dir / "predictions.csv", aligned_rows("model_b", good=False))
    (first_dir / "run_summary.json").write_text(
        json.dumps({"input_sha256": "aaa", "configuration": {"batch_size": 64}}), encoding="utf-8"
    )
    (second_dir / "run_summary.json").write_text(
        json.dumps({"input_sha256": "bbb", "configuration": {"batch_size": 32}}), encoding="utf-8"
    )
    result = compare_models([first, second], missing_policy="strict")
    assert result["run_summaries"]["model_a"]["available"] is True
    assert result["run_summaries"]["model_b"]["input_sha256"] == "bbb"
    assert any("SHA256" in note for note in result["notes"])
    assert any("运行配置不完全相同" in note for note in result["notes"])


def test_compare_notes_missing_run_summary(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    result = compare_models([first, second])
    assert result["run_summaries"]["model_a"]["available"] is False
    assert any("未找到 run_summary.json" in note for note in result["notes"])


def test_run_comparison_writes_all_three_formats(tmp_path):
    first = write(tmp_path / "a.csv", aligned_rows("model_a"))
    second = write(tmp_path / "b.csv", aligned_rows("model_b", good=False))
    result = run_comparison([first, second], tmp_path / "out", overwrite=True)
    out = tmp_path / "out" / "comparison"
    assert (out / "model_comparison.csv").is_file()
    assert (out / "model_comparison.md").is_file()
    stored = json.loads((out / "model_comparison.json").read_text(encoding="utf-8"))
    assert stored["aligned_sample_count"] == result["aligned_sample_count"]
    markdown = (out / "model_comparison.md").read_text(encoding="utf-8")
    assert "结论边界" in markdown and "多模型对比测试" in markdown


def test_build_comparison_markdown_marks_undefined_auc(tmp_path):
    rows_a = [row(f"r{index}", 1, 0.9, model="a") for index in range(4)]
    rows_b = [row(f"r{index}", 1, 0.2, model="b") for index in range(4)]
    result = compare_models([write(tmp_path / "a.csv", rows_a), write(tmp_path / "b.csv", rows_b)])
    markdown = build_comparison_markdown(result)
    assert "不适用" in markdown
    assert result["ranking_by_auc"][0]["auc_status"] == "undefined_single_class"
