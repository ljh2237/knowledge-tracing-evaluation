from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from contracts import DataValidationError, ModelValidationError, Prediction, Target
from data_loader import SkillVocabulary, load_data, parse_skills, parse_timestamp, read_csv_strings
from generate_sample_data import generate_sample_data
from io_utils import sha256_file
from mock_model import MockModel
from model_adapter import ModelAdapter
from pipeline import iter_requests, run_pipeline


def test_load_groups_students_and_sorts(csv_file, records):
    data = load_data(csv_file(records))
    assert [r.record_id for r in data.records] == ["a1", "a2", "a3", "b1"]
    assert data.records[0].student_id == "001"
    assert data.records[1].skill_ids == (1, 2)
    assert data.summary["missing_response_time_rows"] == 2
    assert data.records[0].response_time_ms is None


@pytest.mark.parametrize("column,value", [
    ("student_id", ""), ("question_id", " "), ("skill_id", ""), ("skill_id", "k1;"),
    ("skill_id", "-1"), ("correct", ""), ("correct", "2"), ("correct", "0.5"),
    ("correct", "NaN"), ("timestamp", "not-a-time"), ("timestamp", "2026-10-01T10:00:00"),
    ("timestamp", "1700000000"), ("timestamp", "NaT"), ("response_time_ms", "-1"),
    ("response_time_ms", "NaN"), ("response_time_ms", "inf"), ("response_time_ms", "bad"),
    ("record_id", ""), ("available_at", "2026-09-01T00:00:00Z"),
])
def test_invalid_rows_are_rejected(csv_file, records, column, value):
    records[0][column] = value
    with pytest.raises(DataValidationError):
        load_data(csv_file(records))


def test_drop_invalid_is_audited(csv_file, records):
    records[0]["correct"] = ""
    data = load_data(csv_file(records), invalid_policy="drop")
    assert len(data.records) == 3
    assert data.summary["rejected_rows"] == 1
    assert data.rejected.iloc[0]["source_row"] == 2
    assert data.rejected.iloc[0]["record_id"] == "a2"
    assert "correct" in data.rejected.iloc[0]["reason"]


def test_duplicate_id_does_not_duplicate_predictions(csv_file, records):
    records[0]["record_id"] = "a1"
    with pytest.raises(DataValidationError, match="重复"):
        load_data(csv_file(records))


@pytest.mark.parametrize("content", ["", "student_id,student_id\n1,2\n", ",correct\n1,1\n",
                                        "student_id,question_id\na,b,c\n", "student_id\n\"unclosed"])
def test_empty_or_malformed_csv(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(DataValidationError):
        read_csv_strings(path)


def test_header_only_and_missing_columns(tmp_path, csv_file):
    header = tmp_path / "empty.csv"
    header.write_text("student_id,question_id,skill_id,correct,timestamp\n")
    with pytest.raises(DataValidationError, match="没有答题"):
        load_data(header)
    with pytest.raises(DataValidationError, match="必填列"):
        load_data(csv_file([{"student_id": "1"}]))


def test_all_invalid_drop_fails(csv_file, records):
    for record in records:
        record["correct"] = ""
    with pytest.raises(DataValidationError, match="没有有效"):
        load_data(csv_file(records), invalid_policy="drop")


def test_literal_na_and_leading_zeros_are_ids(csv_file, records):
    records[0]["student_id"] = "NA"
    records[0]["question_id"] = "0001"
    data = load_data(csv_file(records))
    selected = next(r for r in data.records if r.record_id == "a2")
    assert selected.student_id == "NA" and selected.question_id == "0001"


def test_missing_record_ids_are_generated(csv_file, records):
    for record in records:
        del record["record_id"]
    data = load_data(csv_file(records))
    assert len({r.record_id for r in data.records}) == len(records)


def test_missing_clock_requires_valid_order(csv_file, records):
    for i, record in enumerate(records):
        record["timestamp"] = ""
        record["event_order"] = str(i)
    data = load_data(csv_file(records))
    assert data.summary["time_source"] == "order"
    assert all(r.timestamp is None for r in data.records)
    assert all(r.delta_time_seconds is None for r in iter_requests(data.records))
    records[0]["event_order"] = "1.5"
    with pytest.raises(DataValidationError, match="非负整数"):
        load_data(csv_file(records))


def test_mixed_order_and_clock_rejected(csv_file, records):
    records[0]["timestamp"] = ""
    records[0]["event_order"] = "2"
    with pytest.raises(DataValidationError, match="混用"):
        load_data(csv_file(records))


@pytest.mark.parametrize("unit,value", [("s", "1700000000"), ("ms", "1700000000000")])
def test_numeric_time_has_explicit_units(unit, value):
    assert parse_timestamp(value, unit) == 1700000000


def test_timezones_are_normalized():
    assert parse_timestamp("2026-10-01T08:00:00+08:00", "iso") == parse_timestamp("2026-10-01T00:00:00Z", "iso")


def test_vocab_persistence_and_unknowns(tmp_path, csv_file, records):
    vocabulary = SkillVocabulary({"k2": 1, "k1": 2})
    vocabulary.save(tmp_path / "vocab.json")
    loaded = SkillVocabulary.load(tmp_path / "vocab.json")
    data = load_data(csv_file(records), vocabulary=loaded)
    assert data.records[0].skill_ids == (2,)
    records[0]["skill_id"] = "new"
    with pytest.raises(DataValidationError, match="训练词表"):
        load_data(csv_file(records), vocabulary=loaded)
    data = load_data(csv_file(records), vocabulary=loaded, unknown_skill_policy="unk")
    assert data.summary["unknown_skill_rows"] == 1
    assert next(r for r in data.records if r.record_id == "a2").skill_ids == (0,)


@pytest.mark.parametrize("mapping", [{"k1": True}, {"k1": 0}, {"k1": 1, "k2": 1}, {"k1": 2},
                                      {"k1;k2": 1}, {}, {"": 1}])
def test_invalid_vocab(tmp_path, mapping):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"schema_version": "1.0", "unknown_id": 0, "skills": mapping}))
    with pytest.raises(DataValidationError):
        SkillVocabulary.load(path)


def test_multi_skill_normalization():
    assert parse_skills(" k2;k1;k2 ") == ("k1", "k2")


def test_request_never_includes_current_or_future_label(csv_file, records):
    data = load_data(csv_file(records))
    requests = list(iter_requests(data.records))
    assert "correct" not in {f.name for f in fields(Target)}
    assert "response_time_ms" not in {f.name for f in fields(Target)}
    assert [h.record_id for h in requests[1].history] == ["a1"]
    assert requests[1].delta_time_seconds == 120
    for request in requests:
        assert all(h.student_id == request.target.student_id for h in request.history)
        assert all(h.timestamp < request.target.timestamp for h in request.history)
        assert all(h.available_at <= request.target.timestamp for h in request.history)
        assert all(h.record_id != request.target.record_id for h in request.history)


def test_equal_timestamp_excludes_simultaneous_answers(csv_file, records):
    records[0]["timestamp"] = records[2]["timestamp"]
    requests = list(iter_requests(load_data(csv_file(records)).records))
    assert requests[0].history == requests[1].history == ()
    assert len(requests[2].history) == 2


def test_answer_unavailable_until_finished(csv_file, records):
    records[2]["available_at"] = "2026-10-01T10:02:30+08:00"
    requests = list(iter_requests(load_data(csv_file(records)).records))
    assert requests[1].history == ()
    assert [h.record_id for h in requests[2].history] == ["a1", "a2"]


def test_same_bundle_answers_are_excluded(csv_file, records):
    records[0]["interaction_group"] = "bundle1"
    records[2]["interaction_group"] = "bundle1"
    requests = list(iter_requests(load_data(csv_file(records)).records))
    assert requests[1].history == ()
    assert len(requests[2].history) == 2


def test_history_window(csv_file, records):
    requests = list(iter_requests(load_data(csv_file(records)).records, max_history=1))
    assert [h.record_id for h in requests[2].history] == ["a2"]
    with pytest.raises(ValueError):
        list(iter_requests(load_data(csv_file(records)).records, max_history=0))


def test_mock_cold_start_and_history(csv_file, records):
    requests = list(iter_requests(load_data(csv_file(records)).records))
    output = MockModel().predict(requests)
    assert [p.probability for p in output] == [0.5, 0.6, 0.4, 0.5]


def test_changing_target_and_future_answers_cannot_change_earlier_predictions(csv_file, records):
    model = MockModel()
    first = model.predict(list(iter_requests(load_data(csv_file(records, "before.csv")).records)))
    records[0]["correct"] = "1"  # a2: current target of second prediction
    records[3]["correct"] = "0"  # a3: future interaction
    second = model.predict(list(iter_requests(load_data(csv_file(records, "after.csv")).records)))
    assert first[:2] == second[:2]
    assert first[3] == second[3]  # another student's prediction


class ReturnModel:
    name = "test_model"
    is_mock = True

    def __init__(self, result):
        self.result = result

    def predict(self, requests):
        return self.result


@pytest.mark.parametrize("value", [-0.1, 1.01, float("nan"), float("inf"), "0.5", True, None,
                                    np.complex128(0.5)])
def test_invalid_model_probabilities(csv_file, records, value):
    request = next(iter_requests(load_data(csv_file(records)).records))
    adapter = ModelAdapter(ReturnModel([Prediction(request.target.record_id, value)]))
    with pytest.raises(ModelValidationError):
        adapter.predict([request])


def test_model_count_ids_type_and_order(csv_file, records):
    batch = list(iter_requests(load_data(csv_file(records)).records))[:2]
    ids = [r.target.record_id for r in batch]
    for response in [[], [Prediction(ids[0], 0.5)] * 2, [Prediction("wrong", 0.5)] * 2,
                     [0.5, 0.5], "bad", None]:
        with pytest.raises(ModelValidationError):
            ModelAdapter(ReturnModel(response)).predict(batch)
    adapter = ModelAdapter(ReturnModel([Prediction(ids[1], 0.7), Prediction(ids[0], 0.2)]))
    assert adapter.predict(batch) == [Prediction(ids[0], 0.2), Prediction(ids[1], 0.7)]
    assert adapter.predict([]) == []
    with pytest.raises(ModelValidationError, match="请求"):
        adapter.predict([batch[0], batch[0]])


def test_factory_and_missing_real_model(csv_file, records):
    requests = list(iter_requests(load_data(csv_file(records)).records))
    adapter = ModelAdapter.from_factory("examples.constant_plugin:create_model", {"probability": 0.8})
    assert all(p.probability == 0.8 for p in adapter.predict(requests))
    with pytest.raises(ModelValidationError, match="尚未接入"):
        ModelAdapter.from_factory("examples.training_model_template:create_model", {})
    with pytest.raises(ModelValidationError):
        ModelAdapter.from_factory("not_valid", {})


def test_predictor_exception_and_missing_metadata(csv_file, records):
    class Broken:
        name = "broken"
        is_mock = False

        def predict(self, requests):
            raise RuntimeError("missing checkpoint")
    batch = list(iter_requests(load_data(csv_file(records)).records))
    with pytest.raises(ModelValidationError, match="missing checkpoint"):
        ModelAdapter(Broken()).predict(batch)
    for model in [object(), type("NoMetadata", (), {"predict": lambda self, r: []})(),
                  type("NoFlag", (), {"name": "missing_flag", "predict": lambda self, r: []})()]:
        with pytest.raises(ModelValidationError):
            ModelAdapter(model)


def test_pipeline_exports_aligned_csv_and_hashes(tmp_path, csv_file, records):
    path = csv_file(records)
    summary = run_pipeline(path, tmp_path / "results", ModelAdapter(MockModel()), batch_size=2)
    output = pd.read_csv(tmp_path / "results" / "predictions.csv", dtype={"student_id": str})
    assert len(output) == 4 and summary["batches"] == 2
    assert output.record_id.tolist() == ["a1", "a2", "a3", "b1"]
    assert output.y_true.tolist() == [1, 0, 1, 0]
    assert output.is_mock.all()
    assert output.student_id.tolist() == ["001", "001", "001", "002"]
    for name, fingerprint in summary["artifacts"].items():
        assert sha256_file(tmp_path / "results" / name) == fingerprint
    assert summary["input_sha256"] == sha256_file(path)


def test_batch_size_does_not_change_predictions(tmp_path, csv_file, records):
    path = csv_file(records)
    for batch_size in [1, 2, 3, 64]:
        run_pipeline(path, tmp_path / f"b{batch_size}", ModelAdapter(MockModel()), batch_size=batch_size)
    reference = (tmp_path / "b1" / "predictions.csv").read_bytes()
    assert all((tmp_path / f"b{b}" / "predictions.csv").read_bytes() == reference for b in [2, 3, 64])


def test_failed_run_keeps_previous_output_and_success_replaces(tmp_path, csv_file, records):
    path, output = csv_file(records), tmp_path / "results"
    run_pipeline(path, output, ModelAdapter(MockModel()))
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ModelValidationError):
        run_pipeline(path, output, ModelAdapter(ReturnModel([])), overwrite=True)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    run_pipeline(path, output, ModelAdapter(MockModel()), overwrite=True)
    assert len(list(output.iterdir())) == 4
    assert not list(tmp_path.glob(".kt-*"))


def test_output_safety_and_parameters(tmp_path, csv_file, records):
    path = csv_file(records)
    adapter = ModelAdapter(MockModel())
    for kwargs in [{"batch_size": 0}, {"max_history": -1}, {"threshold": 2}, {"threshold": float("nan")}]:
        with pytest.raises(ValueError):
            run_pipeline(path, tmp_path / "out", adapter, **kwargs)
    with pytest.raises(ValueError):
        run_pipeline(path, tmp_path, adapter, overwrite=True)
    output = tmp_path / "out"
    output.mkdir()
    (output / "important.txt").write_text("keep me")
    with pytest.raises(ValueError, match="其他文件"):
        run_pipeline(path, output, adapter, overwrite=True)
    assert (output / "important.txt").read_text() == "keep me"
    with pytest.raises(ValueError, match="已存在"):
        run_pipeline(path, output, adapter)


def test_drop_pipeline_writes_rejected_rows(tmp_path, csv_file, records):
    records[0]["correct"] = "-1"
    output = tmp_path / "out"
    summary = run_pipeline(csv_file(records), output, ModelAdapter(MockModel()), invalid_policy="drop")
    assert summary["prediction_count"] == 3
    assert len(pd.read_csv(output / "rejected_rows.csv")) == 1


def test_synthetic_generation_reproducible(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    generate_sample_data(first, students=2, events=8, seed=7)
    generate_sample_data(second, students=2, events=8, seed=7)
    assert first.read_bytes() == second.read_bytes()
    assert len(load_data(first).records) == 16
