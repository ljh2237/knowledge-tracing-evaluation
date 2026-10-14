from pathlib import Path

import pandas as pd
import pytest

from contracts import DataValidationError
from data_loader import load_data
from dataset_converters import convert_assistments, convert_ednet, main, source_time
from io_utils import write_csv
from pipeline import iter_requests

ROOT = Path(__file__).resolve().parents[1]


def test_assistments_multi_skill_collapse_and_order(tmp_path):
    converted, summary = convert_assistments(ROOT / "examples" / "assistments_fixture.csv")
    assert len(converted) == 4 and summary["collapsed_rows"] == 1
    assert converted.iloc[0].skill_id == "10;20"
    path = tmp_path / "converted.csv"
    write_csv(path, converted)
    loaded = load_data(path)
    assert [r.event_order for r in loaded.records] == [100, 101, 102, 103]
    assert loaded.records[0].student_id == "0001"
    assert all(request.delta_time_seconds is None for request in iter_requests(loaded.records))


def test_assistments_conflicting_duplicates(csv_file):
    path = csv_file([
        {"order_id": "1", "user_id": "u1", "problem_id": "q1", "skill_id": "10", "correct": "1"},
        {"order_id": "1", "user_id": "u1", "problem_id": "q1", "skill_id": "20", "correct": "0"},
    ])
    with pytest.raises(DataValidationError, match="冲突"):
        convert_assistments(path)


def test_assistments_naive_source_requires_timezone(csv_file):
    path = csv_file([{"order_id": "1", "user_id": "u1", "problem_id": "q1", "skill_id": "10_20",
                      "correct": "1", "start_time": "2026-10-01 10:00:00", "ms_first_response": "2000"}])
    with pytest.raises(DataValidationError, match="时区"):
        convert_assistments(path)
    converted, _ = convert_assistments(path, source_timezone="Asia/Shanghai")
    assert converted.iloc[0].timestamp == "2026-10-01T02:00:00+00:00"
    assert converted.iloc[0].available_at == "2026-10-01T02:00:02+00:00"


def test_assistments_start_requires_answer_availability(csv_file):
    rows = [{"order_id": "1", "user_id": "u1", "problem_id": "q1", "skill_id": "10",
             "correct": "1", "start_time": "2026-10-01T10:00:00Z"}]
    with pytest.raises(DataValidationError, match="可见时间"):
        convert_assistments(csv_file(rows))
    rows[0]["end_time"] = "2026-10-01T10:00:30Z"
    converted, _ = convert_assistments(csv_file(rows))
    assert converted.iloc[0].available_at == rows[0]["end_time"].replace("Z", "+00:00")


def test_ednet_joins_labels_multi_skills_and_availability(tmp_path):
    converted, summary = convert_ednet(ROOT / "examples" / "u1.csv", ROOT / "examples" / "questions_fixture.csv")
    assert summary["converted_rows"] == 3
    assert converted.correct.tolist() == [1, 0, 0]
    assert converted.skill_id.tolist() == ["10;20", "20", "10;20"]
    path = tmp_path / "ednet.csv"
    write_csv(path, converted)
    requests = list(iter_requests(load_data(path).records))
    assert requests[1].history == ()  # same bundle and not yet available
    assert len(requests[2].history) == 2
    assert requests[2].history[0].available_at - requests[2].history[0].timestamp == 47


def test_ednet_multiple_user_files(tmp_path):
    source = (ROOT / "examples" / "u1.csv").read_text()
    (tmp_path / "u1.csv").write_text(source)
    (tmp_path / "u2.csv").write_text(source)
    converted, summary = convert_ednet(tmp_path, ROOT / "examples" / "questions_fixture.csv")
    assert summary["source_files"] == 2
    assert len(set(converted.record_id)) == 6


@pytest.mark.parametrize("column,value", [("user_answer", ""), ("user_answer", "x"),
                                         ("solving_id", ""), ("elapsed_time", "-1"),
                                         ("elapsed_time", "nan"), ("question_id", "unknown")])
def test_ednet_invalid_record(tmp_path, column, value):
    frame = pd.read_csv(ROOT / "examples" / "u1.csv", dtype=str, keep_default_na=False)
    frame.loc[0, column] = value
    write_csv(tmp_path / "u1.csv", frame)
    with pytest.raises((DataValidationError, ValueError)):
        convert_ednet(tmp_path / "u1.csv", ROOT / "examples" / "questions_fixture.csv")


def test_ednet_duplicate_question_and_invalid_filename(tmp_path):
    questions = pd.read_csv(ROOT / "examples" / "questions_fixture.csv", dtype=str)
    write_csv(tmp_path / "questions.csv", pd.concat([questions, questions]))
    with pytest.raises(DataValidationError, match="唯一"):
        convert_ednet(ROOT / "examples" / "u1.csv", tmp_path / "questions.csv")
    (tmp_path / "wrong.csv").write_text((ROOT / "examples" / "u1.csv").read_text())
    with pytest.raises(DataValidationError, match="文件名"):
        convert_ednet(tmp_path / "wrong.csv", ROOT / "examples" / "questions_fixture.csv")


def test_converter_cli_and_no_overwrite_source(tmp_path):
    output = tmp_path / "out.csv"
    assert main(["--dataset", "assistments", "--input", str(ROOT / "examples" / "assistments_fixture.csv"),
                 "--output", str(output)]) == 0
    assert output.exists() and output.with_suffix(".conversion.json").exists()
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--dataset", "assistments", "--input", str(output), "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == original
    with pytest.raises(SystemExit):
        main(["--dataset", "ednet", "--input", str(ROOT / "examples" / "u1.csv"), "--output", str(output)])


def test_numeric_source_time():
    assert source_time("1700000000000", "ms", None) == "2023-11-14T22:13:20+00:00"
