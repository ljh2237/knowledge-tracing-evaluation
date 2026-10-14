from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def csv_file(tmp_path):
    def write(rows, name="input.csv") -> Path:
        path = tmp_path / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path
    return write


@pytest.fixture
def records():
    return [
        {"record_id": "a2", "student_id": "001", "question_id": "Q2", "skill_id": "k1;k2",
         "correct": "0", "timestamp": "2026-10-01T10:02:00+08:00", "response_time_ms": "2000"},
        {"record_id": "b1", "student_id": "002", "question_id": "Q1", "skill_id": "k1",
         "correct": "0", "timestamp": "2026-10-01T10:00:00+08:00", "response_time_ms": ""},
        {"record_id": "a1", "student_id": "001", "question_id": "Q1", "skill_id": "k1",
         "correct": "1", "timestamp": "2026-10-01T10:00:00+08:00", "response_time_ms": ""},
        {"record_id": "a3", "student_id": "001", "question_id": "Q3", "skill_id": "k2",
         "correct": "1", "timestamp": "2026-10-01T10:03:00+08:00", "response_time_ms": "1000"},
    ]
