"""Causal request generation and transactional, traceable artifact export."""
from __future__ import annotations

import bisect
import heapq
import itertools
import platform
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import Interaction, PredictionRequest, SCHEMA_VERSION, Target, VERSION
from data_loader import SkillVocabulary, load_data
from io_utils import sha256_file, write_csv, write_json
from model_adapter import ModelAdapter

RESULT_COLUMNS = [
    "record_id", "student_id", "question_id", "skill_id", "skill_index", "timestamp",
    "event_order", "time_source", "y_true", "y_pred", "predicted_label", "history_length",
    "delta_time_seconds", "cold_start", "model_name", "is_mock", "source_row", "schema_version",
]


def iter_requests(records: list[Interaction], max_history: int = 256):
    """Only earlier, available, same-student answers can enter a request.

    Equal timestamps/orders are treated as simultaneous. An interaction_group
    (e.g. EdNet bundle session) cannot leak labels within that group.
    """
    if max_history <= 0:
        raise ValueError("max_history 必须为正整数")
    for _, student_records in itertools.groupby(records, key=lambda r: r.student_id):
        pending: list[tuple[float | int, int, Interaction]] = []
        available: list[tuple[tuple[float | int, int], Interaction]] = []

        def time_key(record):
            return record.timestamp if record.timestamp is not None else record.event_order
        for point, simultaneous in itertools.groupby(student_records, key=time_key):
            group = list(simultaneous)
            while pending and pending[0][0] <= point:
                _, row_number, history_record = heapq.heappop(pending)
                # source rows need not be chronological, so order by actual event time.
                key = (time_key(history_record), row_number)
                bisect.insort(available, (key, history_record), key=lambda pair: pair[0])
            for record in group:
                history: list[Interaction] = []
                for _, previous in reversed(available):
                    if record.interaction_group and previous.interaction_group == record.interaction_group:
                        continue
                    history.append(previous)
                    if len(history) == max_history:
                        break
                history.reverse()
                time_source = "timestamp" if record.timestamp is not None else "order"
                delta = None
                if record.timestamp is not None and history:
                    delta = record.timestamp - history[-1].available_at
                yield PredictionRequest(
                    Target(record.record_id, record.student_id, record.question_id,
                           record.skill_tokens, record.skill_ids, record.timestamp,
                           record.event_order, time_source),
                    tuple(history), delta,
                )
            # Insert only after all simultaneous targets have been predicted.
            for record in group:
                availability = record.available_at if record.timestamp is not None else record.event_order
                heapq.heappush(pending, (availability, record.source_row, record))


def iso_timestamp(seconds: float | None) -> str:
    return "" if seconds is None else datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    adapter: ModelAdapter,
    *,
    batch_size: int = 64,
    max_history: int = 256,
    invalid_policy: str = "error",
    timestamp_unit: str = "iso",
    vocabulary: SkillVocabulary | None = None,
    unknown_skill_policy: str = "error",
    threshold: float = 0.5,
    overwrite: bool = False,
    model_provenance: dict | None = None,
) -> dict:
    if batch_size <= 0 or max_history <= 0:
        raise ValueError("batch_size 与 max_history 必须为正整数")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 [0,1] 内")
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or not overwrite):
        raise ValueError(f"输出目录已存在: {output_dir}；如需替换请使用 --overwrite")
    if output_dir.exists():
        expected_files = {"predictions.csv", "rejected_rows.csv", "skill_mapping.json", "run_summary.json"}
        actual_files = {path.name for path in output_dir.iterdir()}
        if actual_files and (actual_files != expected_files or any(p.is_dir() for p in output_dir.iterdir())):
            raise ValueError("只能覆盖本框架生成的完整运行目录或空目录，不能删除其他文件")
    # Prevent a run from deleting its own source when replacing an output directory.
    if input_path == output_dir or output_dir in input_path.parents:
        raise ValueError("输出目录不能包含输入数据")
    start = time.perf_counter()
    loaded = load_data(input_path, invalid_policy=invalid_policy, timestamp_unit=timestamp_unit,
                       vocabulary=vocabulary, unknown_skill_policy=unknown_skill_policy)
    by_id = {record.record_id: record for record in loaded.records}
    requests = iter_requests(loaded.records, max_history)
    rows: list[dict] = []
    model_seconds = 0.0
    batches = 0
    while batch := list(itertools.islice(requests, batch_size)):
        model_start = time.perf_counter()
        predictions = adapter.predict(batch)
        model_seconds += time.perf_counter() - model_start
        batches += 1
        for request, prediction in zip(batch, predictions, strict=True):
            record = by_id[request.target.record_id]
            rows.append(dict(
                record_id=record.record_id, student_id=record.student_id, question_id=record.question_id,
                skill_id=";".join(record.skill_tokens), skill_index=";".join(map(str, record.skill_ids)),
                timestamp=iso_timestamp(record.timestamp), event_order=record.event_order,
                time_source=request.target.time_source, y_true=record.correct, y_pred=prediction.probability,
                predicted_label=int(prediction.probability >= threshold),
                history_length=len(request.history), delta_time_seconds=request.delta_time_seconds,
                cold_start=not request.history, model_name=adapter.model.name, is_mock=adapter.model.is_mock,
                source_row=record.source_row, schema_version=SCHEMA_VERSION,
            ))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".kt-run-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        write_csv(staging / "predictions.csv", pd.DataFrame(rows, columns=RESULT_COLUMNS))
        write_csv(staging / "rejected_rows.csv", loaded.rejected)
        loaded.vocabulary.save(staging / "skill_mapping.json")
        summary = {
            "version": VERSION, "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": input_path.name, "input_sha256": sha256_file(input_path),
            "model_name": adapter.model.name, "is_mock": adapter.model.is_mock,
            "model_provenance": model_provenance or {"factory": "mock_model:MockModel"},
            "experiment_status": "mock_pipeline_only" if adapter.model.is_mock else "external_model_inference",
            "data": loaded.summary,
            "configuration": {"batch_size": batch_size, "max_history": max_history,
                              "threshold": threshold, "invalid_policy": invalid_policy,
                              "timestamp_unit": timestamp_unit, "unknown_skill_policy": unknown_skill_policy},
            "prediction_count": len(rows), "cold_start_count": sum(r["cold_start"] for r in rows),
            "batches": batches, "model_seconds": model_seconds,
            "pipeline_seconds_before_export": time.perf_counter() - start,
            "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
            "artifacts": {name: sha256_file(staging / name) for name in
                          ("predictions.csv", "rejected_rows.csv", "skill_mapping.json")},
            "limitations": ["模拟预测不能证明模型精度或遗忘机制有效"] if adapter.model.is_mock else
                           ["模型名称及 is_mock 由接入方声明，需另行审核训练与评估过程"],
        }
        write_json(staging / "run_summary.json", summary)
        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=".kt-backup-", dir=output_dir.parent))
            backup.rmdir()
            output_dir.rename(backup)
        try:
            staging.rename(output_dir)
        except OSError:
            if backup is not None:
                backup.rename(output_dir)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)
