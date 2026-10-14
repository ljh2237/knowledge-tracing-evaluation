"""Strict CSV loading, persistent skill vocabulary, chronological sequences."""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from contracts import DataValidationError, Interaction, SCHEMA_VERSION
from io_utils import write_json

REQUIRED_COLUMNS = {"student_id", "question_id", "skill_id", "correct"}


def read_csv_strings(path: str | Path) -> pd.DataFrame:
    """Preserve leading zeros and literal 'NA' IDs; reject malformed CSV."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream, strict=True)
            header = next(rows, [])
            if not header or not any(header):
                raise DataValidationError("CSV 为空或没有表头")
            if any(not h.strip() for h in header) or len(set(header)) != len(header):
                raise DataValidationError("CSV 表头包含空列名或重复列名")
            for row in rows:
                if not row:  # completely blank lines are harmless
                    continue
                if len(row) != len(header):
                    raise DataValidationError(f"CSV 第 {rows.line_num} 行的列数与表头不一致")
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except DataValidationError:
        raise
    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"无法读取 CSV {path}: {exc}") from exc


def parse_skills(value: str) -> tuple[str, ...]:
    parts = value.strip().split(";")
    if any(not p.strip() or p.strip() == "-1" for p in parts):
        raise DataValidationError("skill_id 缺失或包含无效标签")
    return tuple(sorted({p.strip() for p in parts}))


def parse_timestamp(value: str, unit: str) -> float:
    value = value.strip()
    try:
        if unit in {"s", "ms"}:
            if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                raise ValueError("需要数值 Unix 时间戳")
            stamp = pd.to_datetime(float(value), unit=unit, utc=True, errors="raise")
        else:
            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                raise ValueError("数值时间戳必须明确指定 --timestamp-unit s 或 ms")
            # A timezone is required so the host timezone cannot change results.
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                raise ValueError("时间字符串必须包含时区，例如 +08:00 或 Z")
            stamp = stamp.tz_convert("UTC")
        if pd.isna(stamp):
            raise ValueError("时间戳为空")
        seconds = stamp.timestamp()
        if not math.isfinite(seconds):
            raise ValueError("时间戳必须有限")
        return seconds
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError(f"无效时间戳 {value!r}: {exc}") from exc


@dataclass(frozen=True)
class SkillVocabulary:
    mapping: dict[str, int]

    @classmethod
    def fit(cls, skills: list[tuple[str, ...]]) -> SkillVocabulary:
        tokens = sorted({token for group in skills for token in group})
        return cls({token: i + 1 for i, token in enumerate(tokens)})

    @classmethod
    def load(cls, path: Path) -> SkillVocabulary:
        import json

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            mapping = payload["skills"]
            if payload["schema_version"] != SCHEMA_VERSION or payload["unknown_id"] != 0:
                raise ValueError("schema_version 或 unknown_id 不兼容")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("skills 必须是非空对象")
            for token, number in mapping.items():
                if parse_skills(token) != (token,):
                    raise ValueError("词表键必须是单个非空知识点")
                if type(number) is not int or number <= 0:
                    raise ValueError("知识点编号必须是正整数")
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("知识点编号重复")
            if sorted(mapping.values()) != list(range(1, len(mapping) + 1)):
                raise ValueError("知识点编号必须从 1 连续递增")
            return cls(mapping)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DataValidationError(f"无效词表 {path}: {exc}") from exc

    def save(self, path: Path) -> None:
        write_json(path, {"schema_version": SCHEMA_VERSION, "unknown_id": 0, "skills": self.mapping})

    def encode(self, skills: tuple[str, ...], policy: str) -> tuple[int, ...]:
        unknown = [s for s in skills if s not in self.mapping]
        if unknown and policy == "error":
            raise DataValidationError(f"训练词表不存在知识点: {unknown}")
        return tuple(self.mapping.get(s, 0) for s in skills)


@dataclass
class LoadedData:
    records: list[Interaction]
    vocabulary: SkillVocabulary
    rejected: pd.DataFrame
    summary: dict


def load_data(
    path: str | Path,
    *,
    invalid_policy: str = "error",
    timestamp_unit: str = "iso",
    vocabulary: SkillVocabulary | None = None,
    unknown_skill_policy: str = "error",
) -> LoadedData:
    if invalid_policy not in {"error", "drop"}:
        raise ValueError("invalid_policy 必须为 error 或 drop")
    if timestamp_unit not in {"iso", "s", "ms"}:
        raise ValueError("timestamp_unit 必须为 iso、s 或 ms")
    if unknown_skill_policy not in {"error", "unk"}:
        raise ValueError("unknown_skill_policy 必须为 error 或 unk")
    raw = read_csv_strings(path)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise DataValidationError(f"缺少必填列: {sorted(missing)}")
    if "timestamp" not in raw and "event_order" not in raw:
        raise DataValidationError("必须提供 timestamp 或 event_order 列")
    if raw.empty:
        raise DataValidationError("CSV 没有答题记录")

    parsed: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    missing_response = 0
    for position, row in enumerate(raw.to_dict("records"), start=2):
        try:
            for column in ("student_id", "question_id"):
                if not row[column].strip():
                    raise DataValidationError(f"{column} 缺失")
            record_id = row.get("record_id", f"r{position - 1:08d}").strip()
            if not record_id:
                raise DataValidationError("record_id 缺失")
            if record_id in seen:
                raise DataValidationError(f"record_id 重复: {record_id}")
            # Never silently invent labels from missing values or fractional numbers.
            label = row["correct"].strip()
            if label not in {"0", "1", "0.0", "1.0"}:
                raise DataValidationError("correct 必须为 0 或 1")
            skills = parse_skills(row["skill_id"])
            stamp = row.get("timestamp", "").strip()
            order = row.get("event_order", "").strip()
            if stamp:
                timestamp = parse_timestamp(stamp, timestamp_unit)
                event_order = None
                availability = row.get("available_at", "").strip()
                available_at = parse_timestamp(availability, timestamp_unit) if availability else timestamp
                if available_at < timestamp:
                    raise DataValidationError("available_at 不得早于 timestamp")
            else:
                if not re.fullmatch(r"\d+", order):
                    raise DataValidationError("timestamp 缺失且 event_order 不是非负整数")
                if row.get("available_at", "").strip():
                    raise DataValidationError("顺序模式不能提供 available_at")
                timestamp, available_at, event_order = None, None, int(order)
            response = row.get("response_time_ms", "").strip()
            response_ms = float(response) if response else None
            if response_ms is not None and (not math.isfinite(response_ms) or response_ms < 0):
                raise DataValidationError("response_time_ms 必须是有限非负数或留空")
            parsed.append(dict(
                record_id=record_id, student_id=row["student_id"].strip(),
                question_id=row["question_id"].strip(), skill_tokens=skills,
                correct=int(float(label)), timestamp=timestamp, available_at=available_at,
                event_order=event_order, response_time_ms=response_ms,
                interaction_group=row.get("interaction_group", "").strip(), source_row=position,
            ))
            seen.add(record_id)
            missing_response += int(response_ms is None)
        except (DataValidationError, ValueError) as exc:
            rejected.append({**row, "source_row": position, "reason": str(exc)})

    if rejected and invalid_policy == "error":
        first = rejected[0]
        raise DataValidationError(
            f"发现 {len(rejected)} 条无效记录；第 {first['source_row']} 条记录: {first['reason']}。"
            "可使用 --invalid-policy drop 导出拒收清单并继续。"
        )
    if not parsed:
        raise DataValidationError("清洗后没有有效答题记录")
    sources = {"timestamp" if p["timestamp"] is not None else "order" for p in parsed}
    if len(sources) != 1:
        raise DataValidationError("同一输入不允许混用时间戳与仅顺序记录，请先分别处理")
    vocab = vocabulary or SkillVocabulary.fit([p["skill_tokens"] for p in parsed])
    records = [Interaction(**p, skill_ids=vocab.encode(p["skill_tokens"], unknown_skill_policy)) for p in parsed]
    records.sort(key=lambda r: (
        r.student_id, r.timestamp if r.timestamp is not None else r.event_order, r.source_row,
    ))
    unknown_count = sum(0 in r.skill_ids for r in records)
    return LoadedData(
        records=records, vocabulary=vocab,
        rejected=pd.DataFrame(rejected, columns=[*raw.columns, "source_row", "reason"]),
        summary={
            "input_rows": len(raw), "accepted_rows": len(records), "rejected_rows": len(rejected),
            "students": len({r.student_id for r in records}), "skill_count": len(vocab.mapping),
            "unknown_skill_rows": unknown_count, "missing_response_time_rows": missing_response,
            "time_source": next(iter(sources)),
            "vocabulary_source": "provided" if vocabulary is not None else "fitted_on_input",
        },
    )
