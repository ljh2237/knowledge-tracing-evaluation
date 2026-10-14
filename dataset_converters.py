"""Explicit adapters for ASSISTments and EdNet KT1 subsets, with audited joins."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from contracts import DataValidationError
from data_loader import load_data, parse_skills, parse_timestamp, read_csv_strings
from io_utils import write_csv, write_json


def require_columns(frame: pd.DataFrame, names: set[str]) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise DataValidationError(f"源数据缺少列: {sorted(missing)}")


def source_time(value: str, unit: str, source_timezone: str | None) -> str:
    if not value.strip():
        raise DataValidationError("源时间列缺失，不能补造时间")
    if unit != "iso":
        return pd.Timestamp(parse_timestamp(value, unit), unit="s", tz="UTC").isoformat()
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        if not source_timezone:
            raise DataValidationError("源时间没有时区，请显式提供 --source-timezone")
        stamp = stamp.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
    return stamp.tz_convert("UTC").isoformat()


def convert_assistments(
    input_path: Path, *, time_column: str | None = None, timestamp_unit: str = "iso",
    source_timezone: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    raw = read_csv_strings(input_path)
    require_columns(raw, {"user_id", "problem_id", "skill_id", "correct", "order_id"})
    if time_column:
        require_columns(raw, {time_column})
    elif "start_time" in raw:
        time_column = "start_time"
    rows = []
    for row in raw.to_dict("records"):
        # Both old per-skill rows and the corrected skill1_skill2 format are accepted.
        skill = ";".join(parse_skills(row["skill_id"].replace("_", ";")))
        record = {
            "record_id": "assist:" + row["user_id"].strip() + ":" + row["order_id"].strip(),
            "student_id": row["user_id"], "question_id": row["problem_id"],
            "skill_id": skill, "correct": row["correct"],
            "timestamp": source_time(row[time_column], timestamp_unit, source_timezone) if time_column else "",
            "event_order": row["order_id"] if not time_column else "",
            "response_time_ms": row.get("ms_first_response", ""),
        }
        if time_column == "start_time":
            if row.get("end_time", "").strip():
                record["available_at"] = source_time(row["end_time"], timestamp_unit, source_timezone)
            else:
                response = row.get("ms_first_response", "").strip()
                if not response:
                    raise DataValidationError("ASSISTments 使用 start_time 时需要 end_time 或 ms_first_response，不能猜测答案可见时间")
                milliseconds = float(response)
                if not 0 <= milliseconds < float("inf"):
                    raise DataValidationError("ASSISTments ms_first_response 必须为有限非负毫秒数")
                record["available_at"] = (
                    pd.Timestamp(record["timestamp"]) + pd.Timedelta(milliseconds, unit="ms")
                ).isoformat()
        rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise DataValidationError("ASSISTments 没有答题记录")
    collapsed = []
    for _, group in frame.groupby("record_id", sort=False):
        for column in frame.columns:
            if column != "skill_id" and group[column].nunique(dropna=False) != 1:
                raise DataValidationError(f"ASSISTments 同一 order_id 的 {column} 冲突，不能合并")
        record = group.iloc[0].to_dict()
        record["skill_id"] = ";".join(sorted({s for value in group.skill_id for s in value.split(";")}))
        collapsed.append(record)
    return pd.DataFrame(collapsed), {
        "dataset": "ASSISTments", "source_rows": len(raw), "converted_rows": len(collapsed),
        "collapsed_rows": len(raw) - len(collapsed),
        "time_source": "timestamp" if time_column else "order",
        "note": "只有 order_id 时不提供时间间隔；原始标签不插补。",
    }


def convert_ednet(input_path: Path, questions_path: Path) -> tuple[pd.DataFrame, dict]:
    """Directory of u*.csv files or one u*.csv; questions.csv is mandatory.

    Timestamp = question presentation; availability = presentation + elapsed_time.
    solving_id is retained as an exclusion group to prevent within-bundle leakage.
    """
    questions = read_csv_strings(questions_path)
    require_columns(questions, {"question_id", "correct_answer", "tags"})
    if questions.question_id.duplicated().any() or (questions.question_id.str.strip() == "").any():
        raise DataValidationError("EdNet 题库 question_id 必须唯一且非空")
    if not questions.correct_answer.isin(["a", "b", "c", "d"]).all():
        raise DataValidationError("EdNet 题库 correct_answer 必须在 a-d 内")
    files = sorted(input_path.glob("u*.csv")) if input_path.is_dir() else [input_path]
    if not files:
        raise DataValidationError("EdNet 输入目录中没有 u*.csv")
    rows = []
    for path in files:
        if not path.stem.startswith("u") or not path.stem[1:].isdigit():
            raise DataValidationError("EdNet 文件名必须为 u{整数}.csv，以保留真实 student_id")
        raw = read_csv_strings(path)
        require_columns(raw, {"timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"})
        if raw.empty:
            raise DataValidationError(f"EdNet 学生文件为空: {path.name}")
        joined = raw.merge(questions[["question_id", "correct_answer", "tags"]],
                           on="question_id", how="left", validate="many_to_one", indicator=True)
        if (joined["_merge"] != "both").any():
            missing = joined.loc[joined["_merge"] != "both", "question_id"].unique().tolist()
            raise DataValidationError(f"EdNet 题库未匹配题目: {missing[:10]}")
        for number, row in enumerate(joined.to_dict("records"), start=1):
            if row["user_answer"] not in {"a", "b", "c", "d"}:
                raise DataValidationError(f"EdNet {path.name} 第 {number} 条 user_answer 缺失或无效")
            if not row["solving_id"].strip():
                raise DataValidationError("EdNet solving_id 缺失，不能防止同题组泄漏")
            elapsed = float(row["elapsed_time"])
            if not pd.notna(elapsed) or not 0 <= elapsed < float("inf"):
                raise DataValidationError("EdNet elapsed_time 必须为有限非负毫秒数")
            timestamp = parse_timestamp(row["timestamp"], "ms")
            rows.append({
                "record_id": f"ednet:{path.stem}:{number:08d}", "student_id": path.stem,
                "question_id": row["question_id"], "skill_id": ";".join(parse_skills(row["tags"])),
                "correct": int(row["user_answer"] == row["correct_answer"]),
                "timestamp": pd.Timestamp(timestamp, unit="s", tz="UTC").isoformat(),
                "available_at": pd.Timestamp(timestamp + elapsed / 1000, unit="s", tz="UTC").isoformat(),
                "response_time_ms": row["elapsed_time"], "interaction_group": row["solving_id"],
            })
    return pd.DataFrame(rows), {
        "dataset": "EdNet-KT1", "source_files": len(files), "converted_rows": len(rows),
        "time_source": "timestamp", "timestamp_unit_in_source": "ms",
        "note": "先关联题库计算标签，再以答题完成时间控制历史可见性，同 solving_id 的答案相互隔离。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASSISTments / EdNet KT1 → 标准答题 CSV")
    parser.add_argument("--dataset", required=True, choices=["assistments", "ednet"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--time-column")
    parser.add_argument("--timestamp-unit", choices=["iso", "s", "ms"], default="iso")
    parser.add_argument("--source-timezone")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.input.resolve() or (args.questions and args.output.resolve() == args.questions.resolve()):
            raise DataValidationError("转换输出不能覆盖源文件")
        if args.dataset == "assistments":
            frame, summary = convert_assistments(args.input, time_column=args.time_column,
                                                 timestamp_unit=args.timestamp_unit,
                                                 source_timezone=args.source_timezone)
        else:
            if args.questions is None:
                raise DataValidationError("EdNet 转换需要 --questions questions.csv")
            frame, summary = convert_ednet(args.input, args.questions)
        # Validate in a temporary file before publishing any conversion result.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="kt-convert-") as temporary:
            check = Path(temporary) / "check.csv"
            write_csv(check, frame)
            loaded = load_data(check)
        write_csv(args.output, frame)
        write_json(args.output.with_suffix(".conversion.json"), {**summary, "validation": loaded.summary})
        print(f"转换完成：{len(frame)} 条记录 → {args.output}")
        return 0
    except (DataValidationError, ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"转换失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
