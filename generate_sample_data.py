"""Generate reproducible synthetic records, never presented as real learner data."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import write_csv


def generate_sample_data(path: Path, students: int = 12, events: int = 24, seed: int = 20261013) -> pd.DataFrame:
    if students <= 0 or events <= 0 or seed < 0:
        raise ValueError("students / events 必须为正数，seed 必须非负")
    rng = np.random.default_rng(seed)
    skills = ["K01", "K02", "K03", "K04", "K05", "K06"]
    rows = []
    for student in range(students):
        current = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=8)))
        mastery = rng.uniform(0.25, 0.75, size=len(skills))
        for event in range(events):
            skill = int(rng.integers(len(skills)))
            current += timedelta(seconds=int(rng.choice([120, 300, 1800, 86400, 259200])))
            correct = int(rng.random() < mastery[skill])
            mastery[skill] = min(0.95, mastery[skill] + 0.025)
            response = "" if event % 11 == 0 else int(rng.integers(12000, 120000))
            tokens = skills[skill]
            if event % 9 == 0:
                tokens += ";" + skills[(skill + 1) % len(skills)]
            rows.append({
                "record_id": f"sim-{student + 1:03d}-{event + 1:03d}",
                "student_id": f"S{student + 1:03d}", "question_id": f"Q{skill + 1:02d}-{event % 5 + 1}",
                "skill_id": tokens, "correct": correct, "timestamp": current.isoformat(),
                "response_time_ms": response,
            })
    # Deliberately shuffle to exercise chronological loading.
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    write_csv(path, frame)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="生成可复现的合成答题数据")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "sample_data.csv")
    parser.add_argument("--students", type=int, default=12)
    parser.add_argument("--events", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    try:
        result = generate_sample_data(args.output, args.students, args.events, args.seed)
        print(f"已生成合成数据 {len(result)} 条: {args.output}")
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"生成失败: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
