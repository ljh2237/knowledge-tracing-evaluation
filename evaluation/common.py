"""评估模块共享的预测结果校验、元数据读取与产物写出。

本模块是六个评估子模块的公共基础层，职责只有三件事：

1. 严格读取并校验负责人 A 交付的 ``predictions.csv``（缺列、空文件、重复 ID、
   非法标签、非法概率、非有限数值全部明确失败，绝不静默丢弃或补默认值）。
2. 关联同目录的 ``run_summary.json``，取得可验证的数据来源与运行配置。
3. 以 UTF-8 写出评估产物（CSV / JSON），并把 NumPy 与 NaN 转成可序列化的 JSON 值。

校验规则与 ``docs/EVALUATION_INTERFACE.md`` 一致。所有错误都抛出
``EvaluationError``，由 CLI 统一转换为退出码 2。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from data_loader import read_csv_strings
from contracts import DataValidationError
from io_utils import write_csv, write_json

EVALUATION_VERSION = "1.0.0"

#: 最少的评估必需列。缺少任何一列都无法在不编造事实的前提下计算指标。
REQUIRED_COLUMNS: tuple[str, ...] = ("record_id", "y_true", "y_pred")

#: 负责人 A 的 predictions.csv 全字段。缺失的可选列会使对应专项分析被跳过并说明原因。
KNOWN_COLUMNS: tuple[str, ...] = (
    "record_id", "student_id", "question_id", "skill_id", "skill_index", "timestamp",
    "event_order", "time_source", "y_true", "y_pred", "predicted_label", "history_length",
    "delta_time_seconds", "cold_start", "model_name", "is_mock", "source_row", "schema_version",
)

#: 只在明确了时间来源时才允许使用的时间字段。
TIME_SOURCES = ("timestamp", "order")

_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "f"})

#: 错误信息中最多列举多少个出错的 record_id，避免超长报错。
_MAX_EXAMPLES = 5


class EvaluationError(ValueError):
    """评估输入无法在不编造事实的前提下解释。"""


def _examples(values: Iterable[object]) -> str:
    items = [str(value) for value in values]
    head = ", ".join(items[:_MAX_EXAMPLES])
    return head if len(items) <= _MAX_EXAMPLES else f"{head} 等 {len(items)} 项"


def sha256_of_text(text: str) -> str:
    """对内存中的文本计算 SHA256，用于复现说明与产物指纹。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """当前 UTC 时间，统一评估产物的时间戳格式。"""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """把 NumPy 标量、数组、NaN/Infinity 递归转换为可 JSON 序列化的值。

    ``io_utils.write_json`` 使用 ``allow_nan=False``，因此 NaN 必须显式转成 null，
    否则会在写出阶段抛出难以定位的异常。
    """
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def read_json_document(path: Path, *, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"无法读取{description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{description}必须是 JSON 对象: {path}")
    return payload


def load_run_summary(predictions_path: Path) -> tuple[dict | None, Path | None]:
    """读取与 predictions.csv 同目录的 run_summary.json；不存在时返回 (None, None)。"""
    candidate = predictions_path.parent / "run_summary.json"
    if not candidate.is_file():
        return None, None
    try:
        return read_json_document(candidate, description="运行摘要"), candidate
    except EvaluationError:
        # 摘要只是审计信息，损坏时不应阻断指标计算，但也不能假装它可用。
        return None, candidate


@dataclass(frozen=True)
class PredictionData:
    """一个模型一次运行的预测结果及其审计元数据。"""

    frame: pd.DataFrame
    path: Path
    digest: str
    model_names: tuple[str, ...]
    is_mock: bool | None
    time_sources: tuple[str, ...]
    run_summary: dict | None
    run_summary_path: Path | None
    label: str

    @property
    def sample_count(self) -> int:
        return int(len(self.frame))

    @property
    def positive_count(self) -> int:
        return int(self.frame["y_true"].sum())

    @property
    def negative_count(self) -> int:
        return self.sample_count - self.positive_count

    @property
    def has_cold_start(self) -> bool:
        return "cold_start" in self.frame.columns

    @property
    def has_delta_time(self) -> bool:
        return "delta_time_seconds" in self.frame.columns

    @property
    def experiment_status(self) -> str:
        if self.run_summary and isinstance(self.run_summary.get("experiment_status"), str):
            return self.run_summary["experiment_status"]
        if self.is_mock is True:
            return "mock_pipeline_only"
        if self.is_mock is False:
            return "external_model_inference"
        return "unknown"

    @property
    def data_nature(self) -> str:
        """数据性质说明，用于报告与图表水印，避免模拟结果被误读。"""
        return "模拟模型输出（仅用于程序功能验证）" if self.is_mock else "外部模型输出"

    def describe(self, *, relative_to: Path | None = None) -> dict:
        path = self.path
        if relative_to is not None:
            try:
                path = path.relative_to(relative_to)
            except ValueError:
                path = self.path
        return {
            "label": self.label,
            "source_file": path.as_posix(),
            "source_sha256": self.digest,
            "model_name": self.model_names[0] if self.model_names else "",
            "model_names": list(self.model_names),
            "is_mock": self.is_mock,
            "data_nature": self.data_nature,
            "experiment_status": self.experiment_status,
            "time_source": ",".join(self.time_sources),
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_rate": (self.positive_count / self.sample_count) if self.sample_count else None,
            "cold_start_count": (
                int(self.frame["cold_start"].sum()) if self.has_cold_start else None
            ),
            "has_delta_time": self.has_delta_time,
            "run_summary_available": self.run_summary is not None,
            "run_summary_input_file": (self.run_summary or {}).get("input_file"),
            "run_summary_input_sha256": (self.run_summary or {}).get("input_sha256"),
            "run_summary_configuration": (self.run_summary or {}).get("configuration"),
        }


def _parse_boolean_column(frame: pd.DataFrame, column: str, path: Path) -> pd.Series | None:
    """把 True/False 文本列解析为布尔列；列不存在返回 None，含未知取值则失败。"""
    if column not in frame.columns:
        return None
    normalized = frame[column].astype(str).str.strip().str.lower()
    valid = normalized.isin(_TRUE_TOKENS | _FALSE_TOKENS)
    if not bool(valid.all()):
        offenders = frame.loc[~valid, "record_id"].tolist()
        raise EvaluationError(
            f"{path} 的 {column} 列必须为布尔值（True/False），"
            f"以下记录取值非法: {_examples(offenders)}"
        )
    return normalized.isin(_TRUE_TOKENS)


def _parse_label_column(frame: pd.DataFrame, path: Path) -> pd.Series:
    """y_true 必须是 0/1；空值、小数标签与其他取值都失败。"""
    raw = frame["y_true"].astype(str).str.strip()
    valid = raw.isin({"0", "1", "0.0", "1.0"})
    if not bool(valid.all()):
        offenders = frame.loc[~valid, "record_id"].tolist()
        raise EvaluationError(
            f"{path} 的 y_true 必须是 0 或 1，以下记录取值非法: {_examples(offenders)}"
        )
    return raw.astype(float).astype(int)


def _parse_probability_column(frame: pd.DataFrame, path: Path) -> pd.Series:
    """y_pred 必须是 [0,1] 内的有限数值；不裁剪、不换默认值。"""
    numeric = pd.to_numeric(frame["y_pred"], errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    in_range = (numeric >= 0) & (numeric <= 1)
    valid = finite & in_range.to_numpy()
    if not bool(valid.all()):
        offenders = frame.loc[~valid, "record_id"].tolist()
        raise EvaluationError(
            f"{path} 的 y_pred 必须是 [0,1] 内的有限数值，"
            f"以下记录取值非法: {_examples(offenders)}"
        )
    return numeric.astype(float)


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def load_predictions(path: str | Path, *, label: str | None = None) -> PredictionData:
    """严格读取 predictions.csv 并完成全部输入校验。

    校验顺序：文件存在且非空 → 必需列齐全 → record_id 非空且唯一 → y_true 为 0/1
    → y_pred 为 [0,1] 有限数值 → 布尔列与 time_source 取值合法。
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise EvaluationError(f"预测文件不存在: {path}")
    if path.is_dir():
        path = path / "predictions.csv"
        if not path.is_file():
            raise EvaluationError(f"运行目录中缺少 predictions.csv: {path.parent}")
    try:
        raw = read_csv_strings(path)
    except DataValidationError as exc:
        raise EvaluationError(f"无法解析预测 CSV: {exc}") from exc
    if raw.empty:
        raise EvaluationError(f"预测文件没有数据行: {path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise EvaluationError(f"{path} 缺少必需列: {missing}；必需列为 {list(REQUIRED_COLUMNS)}")

    frame = pd.DataFrame(index=raw.index)
    frame["record_id"] = raw["record_id"].astype(str).str.strip()
    blank_ids = frame["record_id"] == ""
    if bool(blank_ids.any()):
        raise EvaluationError(
            f"{path} 存在空 record_id，无法对齐样本（CSV 行号 {_examples(raw.index[blank_ids.to_numpy()] + 2)}）"
        )
    duplicated = frame["record_id"][frame["record_id"].duplicated(keep=False)]
    if not duplicated.empty:
        raise EvaluationError(
            f"{path} 的 record_id 必须唯一，重复取值: {_examples(duplicated.unique())}"
        )

    frame["y_true"] = _parse_label_column(raw, path)
    frame["y_pred"] = _parse_probability_column(raw, path)
    frame["student_id"] = raw["student_id"].astype(str) if "student_id" in raw.columns else ""
    frame["question_id"] = raw["question_id"].astype(str) if "question_id" in raw.columns else ""
    frame["skill_id"] = raw["skill_id"].astype(str) if "skill_id" in raw.columns else ""

    cold_start = _parse_boolean_column(raw, "cold_start", path)
    frame["cold_start"] = cold_start if cold_start is not None else False
    delta = _optional_numeric_column(raw, "delta_time_seconds")
    if delta is not None:
        # 负的时间间隔不是真实遗忘场景，直接失败而不是取绝对值。
        negative = delta.notna() & (delta < 0)
        if bool(negative.any()):
            raise EvaluationError(
                f"{path} 存在负的 delta_time_seconds: {_examples(frame.loc[negative, 'record_id'])}"
            )
        frame["delta_time_seconds"] = delta
    history_length = _optional_numeric_column(raw, "history_length")
    if history_length is not None:
        frame["history_length"] = history_length

    if "time_source" in raw.columns:
        sources = tuple(sorted(set(raw["time_source"].astype(str).str.strip())))
        unknown = [source for source in sources if source not in TIME_SOURCES]
        if unknown:
            raise EvaluationError(f"{path} 的 time_source 只允许 {list(TIME_SOURCES)}，实际: {unknown}")
    else:
        sources = ()

    is_mock_column = _parse_boolean_column(raw, "is_mock", path)
    if is_mock_column is None:
        is_mock: bool | None = None
        model_names: tuple[str, ...] = ()
    else:
        distinct = set(is_mock_column.tolist())
        if len(distinct) != 1:
            # 一个文件代表一次运行，is_mock 混用说明输入被拼接，必须拒绝。
            raise EvaluationError(f"{path} 的 is_mock 在同一文件中不一致，无法判定数据性质")
        is_mock = bool(distinct.pop())
        if "model_name" not in raw.columns:
            model_names = ()
        else:
            model_names = tuple(sorted(set(raw["model_name"].astype(str).str.strip())))
            if len(model_names) > 1:
                raise EvaluationError(f"{path} 包含多个 model_name: {list(model_names)}")

    summary, summary_path = load_run_summary(path)
    if label is None:
        label = model_names[0] if model_names else path.parent.name or path.stem
    return PredictionData(
        frame=frame, path=path, digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_names=model_names, is_mock=is_mock, time_sources=sources,
        run_summary=summary, run_summary_path=summary_path, label=label,
    )


def resolve_predictions(path: str | Path) -> Path:
    """允许把预测 CSV 或完整运行目录交给同一个参数。"""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "predictions.csv"
    return candidate


def ensure_output_dir(root: str | Path, *, overwrite: bool) -> Path:
    """准备评估输出根目录；已有内容时必须显式要求覆盖。"""
    root = Path(root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise EvaluationError(f"输出路径已存在且不是目录: {root}")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise EvaluationError(f"输出目录已存在且非空: {root}；如需替换请显式使用 --overwrite")
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_table(path: str | Path, rows: Sequence[dict], columns: Sequence[str]) -> Path:
    """按固定列序写出 UTF-8 CSV；空结果保留表头。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=list(columns))
    write_csv(path, frame)
    return path


def write_document(path: str | Path, payload: Any) -> Path:
    """写出可读的 UTF-8 JSON，自动转换 NumPy 与 NaN。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, to_jsonable(payload))
    return path


def write_text(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path
