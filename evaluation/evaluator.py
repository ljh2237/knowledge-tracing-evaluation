"""模块一：模型评价指标计算（ROC-AUC / Accuracy / Log Loss）。

只消费负责人 A 的 ``predictions.csv``。指标口径：

- ROC-AUC 与 Log Loss 一律使用 ``y_pred``（答对概率），绝不使用 ``predicted_label``。
- Accuracy 用 ``y_pred >= threshold`` 得到的分类标签，默认 threshold = 0.5，可配置。
- Brier 分数按概率均方误差计算，作为补充参考。

``y_pred`` 是「当前目标题答对的预测概率」，不是知识掌握概率。本模块的所有输出都
保留 ``is_mock`` 标识与数据性质说明，模拟结果不得用于证明真实 DKT/AKT 精度。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from evaluation.common import (
    EVALUATION_VERSION,
    EvaluationError,
    PredictionData,
    ensure_output_dir,
    load_predictions,
    resolve_predictions,
    utc_now_iso,
    write_document,
    write_table,
)

GROUP_ALL = "all"
GROUP_NON_COLD = "non_cold_start"
GROUP_COLD = "cold_start"

GROUP_DESCRIPTIONS = {
    GROUP_ALL: "全部有效样本",
    GROUP_NON_COLD: "非冷启动样本（history_length > 0）",
    GROUP_COLD: "冷启动样本（history_length = 0）",
}

AUC_COMPUTED = "computed"
AUC_SINGLE_CLASS = "undefined_single_class"
AUC_UNAVAILABLE = "unavailable"

METRIC_COLUMNS = (
    "label", "model_name", "is_mock", "group", "group_description", "threshold",
    "sample_count", "positive_count", "negative_count", "positive_rate",
    "roc_auc", "auc_status", "auc_reason", "accuracy", "log_loss", "brier_score",
    "source_file", "source_sha256", "data_nature", "experiment_status",
)

SCOPE_NOTE = (
    "指标在给定预测文件上计算；is_mock=True 表示模拟模型输出，"
    "只能验证评估程序正确性，不能证明 DKT/AKT 的真实预测精度。"
)


@dataclass(frozen=True)
class MetricRow:
    """一个模型在一个样本子集上的完整指标记录。"""

    label: str
    model_name: str
    is_mock: bool | None
    group: str
    group_description: str
    threshold: float
    sample_count: int
    positive_count: int
    negative_count: int
    positive_rate: float | None
    roc_auc: float | None
    auc_status: str
    auc_reason: str | None
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    source_file: str
    source_sha256: str
    data_nature: str
    experiment_status: str

    def as_record(self) -> dict:
        return asdict(self)


def compute_core_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """纯函数指标计算，供评估、对比与测试共用。

    返回字典包含 ``roc_auc`` / ``auc_status`` / ``auc_reason`` / ``accuracy`` /
    ``log_loss`` / ``brier_score`` 及样本计数。单一类别时 AUC 明确标记为不可计算，
    并给出原因，不编造数值。
    """
    if not 0 <= threshold <= 1 or not np.isfinite(threshold):
        raise EvaluationError(f"分类阈值必须在 [0,1] 内: {threshold}")
    labels = np.asarray(y_true, dtype=float).ravel()
    probabilities = np.asarray(y_pred, dtype=float).ravel()
    if labels.size != probabilities.size:
        raise EvaluationError(
            f"标签数 {labels.size} 与预测数 {probabilities.size} 不一致，无法计算指标"
        )
    if labels.size == 0:
        return {
            "sample_count": 0, "positive_count": 0, "negative_count": 0,
            "positive_rate": None, "roc_auc": None, "auc_status": AUC_UNAVAILABLE,
            "auc_reason": "样本数为 0，无法计算任何指标",
            "accuracy": None, "log_loss": None, "brier_score": None,
        }
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise EvaluationError("y_true 只能包含 0 或 1")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise EvaluationError("y_pred 必须是 [0,1] 内的有限数值")

    sample_count = int(labels.size)
    positive_count = int(labels.sum())
    negative_count = sample_count - positive_count

    predicted = (probabilities >= threshold).astype(int)
    accuracy = float(accuracy_score(labels.astype(int), predicted))
    # labels=[0,1] 让单一类别样本也能得到定义明确的 Log Loss。
    log_loss_value = float(log_loss(labels.astype(int), probabilities, labels=[0, 1]))
    brier = float(np.mean((probabilities - labels) ** 2))

    if positive_count == 0 or negative_count == 0:
        roc_auc: float | None = None
        auc_status = AUC_SINGLE_CLASS
        present = "正样本" if positive_count else "负样本"
        auc_reason = (
            f"测试集只包含{present}（正样本 {positive_count} 条、负样本 {negative_count} 条），"
            "ROC-AUC 无定义，不编造数值"
        )
    else:
        try:
            roc_auc = float(roc_auc_score(labels.astype(int), probabilities))
            auc_status, auc_reason = AUC_COMPUTED, None
        except ValueError as exc:
            roc_auc, auc_status, auc_reason = None, AUC_UNAVAILABLE, f"scikit-learn 无法计算 AUC: {exc}"

    return {
        "sample_count": sample_count, "positive_count": positive_count,
        "negative_count": negative_count, "positive_rate": positive_count / sample_count,
        "roc_auc": roc_auc, "auc_status": auc_status, "auc_reason": auc_reason,
        "accuracy": accuracy, "log_loss": log_loss_value, "brier_score": brier,
    }


def evaluate_frame(
    data: PredictionData,
    *,
    threshold: float = 0.5,
    include_cold_start: bool = True,
    relative_to: Path | None = None,
) -> list[MetricRow]:
    """对一个预测文件计算全部样本与冷启动分组的指标。

    ``include_cold_start`` 决定 ``all`` 组是否保留冷启动样本；只要存在 ``cold_start``
    列，就始终额外给出 ``non_cold_start`` 与 ``cold_start`` 两个分组结果。
    """
    frame = data.frame
    if not include_cold_start and data.has_cold_start:
        primary = frame[~frame["cold_start"]]
        primary_description = f"{GROUP_DESCRIPTIONS[GROUP_ALL]}（已排除冷启动）"
    else:
        primary = frame
        primary_description = GROUP_DESCRIPTIONS[GROUP_ALL]

    groups: list[tuple[str, str, pd.DataFrame]] = [(GROUP_ALL, primary_description, primary)]
    if data.has_cold_start:
        groups.append((GROUP_NON_COLD, GROUP_DESCRIPTIONS[GROUP_NON_COLD], frame[~frame["cold_start"]]))
        groups.append((GROUP_COLD, GROUP_DESCRIPTIONS[GROUP_COLD], frame[frame["cold_start"]]))

    described = data.describe(relative_to=relative_to)
    rows: list[MetricRow] = []
    for group, description, subset in groups:
        metrics = compute_core_metrics(
            subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), threshold=threshold
        )
        rows.append(MetricRow(
            label=data.label,
            model_name=described["model_name"],
            is_mock=data.is_mock,
            group=group,
            group_description=description,
            threshold=float(threshold),
            source_file=described["source_file"],
            source_sha256=data.digest,
            data_nature=data.data_nature,
            experiment_status=data.experiment_status,
            **metrics,
        ))
    return rows


def run_metrics(
    prediction_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    threshold: float = 0.5,
    include_cold_start: bool = True,
    overwrite: bool = False,
    seed: int | None = None,
    relative_to: Path | None = None,
) -> dict:
    """计算一个或多个预测文件的指标，写出 metrics.csv 与 metrics.json。"""
    if not prediction_paths:
        raise EvaluationError("至少需要一个预测文件（--predictions）")
    datasets = [load_predictions(resolve_predictions(path)) for path in prediction_paths]

    rows: list[MetricRow] = []
    for data in datasets:
        rows.extend(evaluate_frame(
            data, threshold=threshold, include_cold_start=include_cold_start, relative_to=relative_to
        ))

    notes = [SCOPE_NOTE]
    if not include_cold_start:
        notes.append("已按 --exclude-cold-start 排除冷启动样本；冷启动分组仍单独列出。")
    undefined = [row for row in rows if row.auc_status != AUC_COMPUTED]
    for row in undefined:
        notes.append(f"[{row.label} / {row.group}] AUC 不可计算：{row.auc_reason}")

    root = Path(output_dir).expanduser().resolve()
    # 每个子模块只负责自己的子目录，因此 metrics/、forgetting/ 等可以独立覆盖，
    # `all` 命令也不会因为其他子模块已经写过文件而失败。
    metrics_dir = ensure_output_dir(root / "metrics", overwrite=overwrite)

    payload = {
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": utc_now_iso(),
        "scope": SCOPE_NOTE,
        "configuration": {
            "threshold": float(threshold),
            "include_cold_start": bool(include_cold_start),
            "seed": seed,
            "metric_source": "scikit-learn",
        },
        "inputs": [data.describe(relative_to=relative_to) for data in datasets],
        "rows": [row.as_record() for row in rows],
        "notes": notes,
    }
    write_table(metrics_dir / "metrics.csv", [row.as_record() for row in rows], METRIC_COLUMNS)
    write_document(metrics_dir / "metrics.json", payload)
    return payload
