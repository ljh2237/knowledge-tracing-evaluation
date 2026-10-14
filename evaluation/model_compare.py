"""模块三：多模型对比测试。

对齐规则（默认严格，不静默删除样本）：

1. 以 ``record_id`` 为唯一样本标识对齐；各模型必须覆盖完全相同的测试样本。
2. 同一个 ``record_id`` 的 ``y_true`` 必须在所有模型中一致，冲突直接失败。
3. 同一个 ``record_id`` 的 ``student_id`` / ``question_id`` / ``skill_id``
   在所有模型中都必须一致；只有全部模型都带该列时才做交叉核对，否则记录为
   「无法核对」而不是默认通过。
4. 允许显式使用 ``--missing-policy intersect`` 取交集，此时被丢弃的样本数、
   涉及模型与受影响 record_id 样本都会写入产物并给出警告，绝不悄悄比较。

**结论边界**：``evidence/mock_run`` 与 ``evidence/plugin_run`` 都是模拟模型
（``is_mock=True``），它们之间的差异只能验证对比程序本身，不能证明任何新模型
优于基线。真实模型对比必须在训练组交付权重后重新执行。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import pandas as pd

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
    write_text,
)
from evaluation.evaluator import (
    GROUP_ALL,
    GROUP_COLD,
    GROUP_DESCRIPTIONS,
    GROUP_NON_COLD,
    compute_core_metrics,
)

MISSING_POLICIES = ("strict", "intersect")

COMPARISON_COLUMNS = (
    "group", "group_description", "label", "model_name", "is_mock", "is_baseline", "threshold",
    "sample_count", "positive_count", "negative_count", "positive_rate",
    "roc_auc", "auc_status", "auc_reason", "accuracy", "log_loss", "brier_score",
    "delta_roc_auc", "delta_accuracy", "delta_log_loss",
    "source_file", "source_sha256", "data_nature", "experiment_status",
)

_IDENTITY_FIELDS = ("student_id", "question_id", "skill_id")

SCOPE_NOTE = (
    "对比结果只在给定的预测文件之间成立。is_mock=True 的模拟模型之间的差异不能证明任何"
    "新模型优于基线，也不能替代在真实 DKT/AKT 上的对照实验。"
)


@dataclass(frozen=True)
class ComparisonRow:
    group: str
    group_description: str
    label: str
    model_name: str
    is_mock: bool | None
    is_baseline: bool
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
    delta_roc_auc: float | None
    delta_accuracy: float | None
    delta_log_loss: float | None
    source_file: str
    source_sha256: str
    data_nature: str
    experiment_status: str

    def as_record(self) -> dict:
        return asdict(self)


def _resolve_baseline(labels: Sequence[str], baseline: int | str | None) -> int:
    if baseline is None:
        return 0
    if isinstance(baseline, int) and not isinstance(baseline, bool):
        if not 0 <= baseline < len(labels):
            raise EvaluationError(f"基准模型序号超出范围: {baseline}；共 {len(labels)} 个模型")
        return baseline
    if isinstance(baseline, str):
        if baseline.isdigit() and baseline not in labels:
            return _resolve_baseline(labels, int(baseline), None)
        if baseline not in labels:
            raise EvaluationError(f"基准模型名称不存在: {baseline!r}；可选 {list(labels)}")
        return list(labels).index(baseline)
    raise EvaluationError(f"无法解析基准模型: {baseline!r}")


def _check_identity(
    datasets: Sequence[PredictionData], ids: pd.Index
) -> tuple[list[str], list[str]]:
    """核对样本元信息一致性，返回 (notes, errors)。"""
    notes: list[str] = []
    errors: list[str] = []
    for field in _IDENTITY_FIELDS:
        if not all((dataset.frame[field] != "").all() for dataset in datasets):
            notes.append(f"字段 {field} 在部分预测文件中为空，本次无法交叉核对一致性。")
            continue
        reference = datasets[0].frame.set_index("record_id")[field].reindex(ids)
        for dataset in datasets[1:]:
            other = dataset.frame.set_index("record_id")[field].reindex(ids)
            mismatch = reference != other
            if bool(mismatch.any()):
                offenders = list(ids[mismatch.to_numpy()])
                errors.append(
                    f"字段 {field} 在 {datasets[0].label} 与 {dataset.label} 之间不一致，"
                    f"冲突样本示例: {offenders[:5]}（共 {int(mismatch.sum())} 条）"
                )
    return notes, errors


def _check_labels(
    datasets: Sequence[PredictionData], ids: pd.Index
) -> list[str]:
    errors: list[str] = []
    reference = datasets[0].frame.set_index("record_id")["y_true"].reindex(ids)
    for dataset in datasets[1:]:
        other = dataset.frame.set_index("record_id")["y_true"].reindex(ids)
        mismatch = reference != other
        if bool(mismatch.any()):
            offenders = list(ids[mismatch.to_numpy()])
            errors.append(
                f"真实标签 y_true 在 {datasets[0].label} 与 {dataset.label} 之间冲突，"
                f"冲突样本示例: {offenders[:5]}（共 {int(mismatch.sum())} 条）"
            )
    return errors


def _summarise_run_summaries(datasets: Sequence[PredictionData]) -> tuple[dict, list[str]]:
    """核对 run_summary.json 中可验证的数据来源与配置。"""
    notes: list[str] = []
    records: dict[str, dict] = {}
    for dataset in datasets:
        summary = dataset.run_summary
        if summary is None:
            notes.append(f"{dataset.label}: 未找到 run_summary.json，数据来源与配置无法核验。")
            records[dataset.label] = {"available": False}
            continue
        records[dataset.label] = {
            "available": True,
            "experiment_status": summary.get("experiment_status"),
            "input_file": summary.get("input_file"),
            "input_sha256": summary.get("input_sha256"),
            "is_mock": summary.get("is_mock"),
            "configuration": summary.get("configuration"),
            "model_provenance": summary.get("model_provenance"),
        }
    digests = {
        record.get("input_sha256") for record in records.values()
        if record.get("available") and record.get("input_sha256")
    }
    if len(digests) > 1:
        notes.append(
            f"各模型的运行摘要显示输入数据 SHA256 不同（{len(digests)} 种），"
            "说明它们并非来自同一份输入文件；样本对齐结果仍需以 record_id 为准。"
        )
    configurations = {
        str(record.get("configuration")) for record in records.values() if record.get("available")
    }
    if len(configurations) > 1:
        notes.append("各模型的运行配置不完全相同（如 batch_size / threshold），对比时已统一使用本次评估阈值。")
    return records, notes


def compare_models(
    prediction_paths: Sequence[str | Path],
    *,
    labels: Sequence[str] | None = None,
    baseline: int | str | None = None,
    threshold: float = 0.5,
    include_cold_start: bool = True,
    missing_policy: str = "strict",
    relative_to: Path | None = None,
) -> dict:
    """对齐多个模型的预测结果并计算指标差异，返回完整对比结果字典。"""
    if len(prediction_paths) < 2:
        raise EvaluationError("模型对比至少需要两个预测文件（--predictions a.csv b.csv）")
    if missing_policy not in MISSING_POLICIES:
        raise EvaluationError(f"missing_policy 必须是 {list(MISSING_POLICIES)} 之一")
    if labels is not None and len(labels) != len(prediction_paths):
        raise EvaluationError(f"--labels 数量 {len(labels)} 与预测文件数量 {len(prediction_paths)} 不一致")
    if labels is not None and len(set(labels)) != len(labels):
        raise EvaluationError(f"--labels 存在重复名称: {list(labels)}")

    datasets = [
        load_predictions(resolve_predictions(path), label=labels[index] if labels else None)
        for index, path in enumerate(prediction_paths)
    ]
    names = [dataset.label for dataset in datasets]
    if len(set(names)) != len(names):
        raise EvaluationError(f"模型名称重复，无法区分对比对象: {names}；请使用 --labels 指定唯一名称")

    index = _resolve_baseline(names, baseline)
    id_sets = {dataset.label: set(dataset.frame["record_id"]) for dataset in datasets}
    union = set().union(*id_sets.values())
    intersection = set.intersection(*id_sets.values())
    notes: list[str] = []
    coverage: dict[str, dict] = {}

    if len(intersection) != len(union):
        detail = {
            label: {"sample_count": len(ids), "only_here": len(ids - intersection)}
            for label, ids in id_sets.items()
        }
        if missing_policy == "strict":
            raise EvaluationError(
                "各模型的测试样本覆盖不一致，默认严格失败而不静默删除样本："
                f"{detail}；如确实需要取交集，请显式使用 --missing-policy intersect"
            )
        notes.append(
            f"已显式启用 --missing-policy intersect：从 {len(union)} 个样本中取交集的 "
            f"{len(intersection)} 个样本，丢弃 {len(union) - len(intersection)} 个样本。"
            f"丢弃明细：{detail}"
        )
        for label, ids in id_sets.items():
            dropped = ids - intersection
            coverage[label] = {
                "sample_count": len(ids), "used": len(intersection),
                "dropped": len(dropped), "dropped_examples": sorted(dropped)[:5],
            }
    else:
        notes.append(f"各模型覆盖完全相同的 {len(union)} 个测试样本。")
        for label, ids in id_sets.items():
            coverage[label] = {"sample_count": len(ids), "used": len(ids), "dropped": 0,
                               "dropped_examples": []}

    ids = pd.Index(sorted(intersection))
    if ids.empty:
        raise EvaluationError("各模型没有任何共同样本，无法对比")

    identity_notes, identity_errors = _check_identity(datasets, ids)
    errors = [*identity_errors, *_check_labels(datasets, ids)]
    if errors:
        raise EvaluationError("样本对齐失败：\n" + "\n".join(f"- {message}" for message in errors))
    notes.extend(identity_notes)

    run_summaries, summary_notes = _summarise_run_summaries(datasets)
    notes.extend(summary_notes)

    subsets: list[tuple[str, str]] = [(GROUP_ALL, GROUP_DESCRIPTIONS[GROUP_ALL])]
    if all(dataset.has_cold_start for dataset in datasets):
        subsets.append((GROUP_NON_COLD, GROUP_DESCRIPTIONS[GROUP_NON_COLD]))
        subsets.append((GROUP_COLD, GROUP_DESCRIPTIONS[GROUP_COLD]))

    frames = {
        dataset.label: dataset.frame.set_index("record_id").reindex(ids).reset_index()
        for dataset in datasets
    }
    described = {dataset.label: dataset.describe(relative_to=relative_to) for dataset in datasets}

    rows: list[ComparisonRow] = []
    for group, description in subsets:
        baseline_metrics: dict | None = None
        group_rows: list[ComparisonRow] = []
        for position, dataset in enumerate(datasets):
            frame = frames[dataset.label]
            if group == GROUP_ALL:
                subset = frame if include_cold_start else frame[~frame["cold_start"]]
            elif group == GROUP_NON_COLD:
                subset = frame[~frame["cold_start"]]
            else:
                subset = frame[frame["cold_start"]]
            metrics = compute_core_metrics(
                subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), threshold=threshold
            )
            if position == index:
                baseline_metrics = metrics
            info = described[dataset.label]
            group_rows.append(ComparisonRow(
                group=group, group_description=description, label=dataset.label,
                model_name=info["model_name"], is_mock=dataset.is_mock, is_baseline=position == index,
                threshold=float(threshold), source_file=info["source_file"],
                source_sha256=dataset.digest, data_nature=dataset.data_nature,
                experiment_status=dataset.experiment_status,
                delta_roc_auc=None, delta_accuracy=None, delta_log_loss=None, **metrics,
            ))
        assert baseline_metrics is not None  # index 始终落在有效范围内
        for row in group_rows:
            rows.append(replace(
                row,
                delta_roc_auc=_difference(row.roc_auc, baseline_metrics["roc_auc"]),
                delta_accuracy=_difference(row.accuracy, baseline_metrics["accuracy"]),
                delta_log_loss=_difference(row.log_loss, baseline_metrics["log_loss"]),
            ))

    mock_labels = [dataset.label for dataset in datasets if dataset.is_mock]
    real_labels = [dataset.label for dataset in datasets if dataset.is_mock is False]
    scope_caveats = [SCOPE_NOTE]
    if len(mock_labels) == len(datasets):
        scope_caveats.append(
            f"本次参与对比的 {len(datasets)} 个模型全部为模拟模型"
            f"（{mock_labels}），结果不能证明任何模型优于基线。"
        )
    elif mock_labels:
        scope_caveats.append(
            f"混合对比：模拟模型 {mock_labels} 与声明为非模拟的模型 {real_labels} 同时存在，"
            "结论需人工复核训练与评估过程。"
        )

    ranked = [row for row in rows if row.group == GROUP_ALL]
    return {
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": utc_now_iso(),
        "configuration": {
            "threshold": float(threshold),
            "include_cold_start": bool(include_cold_start),
            "missing_policy": missing_policy,
        },
        "baseline": {"index": index, "label": names[index]},
        "models": list(described.values()),
        "coverage": coverage,
        "run_summaries": run_summaries,
        "aligned_sample_count": int(len(ids)),
        "union_sample_count": int(len(union)),
        "common_record_id_samples": list(ids[:5]),
        "rows": [row.as_record() for row in rows],
        "ranking_by_auc": [
            {"label": row.label, "roc_auc": row.roc_auc, "auc_status": row.auc_status}
            for row in sorted(
                ranked, key=lambda item: (item.roc_auc is None, -(item.roc_auc or 0.0))
            )
        ],
        "notes": notes,
        "scope_caveats": scope_caveats,
    }


def _difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def build_comparison_markdown(result: dict) -> str:
    """把对比结果渲染为 Markdown 汇总。"""
    baseline = result["baseline"]["label"]
    lines = [
        "## 多模型对比测试",
        "",
        f"- 基准模型：**{baseline}**；对齐样本数：{result['aligned_sample_count']}",
        f"- 对齐策略：`{result['configuration']['missing_policy']}`；"
        f"分类阈值：{result['configuration']['threshold']}；"
        f"包含冷启动：{'是' if result['configuration']['include_cold_start'] else '否'}",
        "",
        "| 样本分组 | 模型 | 模拟模型 | 样本数 | ROC-AUC | Accuracy | Log Loss | ΔAUC | ΔAccuracy | ΔLogLoss |",
        "| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['group']} | {row['label']}{'（基准）' if row['is_baseline'] else ''} | "
            f"{'是' if row['is_mock'] else '否'} | {row['sample_count']} | "
            f"{_fmt(row['roc_auc'])} | {_fmt(row['accuracy'])} | {_fmt(row['log_loss'], 4)} | "
            f"{_fmt(row['delta_roc_auc'], 4, signed=True)} | "
            f"{_fmt(row['delta_accuracy'], 4, signed=True)} | "
            f"{_fmt(row['delta_log_loss'], 4, signed=True)} |"
        )
    lines += ["", "### 对齐与核对说明", ""]
    lines.extend(f"- {note}" for note in result["notes"])
    lines += ["", "### 结论边界", ""]
    lines.extend(f"> {caveat}" for caveat in result["scope_caveats"])
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "不适用"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def run_comparison(
    prediction_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    labels: Sequence[str] | None = None,
    baseline: int | str | None = None,
    threshold: float = 0.5,
    include_cold_start: bool = True,
    missing_policy: str = "strict",
    overwrite: bool = False,
    relative_to: Path | None = None,
) -> dict:
    """执行模型对比并写出 CSV / JSON / Markdown 三种产物。

    先完成对齐与计算再创建输出目录：对齐失败时必须不留下任何产物目录。
    """
    result = compare_models(
        prediction_paths, labels=labels, baseline=baseline, threshold=threshold,
        include_cold_start=include_cold_start, missing_policy=missing_policy,
        relative_to=relative_to,
    )
    comparison_dir = ensure_output_dir(
        Path(output_dir).expanduser().resolve() / "comparison", overwrite=overwrite
    )
    write_table(comparison_dir / "model_comparison.csv", result["rows"], COMPARISON_COLUMNS)
    write_document(comparison_dir / "model_comparison.json", result)
    write_text(comparison_dir / "model_comparison.md", build_comparison_markdown(result))
    return result


__all__ = [
    "COMPARISON_COLUMNS",
    "ComparisonRow",
    "MISSING_POLICIES",
    "build_comparison_markdown",
    "compare_models",
    "run_comparison",
]
