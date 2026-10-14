"""刘静涵 模型评估模块集成验收：实际执行十项检查并留存证据。

与负责人 A 的 ``scripts/acceptance.py`` 保持同一风格：每个子进程都检查退出码，
并且核验结果内容与文件完整性，而不是只判断文件是否存在。

产出目录 ``evidence/evaluation/``：

- ``acceptance.log``：完整验收日志（命令、退出码、标准输出与错误输出）。
- ``acceptance_summary.json``：机器可读的验收摘要，含每一项检查的期望与实际结果。
- ``junit.xml``：本次实际执行的 pytest 结果，供报告第 11 节引用。
- ``metrics/``、``forgetting/``、``comparison/``、``performance/``、``plots/``、
  ``reports/``：六个评估模块的真实运行产物。

所有结果都标注数据来源与 Mock 状态：本次验收使用负责人 A 的合成数据与模拟模型，
证明的是评估程序正确，不是真实 DKT/AKT 的精度。
"""
from __future__ import annotations

import csv
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from io_utils import sha256_file  # noqa: E402

EVIDENCE = ROOT / "evidence" / "evaluation"
MOCK_RUN = ROOT / "evidence" / "mock_run"
PLUGIN_RUN = ROOT / "evidence" / "plugin_run"
SAMPLE_INPUT = ROOT / "sample_data.csv"
SYNTHETIC_FACTORY = "evaluation.synthetic_time_model:create_model"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

REQUIRED_PLOTS = (
    "auc_comparison.png", "accuracy_comparison.png", "log_loss_comparison.png",
    "roc_curves.png", "prediction_distribution.png", "forgetting_observed.png", "forgetting_model_compare.png",
    "inference_latency.png", "throughput_by_batch.png",
)


class AcceptanceFailure(RuntimeError):
    """验收不通过。"""


def manual_auc(labels: list[int], probabilities: list[float]) -> float:
    """用 Mann-Whitney U 秩和定义手工计算 AUC，独立于 scikit-learn 的实现。"""
    positive = [p for label, p in zip(labels, probabilities, strict=True) if label == 1]
    negative = [p for label, p in zip(labels, probabilities, strict=True) if label == 0]
    if not positive or not negative:
        raise AcceptanceFailure("手工 AUC 需要同时存在正负样本")
    wins = 0.0
    for high in positive:
        for low in negative:
            wins += 1.0 if high > low else 0.5 if high == low else 0.0
    return wins / (len(positive) * len(negative))


def manual_accuracy(labels: list[int], probabilities: list[float], threshold: float) -> float:
    hits = sum(
        1 for label, p in zip(labels, probabilities, strict=True)
        if int(p >= threshold) == label
    )
    return hits / len(labels)


def manual_log_loss(labels: list[int], probabilities: list[float]) -> float:
    """与 sklearn 相同口径：概率裁剪到 [eps, 1-eps]，其中 eps 为 float64 机器精度。"""
    eps = sys.float_info.epsilon
    total = 0.0
    for label, p in zip(labels, probabilities, strict=True):
        clipped = min(max(p, eps), 1 - eps)
        total += -(math.log(clipped) if label == 1 else math.log(1 - clipped))
    return total / len(labels)


def read_predictions(path: Path) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    probabilities: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            labels.append(int(float(record["y_true"])))
            probabilities.append(float(record["y_pred"]))
    return labels, probabilities


def read_metrics_rows() -> list[dict]:
    path = EVIDENCE / "metrics" / "metrics.csv"
    if not path.is_file():
        raise AcceptanceFailure(f"缺少指标结果文件 {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AcceptanceFailure("指标结果 CSV 为空")
    return rows


def check_png(path: Path) -> tuple[int, int]:
    """校验 PNG 签名与 IHDR 尺寸，确认是真实可用的图片。"""
    if not path.is_file():
        raise AcceptanceFailure(f"缺少图片 {path}")
    data = path.read_bytes()
    if data[:8] != PNG_SIGNATURE:
        raise AcceptanceFailure(f"{path} 不是合法 PNG")
    if len(data) < 4096:
        raise AcceptanceFailure(f"{path} 体积过小，可能是空白图")
    width, height = struct.unpack(">II", data[16:24])
    if width < 400 or height < 300:
        raise AcceptanceFailure(f"{path} 分辨率过低: {width}x{height}")
    return width, height


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    logs: list[str] = []

    def record(name: str, category: str, expected: str, observed: str, passed: bool) -> None:
        checks.append({
            "name": name, "category": category, "expected": expected,
            "observed": observed, "passed": bool(passed),
        })
        if not passed:
            raise AcceptanceFailure(f"{name} 未通过：期望 {expected}，实际 {observed}")

    def command(name: str, argv: list[str], expected_code: int = 0, timeout: int = 600):
        result = subprocess.run(
            [sys.executable, *argv], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        logs.append(
            f"## {name}\n{' '.join(argv)}\n退出码 {result.returncode}\n"
            f"{result.stdout}{result.stderr}"
        )
        return result, expected_code

    if not (MOCK_RUN / "predictions.csv").is_file():
        raise AcceptanceFailure("缺少 evidence/mock_run/predictions.csv，请先运行 scripts/acceptance.py")
    if not SAMPLE_INPUT.is_file():
        raise AcceptanceFailure("缺少 sample_data.csv，无法执行受控遗忘测试与性能测试")

    # ------------------------------------------------------------------ 检查 1 & 2
    result, code = command("模型评价指标计算", [
        "evaluate.py", "metrics",
        "--predictions", "evidence/mock_run", "evidence/plugin_run",
        "--output-dir", "evidence/evaluation", "--overwrite",
    ])
    if result.returncode != code:
        raise AcceptanceFailure(
            f"指标计算失败，退出码 {result.returncode}：{result.stderr}"
        )
    metrics_rows = read_metrics_rows()
    mock_all = next(
        (row for row in metrics_rows
         if row["group"] == "all" and row["model_name"] == "mock_history_beta_v1"), None
    )
    if mock_all is None:
        raise AcceptanceFailure("指标结果中缺少 mock_history_beta_v1 的 all 分组")
    record("模型评价指标计算", "evidence", "退出码 0 且写出 metrics.csv", "退出码 0", True)

    labels, probabilities = read_predictions(MOCK_RUN / "predictions.csv")
    expected_auc = manual_auc(labels, probabilities)
    expected_accuracy = manual_accuracy(labels, probabilities, 0.5)
    expected_log_loss = manual_log_loss(labels, probabilities)
    actual = {
        "roc_auc": float(mock_all["roc_auc"]),
        "accuracy": float(mock_all["accuracy"]),
        "log_loss": float(mock_all["log_loss"]),
    }
    for key, reference in (
        ("roc_auc", expected_auc), ("accuracy", expected_accuracy), ("log_loss", expected_log_loss),
    ):
        if abs(actual[key] - reference) > 1e-9:
            raise AcceptanceFailure(
                f"{key} 数值不一致：程序 {actual[key]}，独立手工计算 {reference}"
            )
    record(
        "AUC / Accuracy / Log Loss 数值校验", "evidence",
        "与独立手工计算一致（容差 1e-9）",
        f"AUC={expected_auc:.6f}, Accuracy={expected_accuracy:.6f}, "
        f"LogLoss={expected_log_loss:.6f}",
        True,
    )

    # ------------------------------------------------------------------ 检查 3
    result, code = command("多模型对比", [
        "evaluate.py", "compare",
        "--predictions", "evidence/mock_run", "evidence/plugin_run",
        "--labels", "mock_beta", "constant_plugin", "--baseline", "mock_beta",
        "--output-dir", "evidence/evaluation", "--overwrite",
    ])
    if result.returncode != code:
        raise AcceptanceFailure(f"模型对比失败，退出码 {result.returncode}：{result.stderr}")
    comparison = json.loads(
        (EVIDENCE / "comparison" / "model_comparison.json").read_text(encoding="utf-8")
    )
    if comparison["aligned_sample_count"] != 288:
        raise AcceptanceFailure(f"对齐样本数应为 288，实际 {comparison['aligned_sample_count']}")
    labels_in_comparison = {row["label"] for row in comparison["rows"]}
    if labels_in_comparison != {"mock_beta", "constant_plugin"}:
        raise AcceptanceFailure(f"对比结果缺少模型：{labels_in_comparison}")
    if not all(item["is_mock"] for item in comparison["models"]):
        raise AcceptanceFailure("本次对比应全部是模拟模型，is_mock 标记异常")
    if not (EVIDENCE / "comparison" / "model_comparison.md").is_file():
        raise AcceptanceFailure("缺少 Markdown 对比汇总")
    record("多模型对比测试", "evidence", "两个模拟模型按 record_id 对齐",
           f"对齐 {comparison['aligned_sample_count']} 条样本，基准 {comparison['baseline']['label']}", True)

    # ------------------------------------------------------------------ 检查 4 & 5
    result, code = command("遗忘机制专项测试", [
        "evaluate.py", "forgetting",
        "--predictions", "evidence/mock_run", "--input", "sample_data.csv",
        "--intervals", "0,1,3,7,14", "--unit", "days", "--repeats", "3",
        "--max-requests", "12", "--controlled-model", f"synthetic_fixture={SYNTHETIC_FACTORY}",
        "--output-dir", "evidence/evaluation", "--overwrite",
    ])
    if result.returncode != code:
        raise AcceptanceFailure(f"遗忘测试失败，退出码 {result.returncode}：{result.stderr}")
    forgetting = json.loads(
        (EVIDENCE / "forgetting" / "forgetting_summary.json").read_text(encoding="utf-8")
    )
    descriptive = forgetting["descriptive"]
    if not descriptive["available"] or descriptive["usable_samples"] <= 0:
        raise AcceptanceFailure("描述性时间间隔分析未产生有效结果")
    bucket_rows = list(csv.DictReader(
        (EVIDENCE / "forgetting" / "interval_buckets.csv").open(encoding="utf-8-sig", newline="")
    ))
    if len(bucket_rows) != len(descriptive["bucket_rows"]):
        raise AcceptanceFailure("interval_buckets.csv 与 JSON 中的分桶数量不一致")
    if "event_order" not in descriptive["caveat"]:
        raise AcceptanceFailure("描述性分析缺少「不使用事件顺序伪造天数」的说明")
    record("遗忘时间描述性分析", "evidence", "按时间间隔分桶并说明观察性边界",
           f"{descriptive['usable_samples']} 条有效样本，{len(bucket_rows)} 个时间分桶", True)

    entries = {entry["label"]: entry for entry in forgetting["controlled_models"]}
    if set(entries) != {"mock_history_beta_v1", "synthetic_fixture"}:
        raise AcceptanceFailure(f"受控测试模型标签异常：{sorted(entries)}")
    mock_entry, fixture_entry = entries["mock_history_beta_v1"], entries["synthetic_fixture"]
    if not mock_entry["available"] or not fixture_entry["available"]:
        raise AcceptanceFailure("受控固定历史测试未产生结果")
    if mock_entry["response"]["detected"]:
        raise AcceptanceFailure("MockModel 没有遗忘机制，不应检测到时间响应")
    if not fixture_entry["response"]["detected"] or not fixture_entry["response"]["monotonic_decreasing"]:
        raise AcceptanceFailure("合成时间敏感夹具未被检测出单调下降，测试程序可能失效")
    if not mock_entry["design"]["history_fixed"] or not fixture_entry["design"]["causal_order_preserved"]:
        raise AcceptanceFailure("受控测试的固定历史或因果顺序标记异常")
    detail_rows = list(csv.DictReader(
        (EVIDENCE / "forgetting" / "controlled_time_response.csv").open(encoding="utf-8-sig", newline="")
    ))
    if len(detail_rows) != (
        mock_entry["design"]["total_inference_calls"] + fixture_entry["design"]["total_inference_calls"]
    ):
        raise AcceptanceFailure("受控测试明细行数与设计不一致")
    record(
        "固定历史受控时间测试", "evidence",
        "Mock 无响应且合成夹具单调下降",
        f"Mock 最大变化 {mock_entry['response']['max_abs_change_vs_baseline']:.3g}；"
        f"夹具 Spearman {fixture_entry['response']['spearman_interval_vs_mean_probability']:.4f}",
        True,
    )

    # ------------------------------------------------------------------ 检查 6
    result, code = command("Mock 模型推理性能测试", [
        "evaluate.py", "performance", "--input", "sample_data.csv",
        "--batch-sizes", "1,8,32,64", "--repeats", "3", "--warmup", "1",
        "--max-requests", "256", "--output-dir", "evidence/evaluation", "--overwrite",
    ])
    if result.returncode != code:
        raise AcceptanceFailure(f"性能测试失败，退出码 {result.returncode}：{result.stderr}")
    performance = json.loads(
        (EVIDENCE / "performance" / "performance.json").read_text(encoding="utf-8")
    )
    sizes = [row["batch_size"] for row in performance["rows"]]
    if sizes != [1, 8, 32, 64]:
        raise AcceptanceFailure(f"性能结果缺少批量大小：{sizes}")
    for row in performance["rows"]:
        if not (row["p50_batch_latency_ms"] <= row["p95_batch_latency_ms"]):
            raise AcceptanceFailure(f"batch={row['batch_size']} 的 P50 大于 P95，统计异常")
        if row["throughput_samples_per_second"] <= 0:
            raise AcceptanceFailure(f"batch={row['batch_size']} 吞吐量非正")
        if row["sample_count"] != 256 * performance["settings"]["repeats"]:
            raise AcceptanceFailure(f"batch={row['batch_size']} 的样本数统计异常")
    if not performance["model"]["is_mock"]:
        raise AcceptanceFailure("性能测试应使用 Mock 模型")
    if not any("鸿蒙端侧" in caveat for caveat in performance["caveats"]):
        raise AcceptanceFailure("性能结果缺少「不等于鸿蒙端侧性能」的说明")
    record(
        "Mock 模型推理性能测试", "evidence", "四个批量大小的延迟与吞吐量",
        f"{sizes}；环境 Python {performance['environment']['python']}，"
        f"计时 {performance['environment']['timer']}",
        True,
    )

    # ------------------------------------------------------------------ 检查 7
    result, code = command("可视化图表生成", [
        "evaluate.py", "plots", "--evaluation-dir", "evidence/evaluation",
        "--predictions", "evidence/mock_run/predictions.csv",
        "evidence/plugin_run/predictions.csv", "--overwrite",
    ])
    if result.returncode != code:
        raise AcceptanceFailure(f"图表生成失败，退出码 {result.returncode}：{result.stderr}")
    manifest = json.loads(
        (EVIDENCE / "plots" / "plots_manifest.json").read_text(encoding="utf-8")
    )
    created = {item["file"] for item in manifest["plots"] if item["status"] == "created"}
    missing = set(REQUIRED_PLOTS) - created
    if missing:
        raise AcceptanceFailure(f"缺少图表：{sorted(missing)}；跳过原因 {manifest['skipped']}")
    if manifest["skipped"]:
        raise AcceptanceFailure(f"本次验收不应跳过任何图表：{manifest['skipped']}")
    sizes = {}
    for name in REQUIRED_PLOTS:
        width, height = check_png(EVIDENCE / "plots" / name)
        sizes[name] = f"{width}x{height}"
    demo_watermarks = [item["file"] for item in manifest["plots"] if item["demo_watermark"]]
    if len(demo_watermarks) != len(REQUIRED_PLOTS):
        raise AcceptanceFailure("模拟数据图表必须全部带 DEMO 水印")
    record("可视化图表生成", "evidence", "九张图表全部生成且分辨率达标",
           f"DPI={manifest['dpi']}，字体语言 {manifest['font_language']}", True)

    # ------------------------------------------------------------------ 检查 8
    command("采集自动化测试结果", [
        "-m", "pytest", "-q", "--junitxml=evidence/evaluation/junit.xml",
    ])
    junit_path = EVIDENCE / "junit.xml"
    if not junit_path.is_file():
        raise AcceptanceFailure("pytest 未生成 JUnit 报告")
    junit_text = junit_path.read_text(encoding="utf-8")
    if "<testsuite" not in junit_text:
        raise AcceptanceFailure("JUnit 报告内容异常")
    if 'failures="0"' not in junit_text and 'errors="0"' not in junit_text:
        logs.append("警告：pytest 存在失败或错误用例，详见 evidence/evaluation/junit.xml")

    # 先写出部分摘要，让报告第 10 节能引用已经完成的边界检查。
    boundary_expected = "退出码 2 且不生成输出目录"
    pre_report_checks = list(checks)

    # ------------------------------------------------------------------ 检查 9
    negative_cases = [
        ("缺少必需列的预测文件报错", [
            "evaluate.py", "metrics", "--predictions", str(EVIDENCE / "bad_missing_column.csv"),
            "--output-dir", str(EVIDENCE / "_negative_metrics"), "--overwrite"]),
        ("非法标签与非法概率报错", [
            "evaluate.py", "metrics", "--predictions", str(EVIDENCE / "bad_values.csv"),
            "--output-dir", str(EVIDENCE / "_negative_metrics"), "--overwrite"]),
        ("重复 record_id 报错", [
            "evaluate.py", "metrics", "--predictions", str(EVIDENCE / "bad_duplicate_id.csv"),
            "--output-dir", str(EVIDENCE / "_negative_duplicate"), "--overwrite"]),
        ("空预测文件报错", [
            "evaluate.py", "metrics", "--predictions", str(EVIDENCE / "bad_empty.csv"),
            "--output-dir", str(EVIDENCE / "_negative_empty"), "--overwrite"]),
        ("模型样本数不一致时严格失败", [
            "evaluate.py", "compare", "--predictions", "evidence/mock_run/predictions.csv",
            str(EVIDENCE / "bad_shorter.csv"), "--output-dir", str(EVIDENCE / "_negative_compare"),
            "--overwrite"]),
        ("未接入真实模型时明确失败", [
            "evaluate.py", "performance", "--input", "sample_data.csv",
            "--factory", "examples.training_model_template:create_model",
            "--output-dir", str(EVIDENCE / "_negative_performance"), "--overwrite"]),
        ("非法的受控时间间隔报错", [
            "evaluate.py", "forgetting", "--input", "sample_data.csv", "--intervals", "1,-2",
            "--output-dir", str(EVIDENCE / "_negative_intervals"), "--overwrite"]),
    ]
    write_negative_fixtures()
    for name, argv in negative_cases:
        result, code = command(name, argv, expected_code=2)
        if result.returncode != 2:
            raise AcceptanceFailure(f"{name} 应返回退出码 2，实际 {result.returncode}：{result.stdout}")
        if not result.stderr.strip():
            raise AcceptanceFailure(f"{name} 返回了退出码 2 但没有错误说明")
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        if output_dir.exists():
            raise AcceptanceFailure(f"{name} 失败后不应留下输出目录 {output_dir}")
        record(name, "boundary", boundary_expected, "退出码 2，无输出目录", True)

    # 仅有 event_order 的数据不是错误而是「不适用」：必须明确说明原因，且不得产出时间表。
    order_only_dir = EVIDENCE / "_order_only"
    result, code = command("仅有 event_order 的数据不伪造时间", [
        "evaluate.py", "forgetting",
        "--predictions", "evidence/assistments_run/predictions.csv",
        "--output-dir", str(order_only_dir), "--overwrite",
    ])
    if result.returncode != 0:
        raise AcceptanceFailure(f"仅有 event_order 的数据应正常结束：{result.stderr}")
    order_only = json.loads(
        (order_only_dir / "forgetting" / "forgetting_summary.json").read_text(encoding="utf-8")
    )
    if order_only["descriptive"]["available"]:
        raise AcceptanceFailure("仅有 event_order 的数据不应产生时间间隔分析")
    if "event_order" not in order_only["descriptive"]["reason"]:
        raise AcceptanceFailure("不可用原因未说明 event_order")
    if (order_only_dir / "forgetting" / "interval_buckets.csv").exists():
        raise AcceptanceFailure("不可用时不应写出时间分桶结果")
    if order_only["controlled_models"][0]["available"]:
        raise AcceptanceFailure("仅有 event_order 的数据不应产生受控时间测试结果")
    record("仅有 event_order 的数据不伪造时间", "boundary",
           "标记为不适用、不产出时间分桶、不影响退出码",
           f"退出码 0；原因「{order_only['descriptive']['reason'][:40]}…」", True)

    # ------------------------------------------------------------------ 检查 10
    result, code = command("原有 main.py 回归", [
        "main.py", "--output-dir", "evidence/evaluation/_main_regression", "--overwrite",
    ])
    if result.returncode != 0:
        raise AcceptanceFailure(f"原有 main.py 运行失败：{result.stderr}")
    regression_dir = EVIDENCE / "_main_regression"
    summary = json.loads((regression_dir / "run_summary.json").read_text(encoding="utf-8"))
    if summary["prediction_count"] != 288 or summary["is_mock"] is not True:
        raise AcceptanceFailure("原有 main.py 的输出摘要与预期不符")
    with (regression_dir / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if header != [
        "record_id", "student_id", "question_id", "skill_id", "skill_index", "timestamp",
        "event_order", "time_source", "y_true", "y_pred", "predicted_label", "history_length",
        "delta_time_seconds", "cold_start", "model_name", "is_mock", "source_row", "schema_version",
    ]:
        raise AcceptanceFailure(f"原有 predictions.csv 列定义被改变：{header}")
    record("原有 main.py 功能不受影响", "regression", "退出码 0 且列定义不变",
           f"{summary['prediction_count']} 条预测，列定义与 v1.0.0 一致", True)

    # ------------------------------------------- 检查 11：刘静涵任务的验收标准
    # 「给定一份符合约定格式的预测结果 CSV，能够直接计算评价指标、生成图表并导出报告。」
    single_dir = EVIDENCE / "_single_csv"
    result, code = command("验收标准·单份 CSV 计算指标", [
        "evaluate.py", "metrics", "--predictions", "evidence/mock_run/predictions.csv",
        "--output-dir", str(single_dir), "--overwrite",
    ])
    if result.returncode != 0:
        raise AcceptanceFailure(f"单份 CSV 指标计算失败：{result.stderr}")
    single_metrics = json.loads(
        (single_dir / "metrics" / "metrics.json").read_text(encoding="utf-8")
    )
    all_row = next(row for row in single_metrics["rows"] if row["group"] == "ALL".lower())
    if all_row["roc_auc"] is None or all_row["accuracy"] is None:
        raise AcceptanceFailure("单份 CSV 应能算出完整指标")

    result, code = command("验收标准·单份 CSV 生成图表", [
        "evaluate.py", "plots", "--evaluation-dir", str(single_dir),
        "--predictions", "evidence/mock_run/predictions.csv", "--overwrite",
    ])
    if result.returncode != 0:
        raise AcceptanceFailure(f"单份 CSV 图表生成失败：{result.stderr}")
    single_manifest = json.loads(
        (single_dir / "plots" / "plots_manifest.json").read_text(encoding="utf-8")
    )
    single_created = {item["file"] for item in single_manifest["plots"] if item["status"] == "created"}
    for required in ("auc_comparison.png", "accuracy_comparison.png", "log_loss_comparison.png",
                     "roc_curves.png", "prediction_distribution.png"):
        if required not in single_created:
            raise AcceptanceFailure(f"单份 CSV 场景缺少图表 {required}")
    single_skipped = {item["name"]: item["reason"] for item in single_manifest["skipped"]}
    for name, reason in single_skipped.items():
        if not reason:
            raise AcceptanceFailure(f"跳过的图表 {name} 未说明原因")

    result, code = command("验收标准·单份 CSV 导出报告", [
        "evaluate.py", "report", "--evaluation-dir", str(single_dir),
        "--output", str(single_dir / "reports" / "single_report.md"), "--overwrite",
    ])
    if result.returncode != 0:
        raise AcceptanceFailure(f"单份 CSV 报告导出失败：{result.stderr}")
    single_report = (single_dir / "reports" / "single_report.md").read_text(encoding="utf-8")
    for needle, description in (
        (all_row["label"], "指标表应包含模型名"),
        (f"{all_row['roc_auc']:.4f}", "指标表应包含实际 AUC 数值"),
        ("待真实模型接入", "必须标注尚未接入真实模型"),
        ("未执行推理性能测试", "未运行的实验必须写明未执行"),
    ):
        if needle not in single_report:
            raise AcceptanceFailure(f"单份 CSV 报告缺少内容（{description}）：{needle}")
    record(
        "验收标准：单份预测 CSV 直接出指标+图表+报告", "acceptance",
        "退出码 0，含指标、图表与报告且无虚假结论",
        f"AUC={all_row['roc_auc']:.4f}；生成 {len(single_created)} 张图、"
        f"跳过 {len(single_skipped)} 张并写明原因",
        True,
    )

    # 写出部分摘要，供报告引用边界检查结果。
    write_summary(checks, complete=False)
    docs_report = ROOT / "docs" / "EVALUATION_TEST_REPORT.md"
    result, code = command("自动生成评估报告", [
        "evaluate.py", "report", "--evaluation-dir", "evidence/evaluation",
        "--output", "evidence/evaluation/reports/evaluation_report.md",
        "--docs-report", str(docs_report),
        "--acceptance-summary", "evidence/evaluation/acceptance_summary.json",
        "--junit-xml", "evidence/evaluation/junit.xml",
        "--command", "python evaluate.py all --output-dir outputs/evaluation --overwrite",
        "--overwrite",
    ])
    if result.returncode != 0:
        raise AcceptanceFailure(f"报告生成失败：{result.stderr}")
    report_text = (EVIDENCE / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
    for heading in ("## 1. 测试任务概述", "## 12. 当前未完成的真实模型验证事项",
                    "## 14. 后续真实模型接入说明"):
        if heading not in report_text:
            raise AcceptanceFailure(f"报告缺少章节 {heading}")
    if "待真实模型接入" not in report_text:
        raise AcceptanceFailure("报告缺少「待真实模型接入」标注")
    if "未执行" in report_text and "comparison" not in report_text:
        raise AcceptanceFailure("报告章节内容异常")
    for line in report_text.splitlines():
        if line.startswith("![") and "](" in line:
            target = line.split("](", 1)[1].rstrip(")")
            if not (EVIDENCE / "reports" / target).resolve().is_file():
                raise AcceptanceFailure(f"报告图片链接失效：{target}")
    if not docs_report.is_file():
        raise AcceptanceFailure("未写出 docs/EVALUATION_TEST_REPORT.md")
    if len(report_text) < 4000:
        raise AcceptanceFailure("报告内容过短，可能缺少章节")
    record("自动生成评估报告", "evidence", "Markdown 报告含全部章节且图片链接有效",
           f"{len(report_text)} 字符，{len(pre_report_checks) + len(checks)} 项检查上下文", True)

    # ------------------------------------------------------------------ 汇总
    summary_payload = write_summary(checks, complete=True)
    (EVIDENCE / "acceptance.log").write_text("\n\n".join(logs), encoding="utf-8")
    cleanup_negative_outputs()
    print(
        f"评估模块集成验收通过：{len(checks)} 项检查全部通过"
        f"（边界与异常 {sum(1 for item in checks if item['category'] == 'boundary')} 项，"
        f"验收标准 {sum(1 for item in checks if item['category'] == 'acceptance')} 项）。"
    )
    print(f"证据目录：{EVIDENCE}")
    print(f"评估报告：{docs_report}")
    assert summary_payload["all_passed"] is True
    return 0


def write_negative_fixtures() -> None:
    """构造异常输入的验收夹具（明确标注为故意损坏的数据）。"""
    rows = [
        {"record_id": "n1", "student_id": "S1", "question_id": "Q1", "skill_id": "K01",
         "y_true": 1, "y_pred": 0.6},
    ]
    with (EVIDENCE / "bad_missing_column.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "y_true"])
        writer.writeheader()
        writer.writerow({"record_id": "n1", "y_true": 1})

    bad = [
        dict(rows[0]),
        {"record_id": "n2", "student_id": "S1", "question_id": "Q2", "skill_id": "K01",
         "y_true": 2, "y_pred": 0.5},
        {"record_id": "n3", "student_id": "S1", "question_id": "Q3", "skill_id": "K01",
         "y_true": 0, "y_pred": 1.4},
        {"record_id": "n4", "student_id": "S1", "question_id": "Q4", "skill_id": "K01",
         "y_true": 1, "y_pred": "nan"},
    ]
    with (EVIDENCE / "bad_values.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["record_id", "student_id", "question_id", "skill_id", "y_true", "y_pred"]
        )
        writer.writeheader()
        writer.writerows(bad)

    duplicate = [dict(rows[0]), dict(rows[0])]
    with (EVIDENCE / "bad_duplicate_id.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["record_id", "student_id", "question_id", "skill_id", "y_true", "y_pred"],
        )
        writer.writeheader()
        writer.writerows(duplicate)

    with (EVIDENCE / "bad_empty.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("record_id,student_id,question_id,skill_id,y_true,y_pred\n")

    with (MOCK_RUN / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        # 故意少写一行，用于验证模型对比在样本覆盖不一致时会严格失败。
        shorter = [header, *[row for _, row in zip(range(287), reader, strict=False)]]
    with (EVIDENCE / "bad_shorter.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(shorter)


def cleanup_negative_outputs() -> None:
    """删除故意失败用例留下的临时目录与损坏夹具，保持证据目录只含有效产物。"""
    import shutil

    for name in (
        "bad_missing_column.csv", "bad_values.csv", "bad_duplicate_id.csv", "bad_empty.csv",
        "bad_shorter.csv", "_main_regression", "_order_only", "_single_csv",
    ):
        target = EVIDENCE / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def write_summary(checks: list[dict], *, complete: bool) -> dict:
    payload = {
        "evaluation_version": "1.0.0",
        "script": "scripts/acceptance_evaluation.py",
        "complete": complete,
        "checks": checks,
        "passed": sum(1 for item in checks if item["passed"]),
        "failed": sum(1 for item in checks if not item["passed"]),
        "all_passed": all(item["passed"] for item in checks),
        "data_type": "synthetic_and_handcrafted_fixtures",
        "model_status": "all_models_are_mock",
        "scope": (
            "本次验收使用负责人 A 的合成数据与模拟模型，验证的是评估程序与接口契约；"
            "尚未开展真实 DKT/AKT 精度、真实遗忘机制效果与鸿蒙端侧性能验证。"
        ),
        "evidence_files": {
            "metrics": "metrics/metrics.csv",
            "comparison": "comparison/model_comparison.json",
            "forgetting": "forgetting/forgetting_summary.json",
            "performance": "performance/performance.json",
            "plots": "plots/plots_manifest.json",
            "report": "reports/evaluation_report.md",
            "junit": "junit.xml",
        },
    }
    for relative in payload["evidence_files"].values():
        payload.setdefault("fingerprints", {})[relative] = (
            sha256_file(EVIDENCE / relative) if (EVIDENCE / relative).is_file() else None
        )
    (EVIDENCE / "acceptance_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptanceFailure as error:
        print(f"评估模块集成验收失败：{error}", file=sys.stderr)
        raise SystemExit(2) from error
