"""evaluate.py 命令行接口测试：子命令、退出码与端到端集成。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from evaluate import main

ROOT = Path(__file__).resolve().parents[1]
MOCK_RUN = ROOT / "evidence" / "mock_run"
PLUGIN_RUN = ROOT / "evidence" / "plugin_run"
SAMPLE_INPUT = ROOT / "sample_data.csv"

requires_evidence = pytest.mark.skipif(
    not (MOCK_RUN / "predictions.csv").is_file() or not SAMPLE_INPUT.is_file(),
    reason="需要负责人 A 的 evidence/mock_run 与 sample_data.csv",
)


def run_argv(argv):
    return subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"), *argv],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )


# ------------------------------------------------------------------ --help 系列


@pytest.mark.parametrize("argv", [
    ["--help"], ["metrics", "--help"], ["forgetting", "--help"], ["compare", "--help"],
    ["performance", "--help"], ["plots", "--help"], ["report", "--help"], ["all", "--help"],
])
def test_help_is_available_for_every_subcommand(argv):
    result = run_argv(argv)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_version_flag():
    result = run_argv(["--version"])
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


def test_missing_subcommand_is_an_argument_error():
    assert run_argv([]).returncode == 2


# ------------------------------------------------------------------ 子命令行为


@requires_evidence
def test_metrics_subcommand_writes_results(tmp_path):
    output = tmp_path / "eval"
    assert main(["metrics", "--predictions", str(MOCK_RUN),
                 "--output-dir", str(output)]) == 0
    payload = json.loads((output / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["inputs"][0]["sample_count"] == 288
    assert payload["configuration"]["include_cold_start"] is True


@requires_evidence
def test_metrics_accepts_multiple_sources_and_threshold(tmp_path):
    output = tmp_path / "eval"
    assert main(["metrics", "--predictions", str(MOCK_RUN), str(PLUGIN_RUN),
                 "--threshold", "0.4", "--exclude-cold-start",
                 "--output-dir", str(output)]) == 0
    payload = json.loads((output / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    assert len(payload["inputs"]) == 2
    assert payload["configuration"]["threshold"] == 0.4
    assert payload["configuration"]["include_cold_start"] is False
    all_row = next(row for row in payload["rows"]
                   if row["group"] == "all" and row["label"] == payload["inputs"][0]["label"])
    assert all_row["sample_count"] == 276


def test_metrics_fails_on_missing_file(tmp_path):
    assert main(["metrics", "--predictions", str(tmp_path / "nope.csv"),
                 "--output-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


@requires_evidence
def test_compare_subcommand_with_labels_and_baseline(tmp_path):
    output = tmp_path / "eval"
    assert main(["compare", "--predictions", str(MOCK_RUN), str(PLUGIN_RUN),
                 "--labels", "beta", "constant", "--baseline", "constant",
                 "--output-dir", str(output)]) == 0
    payload = json.loads(
        (output / "comparison" / "model_comparison.json").read_text(encoding="utf-8")
    )
    assert payload["baseline"]["label"] == "constant"
    assert payload["aligned_sample_count"] == 288
    assert (output / "comparison" / "model_comparison.md").is_file()


@requires_evidence
def test_compare_strict_fails_on_mismatched_samples(tmp_path):
    short = tmp_path / "short.csv"
    frame = pd.read_csv(MOCK_RUN / "predictions.csv", dtype=str).iloc[:-1]
    frame.to_csv(short, index=False)
    assert main(["compare", "--predictions", str(MOCK_RUN / "predictions.csv"), str(short),
                 "--output-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


@requires_evidence
def test_forgetting_subcommand_both_parts(tmp_path):
    output = tmp_path / "eval"
    assert main(["forgetting", "--predictions", str(MOCK_RUN), "--input", str(SAMPLE_INPUT),
                 "--intervals", "0,1,7", "--unit", "days", "--repeats", "2",
                 "--max-requests", "4",
                 "--controlled-model",
                 "fixture=evaluation.synthetic_time_model:create_model",
                 "--output-dir", str(output)]) == 0
    payload = json.loads(
        (output / "forgetting" / "forgetting_summary.json").read_text(encoding="utf-8")
    )
    assert payload["descriptive"]["available"] is True
    entries = {entry["label"]: entry for entry in payload["controlled_models"]}
    assert "mock_history_beta_v1" in entries
    assert entries["fixture"]["response"]["detected"] is True
    assert entries["mock_history_beta_v1"]["response"]["detected"] is False
    assert (output / "forgetting" / "controlled_time_response.csv").is_file()


@requires_evidence
def test_forgetting_rejects_bad_intervals(tmp_path):
    assert main(["forgetting", "--predictions", str(MOCK_RUN), "--intervals", "1,-2",
                 "--output-dir", str(tmp_path / "out")]) == 2
    assert main(["forgetting", "--output-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    # 非法的 --unit 取值由 argparse 拒绝，同样以退出码 2 结束。
    with pytest.raises(SystemExit) as exit_info:
        main(["forgetting", "--predictions", str(MOCK_RUN), "--intervals", "1,2",
              "--unit", "weeks", "--output-dir", str(tmp_path / "out")])
    assert exit_info.value.code == 2


@requires_evidence
def test_forgetting_rejects_bad_controlled_model_spec(tmp_path):
    assert main(["forgetting", "--input", str(SAMPLE_INPUT),
                 "--controlled-model", "no-equals-sign",
                 "--output-dir", str(tmp_path / "out")]) == 2
    assert main(["forgetting", "--input", str(SAMPLE_INPUT),
                 "--controlled-model", "dup=mock_model:MockModel",
                 "--controlled-model", "dup=mock_model:MockModel",
                 "--output-dir", str(tmp_path / "out")]) == 2


@requires_evidence
def test_performance_subcommand(tmp_path):
    output = tmp_path / "eval"
    assert main(["performance", "--input", str(SAMPLE_INPUT),
                 "--batch-sizes", "1,8,32", "--repeats", "2", "--warmup", "1",
                 "--max-requests", "64", "--output-dir", str(output)]) == 0
    payload = json.loads(
        (output / "performance" / "performance.json").read_text(encoding="utf-8")
    )
    assert [row["batch_size"] for row in payload["rows"]] == [1, 8, 32]
    assert payload["settings"]["pool_size"] == 64
    assert payload["model"]["is_mock"] is True


def test_performance_rejects_invalid_batch_sizes(tmp_path):
    assert main(["performance", "--input", str(SAMPLE_INPUT), "--batch-sizes", "0",
                 "--output-dir", str(tmp_path / "out")]) == 2


def test_performance_plugin_factory_failure_returns_two(tmp_path):
    assert main(["performance", "--input", str(SAMPLE_INPUT),
                 "--factory", "examples.training_model_template:create_model",
                 "--output-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------ 端到端 all


@requires_evidence
def test_all_end_to_end_creates_everything(tmp_path):
    output = tmp_path / "evaluation"
    docs = tmp_path / "docs" / "EVALUATION_TEST_REPORT.md"
    assert main(["all", "--output-dir", str(output), "--run-dir", str(MOCK_RUN),
                 "--compare-runs", str(MOCK_RUN), str(PLUGIN_RUN),
                 "--input", str(SAMPLE_INPUT), "--max-requests", "4",
                 "--performance-max-requests", "32", "--batch-sizes", "1,8",
                 "--performance-repeats", "2", "--docs-report", str(docs)]) == 0

    expected = [
        "metrics/metrics.csv", "metrics/metrics.json",
        "forgetting/interval_buckets.csv", "forgetting/controlled_time_response.csv",
        "forgetting/forgetting_summary.json",
        "comparison/model_comparison.csv", "comparison/model_comparison.json",
        "comparison/model_comparison.md",
        "performance/performance.csv", "performance/performance.json",
        "plots/plots_manifest.json", "plots/auc_comparison.png", "plots/roc_curves.png",
        "reports/evaluation_report.md", "evaluation_summary.json",
    ]
    for relative in expected:
        assert (output / relative).is_file(), relative
    assert docs.is_file()

    summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert [step["step"] for step in summary["steps"]] == [
        "metrics", "forgetting", "compare", "performance", "plots", "report"
    ]
    assert all(step["status"] == "executed" for step in summary["steps"])
    assert summary["reproduce_commands"]
    report = (output / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
    assert "## 14. 后续真实模型接入说明" in report
    assert "DEMO" not in report or "合成" in report


@requires_evidence
def test_all_skips_steps_with_missing_inputs(tmp_path):
    output = tmp_path / "evaluation"
    assert main(["all", "--output-dir", str(output), "--run-dir", str(MOCK_RUN),
                 "--compare-runs", str(MOCK_RUN), str(PLUGIN_RUN),
                 "--input", str(tmp_path / "missing.csv")]) == 0
    summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    statuses = {step["step"]: step["status"] for step in summary["steps"]}
    assert statuses["forgetting"] == "skipped"
    assert statuses["performance"] == "skipped"
    assert statuses["metrics"] == "executed"
    assert statuses["report"] == "executed"
    # 跳过时必须在报告中说明，而不是静默生成结果。
    report = (output / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
    assert "未执行推理性能测试" in report


@requires_evidence
def test_all_skips_compare_with_single_model(tmp_path):
    output = tmp_path / "evaluation"
    assert main(["all", "--output-dir", str(output), "--run-dir", str(MOCK_RUN),
                 "--compare-runs", str(MOCK_RUN), "--input", str(SAMPLE_INPUT),
                 "--max-requests", "4", "--performance-max-requests", "8",
                 "--batch-sizes", "1", "--performance-repeats", "1"]) == 0
    summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    statuses = {step["step"]: step["status"] for step in summary["steps"]}
    assert statuses["compare"] == "skipped"


@requires_evidence
def test_all_refuses_silent_overwrite(tmp_path):
    output = tmp_path / "evaluation"
    argv = ["all", "--output-dir", str(output), "--run-dir", str(MOCK_RUN),
            "--compare-runs", str(MOCK_RUN), str(PLUGIN_RUN), "--input", str(SAMPLE_INPUT),
            "--max-requests", "2", "--performance-max-requests", "8",
            "--batch-sizes", "1", "--performance-repeats", "1"]
    assert main(argv) == 0
    assert main(argv) == 2
    assert main([*argv, "--overwrite"]) == 0


@requires_evidence
def test_all_without_synthetic_fixture_skips_comparison_plot(tmp_path):
    output = tmp_path / "evaluation"
    assert main(["all", "--output-dir", str(output), "--run-dir", str(MOCK_RUN),
                 "--compare-runs", str(MOCK_RUN), str(PLUGIN_RUN), "--input", str(SAMPLE_INPUT),
                 "--no-synthetic-fixture", "--max-requests", "4",
                 "--performance-max-requests", "8", "--batch-sizes", "1",
                 "--performance-repeats", "1"]) == 0
    manifest = json.loads(
        (output / "plots" / "plots_manifest.json").read_text(encoding="utf-8")
    )
    skipped = {item["name"]: item["reason"] for item in manifest["skipped"]}
    assert "forgetting_model_compare" in skipped
    assert "至少两个模型" in skipped["forgetting_model_compare"]


# ------------------------------------------------------------- 独立子命令串联


@requires_evidence
def test_plots_and_report_subcommands_after_metrics(tmp_path):
    output = tmp_path / "eval"
    assert main(["metrics", "--predictions", str(MOCK_RUN), str(PLUGIN_RUN),
                 "--output-dir", str(output)]) == 0
    assert main(["plots", "--evaluation-dir", str(output),
                 "--predictions", str(MOCK_RUN / "predictions.csv"),
                 str(PLUGIN_RUN / "predictions.csv")]) == 0
    assert (output / "plots" / "auc_comparison.png").is_file()
    report_path = tmp_path / "report.md"
    assert main(["report", "--evaluation-dir", str(output), "--output", str(report_path),
                 "--command", "python evaluate.py metrics --predictions a.csv"]) == 0
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "python evaluate.py metrics --predictions a.csv" in text
    # 未执行性能测试时不得出现性能表格。
    assert "未执行推理性能测试" in text


def test_plots_without_any_data_returns_two(tmp_path):
    assert main(["plots", "--evaluation-dir", str(tmp_path / "empty"),
                 "--output-dir", str(tmp_path / "out")]) == 2


def test_report_on_missing_directory_returns_two(tmp_path):
    assert main(["report", "--evaluation-dir", str(tmp_path / "nope"),
                 "--output", str(tmp_path / "r.md")]) == 2


@requires_evidence
def test_cli_runs_from_another_working_directory(tmp_path):
    """CLI 不依赖当前工作目录，也不依赖任何绝对路径常量。"""
    output = tmp_path / "eval"
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"), "metrics",
         "--predictions", str(MOCK_RUN / "predictions.csv"), "--output-dir", str(output)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "metrics" / "metrics.csv").is_file()


@requires_evidence
def test_evaluation_does_not_touch_ruikaining_evidence(tmp_path):
    """评估程序只读 evidence/，绝不在其中写入或删除任何文件。"""
    before = {
        path.relative_to(ROOT).as_posix(): path.stat().st_mtime_ns
        for path in (ROOT / "evidence").rglob("*") if path.is_file()
    }
    assert main(["metrics", "--predictions", str(MOCK_RUN),
                 "--output-dir", str(tmp_path / "eval")]) == 0
    after = {
        path.relative_to(ROOT).as_posix(): path.stat().st_mtime_ns
        for path in (ROOT / "evidence").rglob("*") if path.is_file()
    }
    assert before == after
