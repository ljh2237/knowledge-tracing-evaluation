"""报告生成模块测试：章节完整性、数据驱动与「不编造」约束。"""
from __future__ import annotations

import json

import pytest

from evaluation.common import EvaluationError
from evaluation.report_generator import (
    TRAINING_TEAM_REQUIREMENTS,
    build_report,
    generate_report,
    parse_junit,
)

REQUIRED_HEADINGS = [
    "## 1. 测试任务概述",
    "## 2. 测试环境与版本",
    "## 3. 数据来源及数据性质",
    "## 4. 模型与运行配置",
    "## 5. 模型评价指标",
    "## 6. 遗忘机制专项测试",
    "## 7. 多模型对比测试",
    "## 8. 推理性能测试",
    "## 9. 实验结果图表",
    "## 10. 异常情况与边界测试",
    "## 11. 自动化测试结果",
    "## 12. 当前未完成的真实模型验证事项",
    "## 13. 复现实验命令",
    "## 14. 后续真实模型接入说明",
]

JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" failures="1" errors="0" skipped="1">
    <testcase classname="tests.test_a" name="test_ok" time="0.01"/>
    <testcase classname="tests.test_a" name="test_also_ok" time="0.01"/>
    <testcase classname="tests.test_b" name="test_skipped" time="0.00">
      <skipped message="no data"/>
    </testcase>
    <testcase classname="tests.test_b" name="test_broken" time="0.02">
      <failure message="boom">traceback here</failure>
    </testcase>
  </testsuite>
</testsuites>
"""


def metrics_payload():
    return {
        "scope": "指标只在该预测文件上成立。",
        "configuration": {"threshold": 0.5, "include_cold_start": True,
                          "metric_source": "scikit-learn"},
        "inputs": [
            {"label": "mock_a", "source_file": "evidence/mock_run/predictions.csv",
             "model_name": "mock_history_beta_v1", "is_mock": True, "data_nature": "模拟模型输出",
             "time_source": "timestamp", "sample_count": 288, "positive_count": 153,
             "negative_count": 135, "positive_rate": 0.53, "cold_start_count": 12,
             "source_sha256": "0" * 64, "experiment_status": "mock_pipeline_only",
             "run_summary_available": True},
        ],
        "rows": [
            {"label": "mock_a", "group": "all", "group_description": "全部有效样本",
             "sample_count": 288, "positive_count": 153, "negative_count": 135,
             "roc_auc": 0.5233, "auc_status": "computed", "auc_reason": None,
             "accuracy": 0.5243, "log_loss": 0.7070, "brier_score": 0.2564,
             "is_mock": True, "threshold": 0.5},
            {"label": "mock_a", "group": "cold_start", "group_description": "冷启动样本",
             "sample_count": 12, "positive_count": 5, "negative_count": 7,
             "roc_auc": None, "auc_status": "undefined_single_class",
             "auc_reason": "测试集只包含正样本（正样本 12 条、负样本 0 条），ROC-AUC 无定义，不编造数值",
             "accuracy": 0.4167, "log_loss": 0.6931, "brier_score": 0.25,
             "is_mock": True, "threshold": 0.5},
        ],
    }


def plots_manifest():
    return {
        "dpi": 200, "font_language": "zh",
        "plots": [
            {"name": "metric_roc_auc", "file": "auc_comparison.png", "status": "created",
             "reason": None, "source": "metrics/metrics.json", "demo_watermark": True},
            {"name": "forgetting_model_compare", "file": None, "status": "skipped",
             "reason": "只有 1 个模型具备受控时间响应数据",
             "source": "forgetting/forgetting_summary.json", "demo_watermark": False},
        ],
    }


def write_artifacts(root, *, metrics=True, manifest=True):
    if metrics:
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        (root / "metrics" / "metrics.json").write_text(
            json.dumps(metrics_payload(), ensure_ascii=False), encoding="utf-8"
        )
    if manifest:
        (root / "plots").mkdir(parents=True, exist_ok=True)
        (root / "plots" / "plots_manifest.json").write_text(
            json.dumps(plots_manifest(), ensure_ascii=False), encoding="utf-8"
        )
        (root / "plots" / "auc_comparison.png").write_bytes(b"\x89PNG\r\n\x1a\nstub")
    return root


def test_report_contains_all_fourteen_sections(tmp_path):
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path)
    for heading in REQUIRED_HEADINGS:
        assert heading in report, heading


def test_report_renders_metrics_and_undefined_auc_reason(tmp_path):
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path)
    assert "0.5233" in report
    assert "mock_history_beta_v1" in report
    assert "undefined_single_class" in report
    assert "ROC-AUC 无定义，不编造数值" in report


def test_report_links_plots_and_lists_skipped_reasons(tmp_path):
    # report_dir 就是评估目录本身时，图片相对路径为 plots/xxx.png。
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path)
    assert "![AUC 对比柱状图](plots/auc_comparison.png)" in report
    assert "未生成的图表及原因" in report
    assert "只有 1 个模型具备受控时间响应数据" in report


def test_report_links_plots_from_reports_subdirectory(tmp_path):
    evaluation = write_artifacts(tmp_path / "evaluation")
    report = build_report(evaluation, report_dir=evaluation / "reports")
    assert "![AUC 对比柱状图](../plots/auc_comparison.png)" in report


def test_report_marks_missing_sections_as_not_executed(tmp_path):
    report = build_report(tmp_path, report_dir=tmp_path)
    assert "未执行模型评价指标计算" in report
    assert "未执行遗忘机制专项测试" in report
    assert "未执行多模型对比测试" in report
    assert "未执行推理性能测试" in report
    assert "未生成可视化图表" in report
    assert "未采集边界与异常测试结果" in report
    assert "未采集自动化测试结果" in report


def test_report_lists_pending_real_model_validation(tmp_path):
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path)
    assert "真实 DKT/AKT 的预测精度尚未验证" in report
    assert "真实遗忘机制的有效性尚未验证" in report
    assert "鸿蒙端侧性能尚未验证" in report


def test_report_includes_training_team_requirements(tmp_path):
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path)
    for requirement in TRAINING_TEAM_REQUIREMENTS:
        assert requirement.split("（")[0] in report


def test_report_uses_provided_reproduce_commands(tmp_path):
    commands = ["python evaluate.py metrics --predictions a.csv --overwrite"]
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path, commands=commands)
    assert commands[0] in report


def test_report_renders_acceptance_summary(tmp_path):
    acceptance = {
        "script": "scripts/acceptance_evaluation.py",
        "checks": [
            {"name": "指标数值校验", "category": "evidence", "expected": "退出码 0",
             "observed": "退出码 0", "passed": True},
            {"name": "缺列输入报错", "category": "boundary", "expected": "退出码 2",
             "observed": "退出码 2", "passed": True},
            {"name": "样本不一致失败", "category": "boundary", "expected": "退出码 2",
             "observed": "退出码 2", "passed": True},
        ],
    }
    report = build_report(
        write_artifacts(tmp_path), report_dir=tmp_path, acceptance=acceptance
    )
    # 第 10 节只呈现边界与异常类检查，不把证据类检查混入。
    assert "共 **2** 项边界与异常检查，通过 **2** 项" in report
    assert "缺列输入报错" in report and "样本不一致失败" in report
    assert "| 1 | 指标数值校验 |" not in report


def test_report_renders_junit_summary(tmp_path):
    junit_path = tmp_path / "junit.xml"
    junit_path.write_text(JUNIT_XML, encoding="utf-8")
    junit = parse_junit(junit_path)
    assert junit["tests"] == 4
    assert junit["passed"] == 2
    assert junit["failures"] == 1
    assert junit["skipped"] == 1
    assert junit["failed_cases"] == ["tests.test_b::test_broken"]
    report = build_report(write_artifacts(tmp_path), report_dir=tmp_path, junit=junit)
    assert "| 用例总数 | 4 |" in report
    assert "test_broken" in report


def test_parse_junit_rejects_missing_and_broken_files(tmp_path):
    with pytest.raises(EvaluationError, match="不存在"):
        parse_junit(tmp_path / "nope.xml")
    broken = tmp_path / "broken.xml"
    broken.write_text("<testsuites>", encoding="utf-8")
    with pytest.raises(EvaluationError, match="无法解析"):
        parse_junit(broken)


def test_generate_report_writes_file_and_refuses_overwrite(tmp_path):
    evaluation = write_artifacts(tmp_path / "eval")
    output = tmp_path / "reports" / "evaluation_report.md"
    written = generate_report(evaluation, output, overwrite=False)
    assert written.is_file()
    assert "## 1. 测试任务概述" in written.read_text(encoding="utf-8")
    with pytest.raises(EvaluationError, match="--overwrite"):
        generate_report(evaluation, output, overwrite=False)
    generate_report(evaluation, output, overwrite=True)


def test_generate_report_also_writes_second_copy_with_correct_links(tmp_path):
    evaluation = write_artifacts(tmp_path / "eval")
    output = evaluation / "reports" / "evaluation_report.md"
    docs = tmp_path / "docs" / "EVALUATION_TEST_REPORT.md"
    generate_report(evaluation, output, overwrite=True, also_write=[docs])
    assert docs.is_file()
    # docs/ 下的副本必须用相对自身目录的路径引用图片，链接不能失效。
    assert "![AUC 对比柱状图](../eval/plots/auc_comparison.png)" in docs.read_text(encoding="utf-8")


def test_generate_report_rejects_missing_evaluation_dir(tmp_path):
    with pytest.raises(EvaluationError, match="评估输出目录不存在"):
        generate_report(tmp_path / "nope", tmp_path / "out.md")


def test_report_contains_no_placeholder_numbers(tmp_path):
    """缺少数据时不得出现占位数值，只能明确写「未执行」或「待真实模型接入」。"""
    report = build_report(tmp_path, report_dir=tmp_path)
    for marker in ("TBD", "TODO", "FIXME", "XXXX", "???"):
        assert marker not in report
    assert "待真实模型接入" in report
    # 缺少数据的四项专项测试只能写「未执行」，不能凭空给出指标表格。
    for section in ("未执行模型评价指标计算", "未执行推理性能测试"):
        assert section in report
    assert "| ROC-AUC |" not in report
