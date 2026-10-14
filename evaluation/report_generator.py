"""模块六：模型评估报告自动生成（Markdown）。

报告的全部数值、表格与图片链接都来自其他五个模块实际写出的 JSON / CSV 产物，
生成器不写死任何实验结果。缺少某项数据时，对应章节明确写「未执行」或
「待真实模型接入」并说明原因，绝不用占位数字填充。

章节顺序与 ``docs/TEST_REPORT.md`` 保持一致的中文技术文档风格。
"""
from __future__ import annotations

import os
import platform
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from evaluation.common import (
    EVALUATION_VERSION,
    EvaluationError,
    ensure_output_dir,
    read_json_document,
    utc_now_iso,
    write_text,
)
from evaluation.forgetting_test import (
    build_controlled_markdown,
    build_descriptive_markdown,
)
from evaluation.model_compare import build_comparison_markdown
from evaluation.performance_test import build_performance_markdown, environment_info
from evaluation.visualizer import load_artifacts

PLOT_TITLES = {
    "metric_roc_auc": "AUC 对比柱状图",
    "metric_accuracy": "Accuracy 对比柱状图",
    "metric_log_loss": "Log Loss 对比柱状图",
    "roc_curve": "ROC 曲线",
    "prediction_distribution": "预测分布图",
    "forgetting_observed": "遗忘时间响应曲线（观察性）",
    "forgetting_model_compare": "不同模型的时间响应对比图",
    "latency": "推理延迟对比图",
    "throughput": "不同 batch size 下的吞吐量图",
}

PENDING_REAL_MODEL = "**待真实模型接入**"

TRAINING_TEAM_REQUIREMENTS = (
    "可导入的模型工厂 `create_model(config)` 与完整推理代码（`module:function`）。",
    "模型权重文件及其版本号、校验值（SHA256）、训练配置与目标设备要求。",
    "与权重严格匹配的知识点词表 JSON（`schema_version` / `unknown_id` / `skills`）。",
    "题目词表与交互编码公式；多知识点聚合策略、未知标签策略、冷启动先验。",
    "历史长度上限、是否需要真实时间及其单位，以及遗忘机制参数的明确定义。",
    "一组固定输入的参考输出，用于核对接入后的数值一致性。",
    "固定的训练／验证／测试划分，或可复现的划分脚本与随机种子。",
)


def parse_junit(path: str | Path) -> dict:
    """解析 pytest 的 JUnit XML，得到可核对的测试统计。"""
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"JUnit 报告不存在: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise EvaluationError(f"无法解析 JUnit 报告 {path}: {exc}") from exc
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    failed_cases: list[str] = []
    for suite in suites:
        for case in suite.iter("testcase"):
            totals["tests"] += 1
            name = f"{case.get('classname', '?')}::{case.get('name', '?')}"
            if case.find("failure") is not None:
                totals["failures"] += 1
                failed_cases.append(name)
            elif case.find("error") is not None:
                totals["errors"] += 1
                failed_cases.append(name)
            if case.find("skipped") is not None:
                totals["skipped"] += 1
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["failed_cases"] = failed_cases
    totals["source"] = path.as_posix()
    return totals


def _relative_link(target: Path, from_dir: Path) -> str:
    return Path(os.path.relpath(target, from_dir)).as_posix()


def _section_environment(performance: dict | None) -> str:
    environment = (performance or {}).get("environment") or environment_info()
    lines = [
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 操作系统 | {environment.get('platform', platform.platform())} |",
        f"| Python | {environment.get('python_implementation', platform.python_implementation())} "
        f"{environment.get('python', platform.python_version())} |",
        f"| CPU 逻辑核数 | {environment.get('cpu_count')} |",
        f"| 计时方式 | `{environment.get('timer', 'time.perf_counter')}` |",
    ]
    for key, label in (("pandas", "pandas"), ("numpy", "NumPy"), ("scikit_learn", "scikit-learn")):
        if environment.get(key):
            lines.append(f"| {label} | {environment[key]} |")
    lines.append("| 评估模块版本 | " + EVALUATION_VERSION + " |")
    return "\n".join(lines)


def _section_data_sources(metrics: dict | None) -> str:
    if not metrics or not metrics.get("inputs"):
        return "未执行模型评价指标计算，无数据来源可报告。\n"
    lines = [
        "| 标签 | 文件 | 模型 | is_mock | 样本数 | 正/负样本 | 时间来源 | 数据性质 |",
        "| --- | --- | --- | :-: | ---: | ---: | --- | --- |",
    ]
    for item in metrics["inputs"]:
        lines.append(
            f"| {item['label']} | `{item['source_file']}` | `{item['model_name']}` | "
            f"{'是' if item['is_mock'] else '否'} | {item['sample_count']} | "
            f"{item['positive_count']}/{item['negative_count']} | "
            f"`{item['time_source'] or '未声明'}` | {item['data_nature']} |"
        )
    lines += ["", "所有预测文件的 SHA256 与输入数据摘要见 `metrics/metrics.json` 的 `inputs` 字段。"]
    return "\n".join(lines) + "\n"


def _section_configuration(metrics: dict | None, performance: dict | None) -> str:
    lines = ["| 配置项 | 值 |", "| --- | --- |"]
    if metrics:
        configuration = metrics.get("configuration", {})
        lines.append(f"| 分类阈值 threshold | {configuration.get('threshold')} |")
        lines.append(
            f"| 是否包含冷启动样本 | {'是' if configuration.get('include_cold_start') else '否'} |"
        )
        lines.append(f"| 指标来源 | {configuration.get('metric_source')} |")
    else:
        lines.append("| 指标配置 | 未执行指标计算 |")
    if performance:
        settings = performance.get("settings", {})
        lines.append(f"| 性能测试 batch sizes | {settings.get('batch_sizes')} |")
        lines.append(f"| 性能测试重复轮数 / 预热批次数 | "
                     f"{settings.get('repeats')} / {settings.get('warmup_batches')} |")
        lines.append(f"| 请求池规模 | {settings.get('pool_size')} |")
    else:
        lines.append("| 性能测试配置 | 未执行性能测试 |")
    return "\n".join(lines) + "\n"


def _section_metrics(metrics: dict | None) -> str:
    if not metrics or not metrics.get("rows"):
        return "未执行模型评价指标计算（缺少 `metrics/metrics.json`）。\n"
    lines = [
        "指标口径：ROC-AUC 与 Log Loss 使用 `y_pred`（答对概率），绝不使用 `predicted_label`；"
        "Accuracy 使用 `y_pred >= threshold` 得到的分类标签。",
        "",
        "| 模型 | 样本分组 | 样本数 | 正/负 | ROC-AUC | AUC 状态 | Accuracy | Log Loss | Brier |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in metrics["rows"]:
        lines.append(
            f"| {row['label']} | {row['group']} | {row['sample_count']} | "
            f"{row['positive_count']}/{row['negative_count']} | {_num(row['roc_auc'])} | "
            f"{row['auc_status']} | {_num(row['accuracy'])} | {_num(row['log_loss'])} | "
            f"{_num(row['brier_score'])} |"
        )
    undefined = [row for row in metrics["rows"] if row["auc_status"] != "computed"]
    if undefined:
        lines += ["", "**AUC 不可计算的样本组（未编造数值）：**", ""]
        lines.extend(
            f"- {row['label']} / {row['group']}：{row['auc_reason']}" for row in undefined
        )
    lines += ["", f"> {metrics.get('scope', '')}", ""]
    return "\n".join(lines)


def _section_forgetting(forgetting: dict | None) -> str:
    if not forgetting:
        return "未执行遗忘机制专项测试（缺少 `forgetting/forgetting_summary.json`）。\n"
    lines = ["### A. 现有预测结果的描述性分析", ""]
    lines.append(build_descriptive_markdown(forgetting.get("descriptive") or {}))
    lines += ["", "### B. 固定学习历史的受控时间测试", ""]
    entries = forgetting.get("controlled_models") or []
    if not entries:
        lines.append("未执行受控时间测试。\n")
    for entry in entries:
        if entry.get("label"):
            lines.append(f"#### 模型：`{entry['label']}`")
            lines.append("")
        lines.append(build_controlled_markdown(entry))
        lines.append("")
    return "\n".join(lines)


def _section_comparison(comparison: dict | None) -> str:
    if not comparison:
        return "未执行多模型对比测试（缺少 `comparison/model_comparison.json`）。\n"
    lines = [build_comparison_markdown(comparison), ""]
    coverage = comparison.get("coverage") or {}
    if any(item.get("dropped") for item in coverage.values()):
        lines += ["#### 样本覆盖与丢弃明细", "",
                  "| 模型 | 原始样本数 | 参与对比 | 丢弃 | 丢弃样本示例 |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for label, item in coverage.items():
            lines.append(
                f"| {label} | {item['sample_count']} | {item['used']} | {item['dropped']} | "
                f"{', '.join(item['dropped_examples']) or '—'} |"
            )
        lines.append("")
    return "\n".join(lines)


def _section_performance(performance: dict | None) -> str:
    if not performance:
        return "未执行推理性能测试（缺少 `performance/performance.json`）。\n"
    return build_performance_markdown(performance) + "\n"


def _section_plots(manifest: dict | None, plots_dir: Path, report_dir: Path) -> str:
    if not manifest:
        return "未生成可视化图表（缺少 `plots/plots_manifest.json`）。\n"
    created = [item for item in manifest.get("plots", []) if item["status"] == "created"]
    skipped = [item for item in manifest.get("plots", []) if item["status"] == "skipped"]
    lines = [f"本图使用 matplotlib 生成，DPI = {manifest.get('dpi')}，"
             f"标签语言 `{manifest.get('font_language')}`。"
             "凡数据来自 `is_mock=True` 的模拟模型，图中均带 `DEMO · SYNTHETIC DATA` 水印。", ""]
    if not created:
        lines.append("本次没有生成任何图表。\n")
    for item in created:
        title = PLOT_TITLES.get(item["name"], item["name"])
        link = _relative_link(plots_dir / item["file"], report_dir)
        lines += [f"### {title}", "", f"![{title}]({link})", "",
                  f"数据来源：`{item['source']}`；模拟数据水印："
                  f"{'有' if item['demo_watermark'] else '无'}", ""]
    if skipped:
        lines += ["### 未生成的图表及原因", ""]
        lines.extend(f"- **{PLOT_TITLES.get(item['name'], item['name'])}**：{item['reason']}"
                     for item in skipped)
        lines.append("")
    return "\n".join(lines)


def _section_boundary(acceptance: dict | None) -> str:
    if not acceptance:
        return ("本次报告未采集边界与异常测试结果。请运行 `python scripts/acceptance_evaluation.py` "
                "后重新生成报告，或使用 `--acceptance-summary` 指定摘要文件。\n")
    checks = acceptance.get("checks") or []
    # 验收摘要带 category 字段时，本节只呈现边界与异常类检查。
    boundary = [item for item in checks if item.get("category") == "boundary"] or [
        item for item in checks if "category" not in item
    ]
    if not boundary:
        return "边界与异常测试摘要为空。\n"
    lines = [
        f"以下结果来自实际执行 `{acceptance.get('script', 'scripts/acceptance_evaluation.py')}` 的摘要："
        f"共 **{len(boundary)}** 项边界与异常检查，通过 "
        f"**{sum(1 for item in boundary if item.get('passed'))}** 项。",
        "",
        "| # | 检查项 | 期望 | 实际 | 结果 |",
        "| ---: | --- | --- | --- | :-: |",
    ]
    for index, item in enumerate(boundary, start=1):
        lines.append(
            f"| {index} | {item.get('name')} | {item.get('expected', '')} | "
            f"{item.get('observed', '')} | {'通过' if item.get('passed') else '失败'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_tests(junit: dict | None) -> str:
    if not junit:
        return ("本次报告未采集自动化测试结果。可使用 `--junit-xml qa/evaluation_junit.xml` "
                "指定 pytest 的 JUnit 报告文件。\n")
    lines = [
        f"实际执行 pytest，数据来源：`{junit['source']}`。",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| 用例总数 | {junit['tests']} |",
        f"| 通过 | {junit['passed']} |",
        f"| 失败 | {junit['failures']} |",
        f"| 错误 | {junit['errors']} |",
        f"| 跳过 | {junit['skipped']} |",
        "",
    ]
    if junit["failed_cases"]:
        lines += ["**失败用例：**", ""]
        lines.extend(f"- `{name}`" for name in junit["failed_cases"])
        lines.append("")
    else:
        lines.append("本次采集到的用例全部通过。\n")
    return "\n".join(lines)


def _section_pending(
    metrics: dict | None, forgetting: dict | None, comparison: dict | None, performance: dict | None
) -> str:
    """根据实际数据状态列出尚未完成的真实模型验证事项，不夸大完成度。"""
    mock_labels: list[str] = []
    if metrics:
        mock_labels = [item["label"] for item in metrics.get("inputs", []) if item["is_mock"]]
    items: list[str] = []
    real_metrics = metrics and any(not item["is_mock"] for item in metrics.get("inputs", []))
    if not real_metrics:
        labels = f"（当前评估对象：{', '.join(mock_labels)}）" if mock_labels else ""
        items.append(
            f"**真实 DKT/AKT 的预测精度尚未验证**{labels}：现有预测来自模拟模型，"
            "只能证明评估程序正确，不能给出真实模型的 AUC / Accuracy / Log Loss 结论。"
        )
    entries = (forgetting or {}).get("controlled_models") or []
    real_time_response = any(
        entry.get("available") and not entry.get("model", {}).get("is_mock") for entry in entries
    )
    if not real_time_response:
        items.append(
            "**真实遗忘机制的有效性尚未验证**：受控时间测试目前只覆盖模拟模型与明确标注的合成"
            "时间敏感夹具，缺少训练组带遗忘机制的模型权重。"
        )
    if not (comparison and any(not item["is_mock"] for item in comparison.get("models", []))):
        items.append(
            "**真实模型与基线的对照实验尚未开展**：现有对比全部发生在模拟模型之间，"
            "不能证明任何新模型优于基线。"
        )
    if not (performance and not (performance.get("model") or {}).get("is_mock")):
        items.append(
            "**真实模型推理性能与鸿蒙端侧性能尚未验证**：本报告中的延迟与吞吐量来自开发机上的"
            "模拟模型，与端侧设备无关。"
        )
    items.append(
        "**真实数据的测试集划分尚未确认**：正式评估需要训练组固定的训练/验证/测试划分，"
        "当前使用的是负责人 A 生成的合成数据与人工夹具。"
    )
    header = (
        f"以下事项在当前条件下无法完成，统一标注为 {PENDING_REAL_MODEL}，"
        "不以任何估算值、占位数字或示例数值代替："
    )
    body = "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))
    return f"{header}\n\n{body}\n"


def _section_reproduce(commands: Sequence[str], evaluation_dir: Path) -> str:
    lines = [
        "在工程根目录（含 `evaluate.py` 的目录）执行以下命令即可复现本报告的数据：",
        "",
        "```bash",
    ]
    if commands:
        lines.extend(commands)
    else:
        lines += [
            "python evaluate.py all --output-dir outputs/evaluation \\",
            "    --run-dir evidence/mock_run --compare-runs evidence/mock_run evidence/plugin_run \\",
            "    --input sample_data.csv --overwrite",
        ]
    lines += [
        "```",
        "",
        "评估产物目录：",
        "",
        f"- `{evaluation_dir.as_posix()}`：`metrics/`、`forgetting/`、`comparison/`、"
        "`performance/`、`plots/`、`reports/`。",
        "- 每个子目录中的 JSON 保留了完整配置与数据指纹，CSV 便于直接查看与二次分析。",
        "",
    ]
    return "\n".join(lines)


def _num(value: float | None, digits: int = 4) -> str:
    return "不可计算" if value is None else f"{value:.{digits}f}"


def build_report(
    evaluation_dir: str | Path,
    *,
    report_dir: Path,
    commands: Sequence[str] = (),
    junit: dict | None = None,
    acceptance: dict | None = None,
    title: str = "鸿蒙智学 · 知识追踪模型评估报告",
) -> str:
    """组织全部章节并返回 Markdown 文本。"""
    evaluation_dir = Path(evaluation_dir).expanduser().resolve()
    artifacts = load_artifacts(evaluation_dir)
    metrics = artifacts["metrics"]
    forgetting = artifacts["forgetting"]
    comparison = artifacts["comparison"]
    performance = artifacts["performance"]
    manifest = None
    manifest_path = evaluation_dir / "plots" / "plots_manifest.json"
    if manifest_path.is_file():
        manifest = read_json_document(manifest_path, description="图表清单")

    generated_at = utc_now_iso()
    sections = [
        f"# {title}",
        "",
        f"评估模块版本 **{EVALUATION_VERSION}**，报告生成时间（UTC）**{generated_at}**。"
        "本报告由 `evaluation/report_generator.py` 从实际运行产物自动生成，"
        "所有数值均可追溯到对应模块输出的 JSON / CSV 文件，没有人手工填入的实验数据。",
        "",
        "## 1. 测试任务概述",
        "",
        "本报告覆盖大创项目《基于知识图谱与认知诊断的个性化学习路径动态生成系统》"
        "（产品名「鸿蒙智学」）中负责人 B 刘静涵负责的模型评估工作，共六个模块：",
        "",
        "| 模块 | 代码 | 说明 |",
        "| --- | --- | --- |",
        "| 模型评价指标 | `evaluation/evaluator.py` | ROC-AUC、Accuracy、Log Loss，支持阈值与冷启动分组 |",
        "| 遗忘机制专项测试 | `evaluation/forgetting_test.py` | 描述性时间间隔分析与固定历史受控时间测试 |",
        "| 模型对比测试 | `evaluation/model_compare.py` | 按 `record_id` 严格对齐的多模型指标对比 |",
        "| 推理性能测试 | `evaluation/performance_test.py` | 批次/单样本延迟、P50/P95、吞吐量 |",
        "| 实验结果可视化 | `evaluation/visualizer.py` | 八类图表的自动绘制 |",
        "| 报告生成 | `evaluation/report_generator.py` | 本报告 |",
        "",
        "评估程序不重新实现负责人 A 的测试框架，只消费 `predictions.csv` 与 "
        "`PredictionRequest` / `ModelAdapter` 接口。",
        "",
        "## 2. 测试环境与版本",
        "",
        _section_environment(performance),
        "",
        "## 3. 数据来源及数据性质",
        "",
        _section_data_sources(metrics),
        "",
        "## 4. 模型与运行配置",
        "",
        _section_configuration(metrics, performance),
        "",
        "## 5. 模型评价指标",
        "",
        _section_metrics(metrics),
        "",
        "## 6. 遗忘机制专项测试",
        "",
        _section_forgetting(forgetting),
        "",
        "## 7. 多模型对比测试",
        "",
        _section_comparison(comparison),
        "",
        "## 8. 推理性能测试",
        "",
        _section_performance(performance),
        "",
        "## 9. 实验结果图表",
        "",
        _section_plots(manifest, evaluation_dir / "plots", report_dir),
        "",
        "## 10. 异常情况与边界测试",
        "",
        _section_boundary(acceptance),
        "",
        "## 11. 自动化测试结果",
        "",
        _section_tests(junit),
        "",
        "## 12. 当前未完成的真实模型验证事项",
        "",
        _section_pending(metrics, forgetting, comparison, performance),
        "",
        "## 13. 复现实验命令",
        "",
        _section_reproduce(commands, evaluation_dir),
        "",
        "## 14. 后续真实模型接入说明",
        "",
        "训练组（谢忠宇、汤书哲）交付 DKT / AKT 模型时，需要提供 `docs/MODEL_INTERFACE.md` "
        "所列的全部训练资产：",
        "",
    ]
    sections.extend(f"{index}. {item}" for index, item in enumerate(TRAINING_TEAM_REQUIREMENTS, start=1))
    sections += [
        "",
        "接入后按下列顺序重跑评估，即可用真实模型替换全部模拟结果：",
        "",
        "```bash",
        "# 1. 用真实模型跑出 predictions.csv（需要训练组词表）",
        "python main.py --model plugin --factory <模块>:create_model \\",
        "    --model-config model/config.json --vocab model/skill_mapping.json \\",
        "    --input data/heldout.csv --output-dir outputs/real_model",
        "",
        "# 2. 用同一份评估程序评估真实模型，并与基线对比",
        "python evaluate.py metrics --predictions outputs/real_model/predictions.csv",
        "python evaluate.py compare --predictions outputs/baseline/predictions.csv \\",
        "    outputs/real_model/predictions.csv --labels baseline dkt",
        "python evaluate.py all --run-dir outputs/real_model --overwrite",
        "```",
        "",
        "只要预测文件满足 `docs/EVALUATION_INTERFACE.md` 的输入格式，本评估模块无需修改代码"
        "即可评估真实模型；`is_mock=False` 时报告会自动去掉模拟数据的说明与图表水印。",
        "",
    ]
    return "\n".join(sections)


def generate_report(
    evaluation_dir: str | Path,
    output_path: str | Path,
    *,
    commands: Sequence[str] = (),
    junit_xml: str | Path | None = None,
    acceptance_summary: str | Path | None = None,
    title: str = "鸿蒙智学 · 知识追踪模型评估报告",
    overwrite: bool = False,
    also_write: Sequence[str | Path] = (),
) -> Path:
    """生成 Markdown 报告并写盘；``also_write`` 可同时写出到其他路径（例如 docs/）。"""
    evaluation_dir = Path(evaluation_dir).expanduser().resolve()
    if not evaluation_dir.is_dir():
        raise EvaluationError(f"评估输出目录不存在: {evaluation_dir}")
    output_path = Path(output_path).expanduser().resolve()
    ensure_output_dir(output_path.parent, overwrite=True)
    if output_path.exists() and not overwrite:
        raise EvaluationError(f"报告已存在: {output_path}；如需替换请使用 --overwrite")

    junit = parse_junit(junit_xml) if junit_xml else None
    acceptance = None
    if acceptance_summary:
        acceptance = read_json_document(Path(acceptance_summary), description="验收摘要")

    content = build_report(
        evaluation_dir, report_dir=output_path.parent, commands=commands,
        junit=junit, acceptance=acceptance, title=title,
    )
    write_text(output_path, content)
    for extra in also_write:
        extra_path = Path(extra).expanduser().resolve()
        # docs/ 中的报告引用图片时改用相对评估目录的路径，避免链接失效。
        extra_content = build_report(
            evaluation_dir, report_dir=extra_path.parent, commands=commands,
            junit=junit, acceptance=acceptance, title=title,
        )
        ensure_output_dir(extra_path.parent, overwrite=True)
        write_text(extra_path, extra_content)
    return output_path
