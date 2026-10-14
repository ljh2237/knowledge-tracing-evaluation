"""刘静涵 知识追踪模型评估模块统一入口。

用法示例::

    python evaluate.py metrics   --predictions evidence/mock_run/predictions.csv
    python evaluate.py forgetting --predictions evidence/mock_run --input sample_data.csv
    python evaluate.py compare   --predictions evidence/mock_run evidence/plugin_run
    python evaluate.py performance --input sample_data.csv --batch-sizes 1,8,32,64
    python evaluate.py plots     --evaluation-dir outputs/evaluation
    python evaluate.py report    --evaluation-dir outputs/evaluation
    python evaluate.py all       --run-dir evidence/mock_run \\
                                 --compare-runs evidence/mock_run evidence/plugin_run

退出码与负责人 A 的 ``main.py`` 保持一致：成功 0，输入或运行错误 2。
``all`` 命令只跳过「输入不存在」的步骤并写明原因；真正的错误（数据损坏、样本无法对齐等）
会立即以退出码 2 失败，不会吞掉异常伪装成功。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from contracts import DataValidationError, ModelValidationError, VERSION
from data_loader import SkillVocabulary
from evaluation.common import (
    EVALUATION_VERSION,
    EvaluationError,
    ensure_output_dir,
    resolve_predictions,
    utc_now_iso,
    write_document,
)
from evaluation.evaluator import run_metrics
from evaluation.forgetting_test import (
    DEFAULT_BUCKET_EDGES,
    DEFAULT_INTERVAL_DAYS,
    parse_intervals,
    run_forgetting,
)
from evaluation.model_compare import MISSING_POLICIES, run_comparison
from evaluation.performance_test import (
    DEFAULT_BATCH_SIZES,
    parse_batch_sizes,
    run_performance,
)
from evaluation.report_generator import generate_report
from evaluation.synthetic_time_model import SyntheticTimeSensitiveModel
from evaluation.visualizer import (
    PlotInputs,
    configure_style,
    generate_plots,
    load_artifacts,
    require_plot_inputs,
)
from evaluation.common import load_predictions
from mock_model import MockModel
from model_adapter import ModelAdapter

ROOT = Path(__file__).resolve().parent
SYNTHETIC_FIXTURE_LABEL = "synthetic_time_sensitive_fixture_v1"


# --------------------------------------------------------------------------- 参数


def _add_output_options(parser: argparse.ArgumentParser, *, default_dir: Path) -> None:
    parser.add_argument("--output-dir", type=Path, default=default_dir,
                        help=f"评估产物根目录（默认 {default_dir}）")
    parser.add_argument("--overwrite", action="store_true",
                        help="显式允许覆盖已存在的评估产物")


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("模型")
    group.add_argument("--model", choices=["mock", "plugin"], default="mock",
                       help="mock 使用负责人 A 的模拟模型；plugin 使用 --factory 指定的外部模型")
    group.add_argument("--factory", help="外部模型工厂 module:function")
    group.add_argument("--model-config", type=Path, help="传给模型工厂的 JSON 配置文件")
    group.add_argument("--vocab", type=Path, help="训练组提供的知识点词表（注入模型配置的 vocab_path）")


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("数据来源")
    group.add_argument("--max-history", type=int, default=256)
    group.add_argument("--invalid-policy", choices=["error", "drop"], default="error")
    group.add_argument("--timestamp-unit", choices=["iso", "s", "ms"], default="iso")
    group.add_argument("--unknown-skill-policy", choices=["error", "unk"], default="error")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="evaluate.py",
        description="鸿蒙智学 · 知识追踪模型评估模块（刘静涵）：指标、遗忘、对比、性能、图表、报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="执行 `python evaluate.py all` 可一次性完成当前条件下所有能执行的评估。",
    )
    result.add_argument("--version", action="version",
                        version=f"{EVALUATION_VERSION} (framework {VERSION})")
    subparsers = result.add_subparsers(dest="command", required=True, metavar="子命令")

    metrics = subparsers.add_parser("metrics", help="模型评价指标：ROC-AUC / Accuracy / Log Loss")
    metrics.add_argument("--predictions", nargs="+", required=True, metavar="PATH",
                         help="预测 CSV 或负责人 A 的完整运行目录，可多个")
    metrics.add_argument("--threshold", type=float, default=0.5, help="Accuracy 分类阈值，默认 0.5")
    metrics.add_argument("--exclude-cold-start", action="store_true",
                         help="在 all 组中排除冷启动样本（冷启动分组仍单独列出）")
    metrics.add_argument("--seed", type=int, default=20261013)
    _add_output_options(metrics, default_dir=ROOT / "outputs" / "evaluation")

    forgetting = subparsers.add_parser("forgetting", help="遗忘机制专项测试（描述性 + 受控固定历史）")
    forgetting.add_argument("--predictions", metavar="PATH",
                            help="用于描述性分析的预测 CSV 或运行目录")
    forgetting.add_argument("--input", type=Path, metavar="CSV",
                            help="用于受控固定历史测试的原始输入 CSV")
    _add_model_options(forgetting)
    forgetting.add_argument("--controlled-model", action="append", default=[], metavar="LABEL=FACTORY",
                            help="额外的受控测试模型，可重复，例如 --controlled-model dkt=mymod:create_model")
    forgetting.add_argument("--intervals", default=",".join(f"{value:g}" for value in DEFAULT_INTERVAL_DAYS),
                            help="受控测试的时间间隔序列，默认 0,1,3,7,14")
    forgetting.add_argument("--unit", choices=["seconds", "minutes", "hours", "days"], default="days",
                            help="--intervals 的单位，默认 days")
    forgetting.add_argument("--repeats", type=int, default=3, help="每个时间间隔的重复次数")
    forgetting.add_argument("--max-requests", type=int, default=24, help="受控测试选取的候选样本数")
    forgetting.add_argument("--batch-size", type=int, default=64)
    forgetting.add_argument("--bucket-edges", default=",".join(f"{edge:g}" for edge in DEFAULT_BUCKET_EDGES),
                            help="描述性分析的时间分桶边界（秒）")
    forgetting.add_argument("--threshold", type=float, default=0.5)
    forgetting.add_argument("--seed", type=int, default=20261013)
    _add_source_options(forgetting)
    _add_output_options(forgetting, default_dir=ROOT / "outputs" / "evaluation")

    compare = subparsers.add_parser("compare", help="多模型对比测试")
    compare.add_argument("--predictions", nargs="+", required=True, metavar="PATH",
                         help="两个及以上模型的预测 CSV 或运行目录")
    compare.add_argument("--labels", nargs="+", metavar="NAME", help="为每个模型指定唯一名称")
    compare.add_argument("--baseline", help="基准模型的名称或序号（从 0 开始），默认第一个")
    compare.add_argument("--missing-policy", choices=list(MISSING_POLICIES), default="strict",
                         help="strict（默认）样本不一致直接失败；intersect 显式取交集并记录丢弃明细")
    compare.add_argument("--threshold", type=float, default=0.5)
    compare.add_argument("--exclude-cold-start", action="store_true")
    _add_output_options(compare, default_dir=ROOT / "outputs" / "evaluation")

    performance = subparsers.add_parser("performance", help="模型推理性能测试")
    performance.add_argument("--input", type=Path, required=True, metavar="CSV")
    _add_model_options(performance)
    performance.add_argument("--batch-sizes", default=",".join(str(size) for size in DEFAULT_BATCH_SIZES),
                             help="要比较的批量大小，默认 1,8,32,64")
    performance.add_argument("--repeats", type=int, default=3, help="每个批量大小的重复轮数")
    performance.add_argument("--warmup", type=int, default=1, help="预热批次数")
    performance.add_argument("--max-requests", type=int, default=1024, help="请求池上限")
    _add_source_options(performance)
    _add_output_options(performance, default_dir=ROOT / "outputs" / "evaluation")

    plots = subparsers.add_parser("plots", help="实验结果可视化")
    plots.add_argument("--evaluation-dir", type=Path, default=ROOT / "outputs" / "evaluation",
                       help="已生成的评估产物目录")
    plots.add_argument("--predictions", nargs="*", default=[], metavar="PATH",
                       help="可选：绘制 ROC 曲线所需的预测文件")
    plots.add_argument("--output-dir", type=Path, default=None,
                       help="图表输出根目录，默认与 --evaluation-dir 相同")
    plots.add_argument("--overwrite", action="store_true")

    report = subparsers.add_parser("report", help="自动生成 Markdown 评估报告")
    report.add_argument("--evaluation-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    report.add_argument("--output", type=Path, default=None, help="报告输出路径")
    report.add_argument("--docs-report", type=Path, default=None,
                        help="同时写出的第二份报告路径，例如 docs/EVALUATION_TEST_REPORT.md")
    report.add_argument("--junit-xml", type=Path, default=None, help="pytest 的 JUnit XML 结果")
    report.add_argument("--acceptance-summary", type=Path, default=None,
                        help="验收摘要 JSON（scripts/acceptance_evaluation.py 的输出）")
    # 必须显式指定 dest，否则会覆盖 subparsers 的 dest="command"。
    report.add_argument("--command", dest="reproduce_commands", action="append", default=[],
                        metavar="CMD", help="写入报告的复现命令，可重复")
    report.add_argument("--title", default="鸿蒙智学 · 知识追踪模型评估报告")
    report.add_argument("--overwrite", action="store_true")

    everything = subparsers.add_parser("all", help="按当前可用输入完成全部评估流程")
    everything.add_argument("--run-dir", type=Path, default=ROOT / "evidence" / "mock_run",
                            help="主预测运行目录或预测 CSV")
    everything.add_argument("--compare-runs", nargs="+", type=Path,
                            default=[ROOT / "evidence" / "mock_run", ROOT / "evidence" / "plugin_run"],
                            help="参与模型对比的运行目录或预测 CSV（默认两个模拟模型）")
    everything.add_argument("--input", type=Path, default=ROOT / "sample_data.csv",
                            help="受控遗忘测试与性能测试使用的原始输入 CSV")
    everything.add_argument("--threshold", type=float, default=0.5)
    everything.add_argument("--exclude-cold-start", action="store_true")
    everything.add_argument("--intervals", default=",".join(f"{value:g}" for value in DEFAULT_INTERVAL_DAYS))
    everything.add_argument("--unit", choices=["seconds", "minutes", "hours", "days"], default="days")
    everything.add_argument("--repeats", type=int, default=3)
    everything.add_argument("--max-requests", type=int, default=24,
                            help="受控遗忘测试的候选样本数")
    everything.add_argument("--performance-max-requests", type=int, default=1024)
    everything.add_argument("--batch-sizes", default=",".join(str(size) for size in DEFAULT_BATCH_SIZES))
    everything.add_argument("--performance-repeats", type=int, default=3)
    everything.add_argument("--warmup", type=int, default=1)
    everything.add_argument("--no-synthetic-fixture", action="store_true",
                            help="不把合成时间敏感夹具纳入受控遗忘测试（图表 6 将因只有一个模型而跳过）")
    everything.add_argument("--junit-xml", type=Path, default=None)
    everything.add_argument("--acceptance-summary", type=Path, default=None)
    everything.add_argument("--docs-report", type=Path, default=None)
    everything.add_argument("--seed", type=int, default=20261013)
    _add_source_options(everything)
    _add_output_options(everything, default_dir=ROOT / "outputs" / "evaluation")
    return result


# --------------------------------------------------------------------------- 辅助


def _parse_seconds_list(text: str, description: str) -> tuple[float, ...]:
    values: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            raise EvaluationError(f"{description} 含空项: {text!r}")
        try:
            value = float(chunk)
        except ValueError as exc:
            raise EvaluationError(f"{description} 含非数值项 {chunk!r}") from exc
        if value < 0:
            raise EvaluationError(f"{description} 必须为非负数，实际为 {chunk!r}")
        values.append(value)
    if len(values) < 2:
        raise EvaluationError(f"{description} 至少需要两个边界")
    if any(b <= a for a, b in zip(values[:-1], values[1:], strict=False)):
        raise EvaluationError(f"{description} 必须严格递增: {text!r}")
    return tuple(values)


def build_model_config(args: argparse.Namespace) -> dict:
    """读取模型配置 JSON，并按需注入 --vocab 指向的词表路径。"""
    config: dict = {}
    if getattr(args, "model_config", None):
        try:
            config = json.loads(args.model_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelValidationError(f"无法读取模型配置 {args.model_config}: {exc}") from exc
        if not isinstance(config, dict):
            raise ModelValidationError("模型配置必须是 JSON 对象")
    vocab = getattr(args, "vocab", None)
    if vocab is not None:
        config.setdefault("vocab_path", str(vocab))
    return config


def build_adapter(args: argparse.Namespace, *, label: str | None = None) -> tuple[str, ModelAdapter]:
    """按命令行参数构造模型适配器，返回 (标签, 适配器)。"""
    factory = getattr(args, "factory", None)
    if factory:
        adapter = ModelAdapter.from_factory(factory, build_model_config(args))
    else:
        adapter = ModelAdapter(MockModel())
    return label or adapter.model.name, adapter


def build_controlled_models(args: argparse.Namespace) -> list[tuple[str, ModelAdapter]]:
    """构造受控遗忘测试使用的模型列表（主模型 + 额外 --controlled-model）。"""
    label, adapter = build_adapter(args)
    models: list[tuple[str, ModelAdapter]] = [(label, adapter)]
    seen = {label}
    for specification in getattr(args, "controlled_model", []) or []:
        if "=" not in specification:
            raise EvaluationError(f"--controlled-model 应为 LABEL=FACTORY 形式: {specification!r}")
        extra_label, factory = specification.split("=", 1)
        extra_label = extra_label.strip()
        if not extra_label or not factory.strip():
            raise EvaluationError(f"--controlled-model 标签或工厂为空: {specification!r}")
        if extra_label in seen:
            raise EvaluationError(f"--controlled-model 标签重复: {extra_label!r}")
        seen.add(extra_label)
        models.append((extra_label, ModelAdapter.from_factory(factory.strip(), build_model_config(args))))
    return models


def _existing(paths: Sequence[Path]) -> list[Path]:
    return [path for path in paths if resolve_predictions(path).is_file()]


def _unique_resolved(paths: Sequence[Path]) -> list[Path]:
    seen: list[Path] = []
    for path in paths:
        resolved = resolve_predictions(path).resolve()
        if resolved not in seen:
            seen.append(resolved)
    return seen


def _load_vocabulary(path: Path | None) -> SkillVocabulary | None:
    return SkillVocabulary.load(path) if path else None


def _relative(path: Path) -> str:
    """尽量输出相对工程根目录的路径，便于复制粘贴复现命令。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# --------------------------------------------------------------------------- 子命令


def command_metrics(args: argparse.Namespace) -> dict:
    return run_metrics(
        args.predictions, args.output_dir, threshold=args.threshold,
        include_cold_start=not args.exclude_cold_start, overwrite=args.overwrite,
        seed=args.seed, relative_to=ROOT,
    )


def command_forgetting(args: argparse.Namespace) -> dict:
    if args.predictions is None and args.input is None:
        raise EvaluationError("forgetting 至少需要 --predictions（描述性分析）或 --input（受控测试）")
    controlled: list[tuple[str, ModelAdapter]] = []
    if args.input is not None:
        controlled = build_controlled_models(args)
    return run_forgetting(
        args.output_dir, predictions=args.predictions, input_path=args.input,
        controlled_models=controlled,
        intervals_seconds=parse_intervals(args.intervals, args.unit),
        repeats=args.repeats, max_requests=args.max_requests, batch_size=args.batch_size,
        bucket_edges=_parse_seconds_list(args.bucket_edges, "--bucket-edges"),
        threshold=args.threshold, invalid_policy=args.invalid_policy,
        timestamp_unit=args.timestamp_unit,
        vocabulary=_load_vocabulary(args.vocab), unknown_skill_policy=args.unknown_skill_policy,
        seed=args.seed, overwrite=args.overwrite,
    )


def command_compare(args: argparse.Namespace) -> dict:
    return run_comparison(
        args.predictions, args.output_dir, labels=args.labels, baseline=args.baseline,
        threshold=args.threshold, include_cold_start=not args.exclude_cold_start,
        missing_policy=args.missing_policy, overwrite=args.overwrite, relative_to=ROOT,
    )


def command_performance(args: argparse.Namespace) -> dict:
    _, adapter = build_adapter(args)
    return run_performance(
        args.input, adapter, args.output_dir,
        batch_sizes=parse_batch_sizes(args.batch_sizes), repeats=args.repeats,
        warmup=args.warmup, max_requests=args.max_requests, max_history=args.max_history,
        invalid_policy=args.invalid_policy, timestamp_unit=args.timestamp_unit,
        vocabulary=_load_vocabulary(args.vocab), unknown_skill_policy=args.unknown_skill_policy,
        overwrite=args.overwrite,
    )


def _plot_inputs(evaluation_dir: Path, predictions: Sequence[str | Path]) -> PlotInputs:
    artifacts = load_artifacts(evaluation_dir)
    sources = [load_predictions(resolve_predictions(path)) for path in predictions]
    return PlotInputs(
        metrics=artifacts["metrics"], forgetting=artifacts["forgetting"],
        comparison=artifacts["comparison"], performance=artifacts["performance"],
        roc_sources=tuple(sources),
    )


def command_plots(args: argparse.Namespace) -> dict:
    target = args.output_dir or args.evaluation_dir
    inputs = _plot_inputs(args.evaluation_dir, args.predictions)
    require_plot_inputs(inputs)
    return generate_plots(inputs, target, overwrite=args.overwrite)


def command_report(args: argparse.Namespace) -> dict:
    output = args.output or (args.evaluation_dir / "reports" / "evaluation_report.md")
    result = generate_report(
        args.evaluation_dir, output, commands=args.reproduce_commands, junit_xml=args.junit_xml,
        acceptance_summary=args.acceptance_summary, title=args.title,
        overwrite=args.overwrite,
        also_write=[args.docs_report] if args.docs_report else (),
    )
    return {"report": str(result)}


def command_all(args: argparse.Namespace) -> dict:
    """按当前可用输入执行全部评估步骤，记录每一步的执行状态与实际命令。"""
    output_dir: Path = args.output_dir
    ensure_output_dir(output_dir, overwrite=args.overwrite)
    steps: list[dict] = []
    commands: list[str] = []

    def record(name: str, status: str, detail: str) -> None:
        steps.append({"step": name, "status": status, "detail": detail})

    run_dir_predictions = resolve_predictions(args.run_dir)
    compare_predictions = _existing(args.compare_runs)
    metric_targets = _unique_resolved([run_dir_predictions, *compare_predictions])

    # 1. 模型评价指标
    available_metrics = _existing(metric_targets)
    if available_metrics:
        run_metrics(
            available_metrics, output_dir, threshold=args.threshold,
            include_cold_start=not args.exclude_cold_start, overwrite=args.overwrite,
            seed=args.seed, relative_to=ROOT,
        )
        relative = " ".join(path.relative_to(ROOT).as_posix() for path in available_metrics)
        commands.append(f"python evaluate.py metrics --predictions {relative} --overwrite")
        record("metrics", "executed", f"{len(available_metrics)} 个预测文件")
    else:
        record("metrics", "skipped", f"未找到预测文件：{[str(path) for path in metric_targets]}")

    # 2. 遗忘机制专项测试
    forgetting_predictions = run_dir_predictions if run_dir_predictions.is_file() else None
    if args.input is not None and args.input.is_file():
        controlled: list[tuple[str, ModelAdapter]] = [
            (MockModel.name, ModelAdapter(MockModel()))
        ]
        if not args.no_synthetic_fixture:
            controlled.append((SYNTHETIC_FIXTURE_LABEL, ModelAdapter(SyntheticTimeSensitiveModel())))
        run_forgetting(
            output_dir, predictions=forgetting_predictions, input_path=args.input,
            controlled_models=controlled,
            intervals_seconds=parse_intervals(args.intervals, args.unit),
            repeats=args.repeats, max_requests=args.max_requests, max_history=args.max_history,
            threshold=args.threshold, invalid_policy=args.invalid_policy,
            timestamp_unit=args.timestamp_unit, unknown_skill_policy=args.unknown_skill_policy,
            seed=args.seed, overwrite=args.overwrite,
        )
        detail = f"受控模型 {[label for label, _ in controlled]}；输入 {args.input.name}"
        record("forgetting", "executed", detail)
        extra = ("" if args.no_synthetic_fixture
                 else f" --controlled-model {SYNTHETIC_FIXTURE_LABEL}=evaluation.synthetic_time_model:create_model")
        commands.append(
            f"python evaluate.py forgetting --predictions {run_dir_predictions.relative_to(ROOT).as_posix()} "
            f"--input {args.input.relative_to(ROOT).as_posix()} "
            f"--intervals {args.intervals} --unit {args.unit} --repeats {args.repeats}{extra} --overwrite"
        )
    else:
        record("forgetting", "skipped", f"未找到原始输入 CSV：{args.input}")

    # 3. 模型对比
    if len(compare_predictions) >= 2:
        run_comparison(
            compare_predictions, output_dir, threshold=args.threshold,
            include_cold_start=not args.exclude_cold_start, overwrite=args.overwrite,
            relative_to=ROOT,
        )
        relative = " ".join(path.relative_to(ROOT).as_posix() for path in compare_predictions)
        commands.append(f"python evaluate.py compare --predictions {relative} --overwrite")
        record("compare", "executed", f"{len(compare_predictions)} 个模型")
    else:
        record("compare", "skipped",
               f"模型对比至少需要两个预测文件，当前找到 {len(compare_predictions)} 个")

    # 4. 推理性能
    if args.input is not None and args.input.is_file():
        run_performance(
            args.input, ModelAdapter(MockModel()), output_dir,
            batch_sizes=parse_batch_sizes(args.batch_sizes), repeats=args.performance_repeats,
            warmup=args.warmup, max_requests=args.performance_max_requests,
            max_history=args.max_history, invalid_policy=args.invalid_policy,
            timestamp_unit=args.timestamp_unit, unknown_skill_policy=args.unknown_skill_policy,
            overwrite=args.overwrite,
        )
        commands.append(
            f"python evaluate.py performance --input {args.input.relative_to(ROOT).as_posix()} "
            f"--batch-sizes {args.batch_sizes} --repeats {args.performance_repeats} "
            f"--warmup {args.warmup} --overwrite"
        )
        record("performance", "executed", f"批量大小 {args.batch_sizes}")
    else:
        record("performance", "skipped", f"未找到原始输入 CSV：{args.input}")

    # 5. 可视化
    plot_inputs = _plot_inputs(output_dir, available_metrics)
    require_plot_inputs(plot_inputs)
    manifest = generate_plots(plot_inputs, output_dir, overwrite=args.overwrite)
    commands.append(f"python evaluate.py plots --evaluation-dir {_relative(output_dir)} --overwrite")
    record("plots", "executed",
           f"生成 {len(manifest['created'])} 张图，跳过 {len(manifest['skipped'])} 张")

    # 6. 报告
    commands.append(f"python evaluate.py report --evaluation-dir {_relative(output_dir)} --overwrite")
    generate_report(
        output_dir, output_dir / "reports" / "evaluation_report.md", commands=commands,
        junit_xml=args.junit_xml, acceptance_summary=args.acceptance_summary,
        overwrite=args.overwrite, also_write=[args.docs_report] if args.docs_report else (),
    )
    record("report", "executed", "reports/evaluation_report.md")

    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
        "configuration": {
            "threshold": args.threshold, "include_cold_start": not args.exclude_cold_start,
            "seed": args.seed, "intervals": args.intervals, "interval_unit": args.unit,
            "batch_sizes": args.batch_sizes, "repeats": args.repeats,
        },
        "steps": steps,
        "reproduce_commands": commands,
        "plot_font_language": configure_style(),
    }
    write_document(output_dir / "evaluation_summary.json", summary)
    return summary


HANDLERS = {
    "metrics": command_metrics,
    "forgetting": command_forgetting,
    "compare": command_compare,
    "performance": command_performance,
    "plots": command_plots,
    "report": command_report,
    "all": command_all,
}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = HANDLERS[args.command](args)
    except (EvaluationError, DataValidationError, ModelValidationError, ValueError, OSError) as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 2
    if args.command == "all":
        executed = [step for step in result["steps"] if step["status"] == "executed"]
        skipped = [step for step in result["steps"] if step["status"] == "skipped"]
        print(f"评估完成：{len(executed)} 个步骤执行，{len(skipped)} 个步骤跳过")
        for step in skipped:
            print(f"  跳过 {step['step']}：{step['detail']}")
        print(f"产物目录：{Path(result['output_dir']).resolve()}")
    elif args.command == "report":
        print(f"报告已生成：{result['report']}")
    else:
        print(f"{args.command} 完成；产物目录：{getattr(args, 'output_dir', None) or args.evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
