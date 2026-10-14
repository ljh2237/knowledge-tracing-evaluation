"""Run python main.py from a clean checkout to produce mock predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contracts import DataValidationError, ModelValidationError, VERSION
from data_loader import SkillVocabulary
from io_utils import sha256_file
from mock_model import MockModel
from model_adapter import ModelAdapter
from pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="鲁凯宁知识追踪测试框架：CSV → 因果历史 → 模型 → 预测 CSV")
    result.add_argument("--version", action="version", version=VERSION)
    result.add_argument("--input", type=Path, default=ROOT / "sample_data.csv")
    result.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--max-history", type=int, default=256)
    result.add_argument("--threshold", type=float, default=0.5)
    result.add_argument("--invalid-policy", choices=["error", "drop"], default="error")
    result.add_argument("--timestamp-unit", choices=["iso", "s", "ms"], default="iso")
    result.add_argument("--vocab", type=Path, help="训练组提供的固定知识点词表")
    result.add_argument("--unknown-skill-policy", choices=["error", "unk"], default="error")
    result.add_argument("--model", choices=["mock", "plugin"], default="mock")
    result.add_argument("--factory", help="外部模型工厂 module:function")
    result.add_argument("--model-config", type=Path, help="传给模型工厂的 JSON 配置文件")
    result.add_argument("--overwrite", action="store_true", help="成功后整体替换已有输出目录；失败保留旧结果")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        vocabulary = SkillVocabulary.load(args.vocab) if args.vocab else None
        if args.model == "plugin":
            if not args.factory or vocabulary is None:
                raise ModelValidationError("plugin 模式必须提供 --factory 和训练组 --vocab")
            config = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else {}
            if not isinstance(config, dict):
                raise ModelValidationError("模型配置必须为 JSON 对象")
            adapter = ModelAdapter.from_factory(args.factory, config)
        else:
            if args.factory or args.model_config:
                raise ModelValidationError("--factory / --model-config 仅用于 --model plugin")
            adapter = ModelAdapter(MockModel())
        summary = run_pipeline(
            args.input, args.output_dir, adapter, batch_size=args.batch_size,
            max_history=args.max_history, invalid_policy=args.invalid_policy,
            timestamp_unit=args.timestamp_unit, vocabulary=vocabulary,
            unknown_skill_policy=args.unknown_skill_policy, threshold=args.threshold,
            overwrite=args.overwrite,
            model_provenance={
                "factory": args.factory if args.model == "plugin" else "mock_model:MockModel",
                "config_sha256": sha256_file(args.model_config) if args.model_config else None,
                "training_vocab_sha256": sha256_file(args.vocab) if args.vocab else None,
            },
        )
        kind = "模拟模型（仅用于流程验证）" if summary["is_mock"] else "外部模型"
        print(f"成功：{kind}；{summary['data']['students']} 位学生，{summary['prediction_count']} 条预测")
        print(f"结果：{args.output_dir.resolve() / 'predictions.csv'}")
        print(f"清洗拒收：{summary['data']['rejected_rows']} 条；运行摘要：run_summary.json")
        return 0
    except (DataValidationError, ModelValidationError, ValueError, OSError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
