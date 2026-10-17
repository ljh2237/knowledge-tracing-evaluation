"""从 ``kt-forgetting`` 的 ``meta.json`` 生成框架词表与适配器配置。

训练仓库每个 ``exp/<tag>/meta.json`` 里都有 ``skill2id``（原始知识点 → 模型密集 id）
与 ``n_skills``。本框架要的是两样东西：

1. **词表** ``vocab.json``：``{原始知识点: 正整数}``，供 ``main.py --vocab`` 校验输入，
   编号从 1 连续递增（``data_loader.SkillVocabulary`` 的硬性要求）；
2. **适配器配置** ``model_config.json``：其中 ``skill_map`` 把同一个原始知识点
   映射到模型密集 id（从 0 开始），``n_skills`` 是词表大小。

两者必须来自同一份 ``meta.json``，否则会出现「输入编码对上了、模型 id 却错位」的
静默错误。本脚本把它们一次生成，避免手工抄写。

可选 ``--fixture`` 会额外生成一份**合成**作答 CSV（``skill_id`` 使用模型真实词表），
用于在没有真实数据时打通端到端通路。合成数据的说明会写进产物文件名与文档，
**不能用于任何精度结论**。

用法::

    python scripts/build_kt_forgetting_vocab.py \
        --meta ../kt-forgetting/exp/v4_mooccube_la_dkt_forget_s2026/meta.json \
        --out-dir examples

    # 同时生成一份使用该词表的合成夹具
    python scripts/build_kt_forgetting_vocab.py \
        --meta ../kt-forgetting/exp/v4_mooccube_la_dkt_forget_s2026/meta.json \
        --out-dir examples --fixture examples/kt_forgetting_mooccube_fixture.csv
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "1.0"
BASE_TIME = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)


def load_meta(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} 必须是 JSON 对象")
    if "skill2id" not in payload or "n_skills" not in payload:
        raise SystemExit(
            f"{path} 缺少 skill2id 或 n_skills。"
            "BKT 等部分实验目录不写 meta.json，此时需手工提供知识点映射"
        )
    return payload


def build_mappings(meta: dict) -> tuple[dict[str, int], dict[str, int]]:
    """返回 ``(vocab_mapping, skill_map)``，两者键相同、值分别从 1 / 0 起。"""
    n_skills = int(meta["n_skills"])
    skill2id = {str(k): int(v) for k, v in meta["skill2id"].items()}
    if len(skill2id) != n_skills:
        raise SystemExit(
            f"skill2id 有 {len(skill2id)} 项，但 n_skills={n_skills}；词表与模型规模不一致"
        )
    if sorted(skill2id.values()) != list(range(n_skills)):
        raise SystemExit("skill2id 的密集 id 必须恰好覆盖 0..n_skills-1，当前不连续或重复")
    tokens = sorted(skill2id, key=lambda token: skill2id[token])
    vocab_mapping = {token: index + 1 for index, token in enumerate(tokens)}
    skill_map = {token: skill2id[token] for token in tokens}
    return vocab_mapping, skill_map


def build_config(meta: dict, skill_map: dict[str, int], args) -> dict:
    config = {
        "repo": args.repo,
        "tag": args.tag,
        "root": "exp",
        "backend": "dkt",
        "n_skills": int(meta["n_skills"]),
        "skill_map": skill_map,
        "multi_skill_policy": "error",
        "allow_order_only_time": False,
        "threads": 2,
    }
    if args.calibration:
        config["calibration"] = args.calibration
    return config


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def emit_fixture(path: Path, tokens: list[str], *, students: int, seed: int) -> None:
    """生成一份**合成**作答 CSV，``skill_id`` 取自模型真实词表。

    时间是虚构的，标签由固定随机种子生成；这只用于验证端到端通路，
    不代表任何真实学习者的表现，也不得用于精度结论。
    """
    rng = random.Random(seed)
    rows: list[tuple[str, str, str, str, str, str]] = []
    for student in range(students):
        student_id = f"MC{student:02d}"
        # 每位学生练 8 个知识点，每个知识点 3 次，间隔逐次拉长（1/3/7/16 天）
        chosen = rng.sample(tokens, k=min(8, len(tokens)))
        stamp = BASE_TIME + timedelta(hours=rng.randint(0, 20))
        mastery = 0.22 + 0.10 * rng.random()
        for round_index, gap_days in enumerate((1, 3, 7)):
            for slot, token in enumerate(chosen):
                stamp = stamp + timedelta(days=gap_days, hours=rng.randint(1, 10))
                mastery = min(0.86, mastery + 0.035)
                correct = 1 if rng.random() < mastery else 0
                rows.append((
                    f"mc-s{student:02d}-r{round_index}-{slot:02d}",
                    student_id,
                    f"Q{round_index}-{slot:02d}",
                    token,
                    str(correct),
                    stamp.isoformat(),
                ))
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "student_id", "question_id", "skill_id", "correct", "timestamp"])
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成 kt-forgetting 词表与适配器配置")
    parser.add_argument("--meta", required=True, help="exp/<tag>/meta.json 路径")
    parser.add_argument("--tag", default=None, help="实验 tag（默认取 meta.json 所在目录名）")
    parser.add_argument("--repo", default="../kt-forgetting", help="写进配置的仓库路径")
    parser.add_argument("--out-dir", default="examples", help="输出目录")
    parser.add_argument("--prefix", default=None, help="输出文件名前缀（默认 kt_forgetting）")
    parser.add_argument("--calibration", default=None, help="温度校准 JSON 路径（可选）")
    parser.add_argument("--fixture", default=None, help="额外生成合成作答 CSV 的路径")
    parser.add_argument("--students", type=int, default=6, help="合成夹具的学生数")
    parser.add_argument("--seed", type=int, default=2026, help="合成夹具的随机种子")
    args = parser.parse_args(argv)

    meta_path = Path(args.meta)
    meta = load_meta(meta_path)
    if args.tag is None:
        args.tag = meta_path.parent.name
    vocab_mapping, skill_map = build_mappings(meta)

    prefix = args.prefix or "kt_forgetting"
    out_dir = Path(args.out_dir)
    vocab_path = out_dir / f"{prefix}_vocab.json"
    config_path = out_dir / f"{prefix}_config.json"
    write_json(vocab_path, {"schema_version": SCHEMA_VERSION, "unknown_id": 0, "skills": vocab_mapping})
    write_json(config_path, build_config(meta, skill_map, args))

    tokens = sorted(skill_map, key=lambda token: skill_map[token])
    print(f"词表大小 {len(skill_map)}，密集 id 0..{len(skill_map) - 1}")
    print(f"  词表   -> {vocab_path}")
    print(f"  配置   -> {config_path}")
    if args.fixture:
        if args.students < 1:
            raise SystemExit("--students 必须为正整数")
        emit_fixture(Path(args.fixture), tokens, students=args.students, seed=args.seed)
        print(f"  合成夹具 -> {args.fixture}（虚构数据，仅用于通路验证）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
