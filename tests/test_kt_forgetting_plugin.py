"""kt-forgetting 训练模型适配器的测试。

分三层：

1. **纯函数层**：时间轴推导与间隔校验，不需要训练仓库，任何环境下都跑。
2. **口径一致性层**：把推导结果与训练仓库自己的 ``load_history_json()`` 逐项比对，
   这是本适配器最关键的正确性保证。
3. **适配器层**：真实加载训练仓库的模型跑通契约，需要仓库存在且装了 torch。

缺少训练仓库或 torch 时，第三层自动跳过并给出原因，不会伪装成通过。
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from contracts import Interaction, ModelValidationError, PredictionRequest, Target
from examples.kt_forgetting_plugin import (
    KtForgettingModel,
    build_history_arrays,
    history_skills_and_corrects,
    target_gap_seconds,
)
from mock_model import MockModel
from model_adapter import ModelAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("KT_FORGETTING_REPO", PROJECT_ROOT.parent / "kt-forgetting")).expanduser()
BASE_TIME = 1_700_000_000.0
DAY = 86400.0

#: 需要一个可运行的训练仓库 + torch 的测试。
requires_repo = pytest.mark.skipif(
    not REPO.is_dir(), reason=f"未找到 kt-forgetting 仓库：{REPO}（可用 KT_FORGETTING_REPO 指定）"
)


def interaction(index: int, skill: str, correct: int, stamp: float) -> Interaction:
    return Interaction(
        record_id=f"h{index}", student_id="U", question_id=f"Q{index}",
        skill_tokens=(skill,), skill_ids=(index,), correct=correct,
        timestamp=stamp, available_at=stamp, event_order=None, response_time_ms=None,
        interaction_group="", source_row=index + 2,
    )


def history(skills, corrects, gaps):
    """按相邻间隔（秒）构造历史，gaps[0] 被忽略（首条从 0 起算）。"""
    stamps, clock = [], 0.0
    for index, gap in enumerate(gaps):
        clock = clock + gap if index else 0.0
        stamps.append(BASE_TIME + clock)
    return tuple(
        interaction(index, skill, correct, stamps[index])
        for index, (skill, correct) in enumerate(zip(skills, corrects, strict=True))
    )


def request(items, target_skill="S000", gap=DAY, record_id="t1"):
    return PredictionRequest(
        target=Target(record_id, "U", "QT", (target_skill,), (0,), BASE_TIME, None, "timestamp"),
        history=items,
        delta_time_seconds=gap,
    )


def identity_resolver(tokens):
    return int(str(tokens[0]).lstrip("S"))


def make_run_dir(tmp_path, *, with_meta=True, with_weights=False, n_skills=6):
    """造一个最小的实验目录，用来测「缺权重」「缺词表」等分支，不依赖真实仓库内容。"""
    run = tmp_path / "exp" / "synthetic_tag"
    run.mkdir(parents=True)
    args = {
        "data": "data/processed/demo/data.npz", "model": "dkt", "d_model": 16, "hidden": 16,
        "n_heads": 2, "dropout": 0.0, "delta_unit": "day", "no_forgetting": False,
        "architecture_version": 2, "time_features": False,
    }
    (run / "metrics.json").write_text(
        json.dumps({"args": args}, ensure_ascii=False), encoding="utf-8"
    )
    if with_meta:
        (run / "meta.json").write_text(
            json.dumps({"n_skills": n_skills,
                        "skill2id": {f"S{i:03d}": i for i in range(n_skills)}}),
            encoding="utf-8",
        )
    if with_weights:
        (run / "model.pt").write_bytes(b"not a real checkpoint")
    return run


# --------------------------------------------------------------- 1. 纯函数层
def test_deltas_are_tracked_per_skill_on_the_global_clock():
    items = history(["S000", "S001", "S000"], [1, 0, 1], [0, 100, 200])
    skills, corrects, deltas, dt_globals = build_history_arrays(items, identity_resolver)
    assert skills == [0, 1, 0]
    assert corrects == [1, 0, 1]
    assert dt_globals == [0.0, 100.0, 200.0], "首条必须为 0，其后是相邻可用时点之差"
    assert deltas == [0.0, 0.0, 300.0], "首次出现为 0；再次出现是距上次同技能练习的时钟差"


def test_first_record_global_gap_is_always_zero():
    items = history(["S000", "S001"], [1, 0], [999, 50])
    _, _, _, dt_globals = build_history_arrays(items, identity_resolver)
    assert dt_globals[0] == 0.0


def test_history_without_timestamps_cannot_be_derived():
    items = tuple(
        Interaction(record_id=f"h{i}", student_id="U", question_id="Q", skill_tokens=("S000",),
                    skill_ids=(0,), correct=1, timestamp=None, available_at=None,
                    event_order=i, response_time_ms=None, interaction_group="", source_row=i + 2)
        for i in range(3)
    )
    with pytest.raises(ModelValidationError, match="时间戳"):
        build_history_arrays(items, identity_resolver)


def test_unsorted_history_is_rejected():
    items = (
        interaction(0, "S000", 1, BASE_TIME + 1000),
        interaction(1, "S001", 0, BASE_TIME),
    )
    with pytest.raises(ModelValidationError, match="升序|因果顺序"):
        build_history_arrays(items, identity_resolver)


def test_bkt_path_needs_no_timestamps():
    """BKT 不使用时间，因此仅 event_order 的历史也能取出技能与标签。"""
    items = tuple(
        Interaction(record_id=f"h{i}", student_id="U", question_id="Q", skill_tokens=("S000",),
                    skill_ids=(0,), correct=i % 2, timestamp=None, available_at=None,
                    event_order=i, response_time_ms=None, interaction_group="", source_row=i + 2)
        for i in range(3)
    )
    skills, corrects = history_skills_and_corrects(items, identity_resolver)
    assert skills == [0, 0, 0]
    assert corrects == [0, 1, 0]


def test_cold_start_gap_is_zero():
    assert target_gap_seconds(request((), gap=None), allow_order_only_time=False) == 0.0


def test_order_only_history_is_rejected_by_default():
    """有历史却没有真实时间时必须失败，不能把事件顺序当秒数。"""
    items = (
        Interaction(record_id="h0", student_id="U", question_id="Q", skill_tokens=("S000",),
                    skill_ids=(0,), correct=1, timestamp=None, available_at=None, event_order=1,
                    response_time_ms=None, interaction_group="", source_row=2),
    )
    with pytest.raises(ModelValidationError, match="event_order|真实时间"):
        target_gap_seconds(request(items, gap=None), allow_order_only_time=False)


def test_order_only_history_allowed_only_when_explicitly_enabled():
    items = (
        Interaction(record_id="h0", student_id="U", question_id="Q", skill_tokens=("S000",),
                    skill_ids=(0,), correct=1, timestamp=None, available_at=None, event_order=1,
                    response_time_ms=None, interaction_group="", source_row=2),
    )
    assert target_gap_seconds(request(items, gap=None), allow_order_only_time=True) == 0.0


@pytest.mark.parametrize("gap", [-1.0, float("nan"), float("inf")])
def test_illegal_gap_is_rejected(gap):
    with pytest.raises(ModelValidationError, match="非负有限数"):
        target_gap_seconds(request((), gap=gap), allow_order_only_time=False)


# ------------------------------------------------------- 词表与配置生成（纯函数）
def test_vocab_and_skill_map_come_from_the_same_meta():
    """词表从 1 起、模型 id 从 0 起，但两者必须指向同一批知识点。"""
    from scripts.build_kt_forgetting_vocab import build_mappings

    meta = {"n_skills": 3, "skill2id": {"B": 1, "A": 0, "C": 2}}
    vocab, skill_map = build_mappings(meta)
    assert vocab == {"A": 1, "B": 2, "C": 3}
    assert skill_map == {"A": 0, "B": 1, "C": 2}
    assert sorted(vocab) == sorted(skill_map)


def test_vocab_builder_rejects_inconsistent_meta():
    from scripts.build_kt_forgetting_vocab import build_mappings

    with pytest.raises(SystemExit, match="n_skills"):
        build_mappings({"n_skills": 3, "skill2id": {"A": 0, "B": 1}})
    with pytest.raises(SystemExit, match="连续"):
        build_mappings({"n_skills": 3, "skill2id": {"A": 0, "B": 2, "C": 3}})


def test_shipped_mooccube_vocab_and_config_are_consistent():
    """交付的 MOOCCube 词表与适配器配置必须一一对应，否则模型 id 会静默错位。"""
    examples = PROJECT_ROOT / "examples"
    vocab = json.loads((examples / "kt_forgetting_mooccube_vocab.json").read_text(encoding="utf-8"))
    config = json.loads((examples / "kt_forgetting_mooccube_config.json").read_text(encoding="utf-8"))
    assert vocab["schema_version"] == "1.0" and vocab["unknown_id"] == 0
    assert set(vocab["skills"]) == set(config["skill_map"])
    assert len(vocab["skills"]) == config["n_skills"] == 22
    assert sorted(vocab["skills"].values()) == list(range(1, 23))
    assert sorted(config["skill_map"].values()) == list(range(22))


# ------------------------------------------------------- 2. 与训练仓库口径一致
@requires_repo
def test_matches_training_repo_history_loader(tmp_path):
    """本适配器推导的四个数组必须与训练仓库自己的 load_history_json 完全一致。"""
    sys_path_added = str(REPO)
    import sys

    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)
    from src.eval.predict_mastery import load_history_json

    rows = [
        {"skill": "S000", "correct": 1, "dt_global_sec": 0},
        {"skill": "S001", "correct": 0, "dt_global_sec": 86400},
        {"skill": "S000", "correct": 0, "dt_global_sec": 259200},
        {"skill": "S002", "correct": 1, "dt_global_sec": 604800},
        {"skill": "S000", "correct": 1, "dt_global_sec": 1209600},
    ]
    meta = {
        "n_skills": 30,
        "skill2id": {f"S{i:03d}": i for i in range(30)},
        "id2skill_name": {str(i): f"S{i:03d}" for i in range(30)},
    }
    source = tmp_path / "history.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    reference = load_history_json(str(source), meta)

    items = history(
        [row["skill"] for row in rows], [row["correct"] for row in rows],
        [row["dt_global_sec"] for row in rows],
    )
    mine = build_history_arrays(items, identity_resolver)
    assert tuple(mine[0]) == tuple(reference[0]), "skills 不一致"
    assert tuple(mine[1]) == tuple(reference[1]), "corrects 不一致"
    assert tuple(mine[2]) == tuple(reference[2]), "deltas 不一致"
    assert tuple(mine[3]) == tuple(reference[3]), "dt_globals 不一致"


# --------------------------------------------------------------- 3. 适配器层
def test_missing_repo_field_is_rejected():
    with pytest.raises(ModelValidationError, match="repo"):
        KtForgettingModel({"tag": "bkt"})


def test_nonexistent_repo_is_rejected(tmp_path):
    with pytest.raises(ModelValidationError, match="repo 目录不存在"):
        KtForgettingModel({"repo": str(tmp_path / "nope"), "tag": "bkt"})


def test_missing_tag_is_rejected(tmp_path):
    # 用一个确实存在的目录，才能在缺仓库的环境里也走到 tag 校验而不是先被 repo 校验拦下。
    with pytest.raises(ModelValidationError, match="tag"):
        KtForgettingModel({"repo": str(tmp_path)})


@requires_repo
def test_bad_options_are_rejected():
    base = {"repo": str(REPO), "tag": "bkt", "backend": "bkt",
            "skill_map": {"S000": 0}}
    with pytest.raises(ModelValidationError, match="multi_skill_policy"):
        KtForgettingModel({**base, "multi_skill_policy": "median"})
    with pytest.raises(ModelValidationError, match="backend"):
        KtForgettingModel({**base, "backend": "transformer"})
    with pytest.raises(ModelValidationError, match="threads"):
        KtForgettingModel({**base, "threads": 0})
    with pytest.raises(ModelValidationError, match="n_skills"):
        KtForgettingModel({**base, "n_skills": 0})
    with pytest.raises(ModelValidationError, match="n_skills"):
        KtForgettingModel({**base, "skill_map": {"S000": 0, "S001": 5}, "n_skills": 2})
    with pytest.raises(ModelValidationError, match="实验目录不存在"):
        KtForgettingModel({**base, "tag": "does_not_exist"})


@requires_repo
def test_missing_weights_explains_how_to_get_them(tmp_path):
    """没有 model.pt 的实验目录，必须明确告诉用户怎么拿到权重。"""
    make_run_dir(tmp_path)
    config = {
        "repo": str(REPO), "tag": "synthetic_tag", "root": str(tmp_path / "exp"),
        "backend": "dkt", "skill_map": {"S000": 0},
    }
    with pytest.raises(ModelValidationError, match="package_model|model.pt"):
        KtForgettingModel(config)


@requires_repo
def test_untrained_flag_forces_mock_marking(tmp_path):
    """随机初始化的模型不是训练成果，必须被标记为模拟。"""
    make_run_dir(tmp_path)
    config = {
        "repo": str(REPO), "tag": "synthetic_tag", "root": str(tmp_path / "exp"),
        "backend": "dkt", "allow_untrained": True,
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    }
    model = KtForgettingModel(config)
    assert model.is_mock is True
    assert "(untrained)" in model.name
    probability = model.predict([request((), record_id="u1")])[0].probability
    assert 0.0 <= probability <= 1.0


@requires_repo
def test_weight_directory_without_meta_needs_skill_map(tmp_path):
    """有真实权重但没有 meta.json 时，必须由调用方补 skill_map。"""
    make_run_dir(tmp_path, with_meta=False, with_weights=True)
    base = {"repo": str(REPO), "tag": "synthetic_tag", "root": str(tmp_path / "exp"),
            "backend": "dkt"}
    with pytest.raises(ModelValidationError, match="skill_map"):
        KtForgettingModel(base)


@requires_repo
def test_vocabulary_size_is_inferred_from_the_checkpoint():
    """akt_* 没有 meta.json，但权重里的 skill_emb 形状能确定词表大小。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "akt_forget_s2026", "backend": "dkt",
        "skill_map": {"S000": 0},
    })
    assert model.n_skills == 149
    assert model.is_mock is False
    assert "(untrained)" not in model.name
    probability = model.predict([request((), record_id="a1")])[0].probability
    assert 0.0 <= probability <= 1.0


@requires_repo
def test_declared_n_skills_must_match_the_checkpoint():
    """配置写了 n_skills 就与权重比对，不一致必须报错而不是照单全收。"""
    with pytest.raises(ModelValidationError, match="不一致"):
        KtForgettingModel({
            "repo": str(REPO), "tag": "akt_forget_s2026", "backend": "dkt",
            "skill_map": {"S000": 0}, "n_skills": 30,
        })


@requires_repo
def test_skill_map_beyond_the_vocabulary_is_rejected():
    with pytest.raises(ModelValidationError, match="超出词表范围"):
        KtForgettingModel({
            "repo": str(REPO), "tag": "v4_controlled_dkt_forget_s2026", "backend": "dkt",
            "skill_map": {"S000": 0, "S999": 40},
        })


@requires_repo
def test_real_weights_are_reported_as_real_model():
    """v3/v4 系列已随仓库交付 model.pt，加载到权重就是真实模型。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "v4_controlled_dkt_forget_s2026", "backend": "dkt",
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    })
    assert model.is_mock is False
    assert "(untrained)" not in model.name
    probability = model.predict([request((), record_id="r1")])[0].probability
    assert 0.0 <= probability <= 1.0


@requires_repo
def test_mooccube_model_loads_its_own_vocabulary():
    """MOOCCube 中文线代模型：词表来自它自己的 meta.json，可直接按中文标签解析。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "v4_mooccube_la_dkt_forget_s2026", "backend": "dkt",
    })
    assert model.is_mock is False
    assert int(model._meta["n_skills"]) == 22
    token = sorted(model._meta["skill2id"], key=lambda key: model._meta["skill2id"][key])[0]
    assert model.resolve_skill((token,)) == 0


@requires_repo
def test_bkt_backend_reports_real_model():
    """BKT 参数是训练组在真实数据上 EM 拟合的成果，不是模拟模型。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    })
    assert model.is_mock is False
    assert model.name == "kt-forgetting:bkt:bkt"


@requires_repo
def test_bkt_without_meta_requires_skill_map():
    """训练仓库的 exp/bkt/ 不带 meta.json，没有 skill_map 就无法映射输入。"""
    with pytest.raises(ModelValidationError, match="skill_map|meta.json"):
        KtForgettingModel({"repo": str(REPO), "tag": "bkt", "backend": "bkt"})


@requires_repo
def test_skill_resolution_and_rejection():
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {"S000": 0, "S001": 1},
    })
    assert model.resolve_skill(("S000",)) == 0
    assert model.resolve_skill(("S001",)) == 1
    with pytest.raises(ModelValidationError, match="不在 skill_map"):
        model.resolve_skill(("S999",))
    with pytest.raises(ModelValidationError, match="没有知识点标签"):
        model.resolve_skill(())


@requires_repo
def test_multi_skill_target_is_rejected_by_default():
    """训练仓库未定义多知识点聚合口径，适配器不得擅自取第一个。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {"S000": 0, "S001": 1},
    })
    with pytest.raises(ModelValidationError, match="多个知识点|聚合"):
        model.resolve_skill(("S000", "S001"))


@requires_repo
def test_multi_skill_first_policy_is_opt_in():
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {"S000": 0, "S001": 1}, "multi_skill_policy": "first",
    })
    assert model.resolve_skill(("S000", "S001")) == 0


@requires_repo
def test_bkt_predicts_one_probability_per_request():
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    })
    items = history(["S000", "S001"], [1, 0], [0, DAY])
    requests = [
        request(items, target_skill="S000", record_id="a"),
        request(items, target_skill="S002", record_id="b"),
    ]
    predictions = model.predict(requests)
    assert [p.record_id for p in predictions] == ["a", "b"]
    assert all(0.0 <= p.probability <= 1.0 for p in predictions)
    # 真实 BKT 对同一历史下不同知识点应给出不同概率
    assert predictions[0].probability != predictions[1].probability


@requires_repo
def test_bkt_survives_order_only_data():
    """BKT 不使用时间，因此仅 event_order 的数据也能正常推理。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    })
    items = tuple(
        Interaction(record_id=f"h{i}", student_id="U", question_id="Q", skill_tokens=("S000",),
                    skill_ids=(0,), correct=i % 2, timestamp=None, available_at=None,
                    event_order=i, response_time_ms=None, interaction_group="", source_row=i + 2)
        for i in range(4)
    )
    request_object = PredictionRequest(
        target=Target("t", "U", "QT", ("S001",), (1,), None, 9, "order"),
        history=items, delta_time_seconds=None,
    )
    probability = model.predict([request_object])[0].probability
    assert 0.0 <= probability <= 1.0


@requires_repo
def test_arrays_are_passed_to_the_training_repo_verbatim():
    """用记录器断言适配器送进训练仓库的四个数组与间隔，不依赖模型是否敏感。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "repro_v2_controlled_dkt_forget_s2026", "backend": "dkt",
        "allow_untrained": True, "skill_map": {f"S{i:03d}": i for i in range(30)},
    })
    captured: dict = {}
    original = model._predict_mastery.predict_next

    def recorder(inner_model, skills, corrects, deltas, dt_globals, next_skill, gap_sec):
        captured.update(skills=list(skills), corrects=list(corrects), deltas=list(deltas),
                        dt_globals=list(dt_globals), next_skill=next_skill, gap_sec=gap_sec)
        return original(inner_model, skills, corrects, deltas, dt_globals,
                        next_skill=next_skill, gap_sec=gap_sec)

    model._predict_mastery.predict_next = recorder
    try:
        items = history(["S000", "S001", "S000"], [1, 0, 1], [0, DAY, 2 * DAY])
        model.predict([request(items, target_skill="S001", gap=7 * DAY)])
    finally:
        model._predict_mastery.predict_next = original

    assert captured["skills"] == [0, 1, 0]
    assert captured["corrects"] == [1, 0, 1]
    assert captured["dt_globals"] == [0.0, DAY, 2 * DAY]
    assert captured["deltas"] == [0.0, 0.0, 3 * DAY], "第三次是 S000 距上次出现 3 天"
    assert captured["next_skill"] == 1
    assert captured["gap_sec"] == 7 * DAY, "目标间隔必须原样传成 gap_sec"


@requires_repo
def test_output_satisfies_the_framework_contract():
    """适配器输出必须能通过框架 ModelAdapter 的全部校验。"""
    adapter = ModelAdapter(KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    }))
    items = history(["S000", "S002"], [1, 1], [0, 3 * DAY])
    requests = [
        request(items, target_skill="S001", record_id=f"r{index}") for index in range(3)
    ]
    predictions = adapter.predict(requests)
    assert [p.record_id for p in predictions] == ["r0", "r1", "r2"]
    assert all(isinstance(p.probability, float) for p in predictions)
    assert adapter.model.is_mock is False


def test_mock_model_is_unaffected_by_this_adapter():
    """确认新增适配器没有改变框架原有模拟模型的行为。"""
    adapter = ModelAdapter(MockModel())
    assert adapter.model.is_mock is True
    predictions = adapter.predict([request((), record_id="m1")])
    assert predictions[0].probability == 0.5


@requires_repo
def test_cold_start_and_warm_start_differ_for_bkt():
    """BKT 在无历史时用先验 L0，有历史后状态被更新，概率应当变化。"""
    model = KtForgettingModel({
        "repo": str(REPO), "tag": "bkt", "backend": "bkt",
        "skill_map": {f"S{i:03d}": i for i in range(6)},
    })
    cold = model.predict([request((), target_skill="S000", gap=None, record_id="c")])[0]
    warm_items = history(["S000", "S000"], [1, 1], [0, DAY])
    warm = model.predict([request(warm_items, target_skill="S000", record_id="w")])[0]
    assert cold.probability != warm.probability


def test_prediction_request_is_immutable():
    """适配器不得修改请求对象（契约要求请求不可变）。"""
    original = request(history(["S000"], [1], [0]))
    snapshot = replace(original)
    assert original == snapshot
