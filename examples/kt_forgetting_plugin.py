"""把训练仓库 ``kt-forgetting`` 的知识追踪模型接入本项目 ``PredictModel`` 契约。

**这是适配层，不修改 ``contracts.py``，也不修改训练仓库的任何一行代码。**
训练仓库暴露的是 CLI 与函数式接口，本项目要求的是「可调用对象 + 请求批次」，
两者的差异全部在本文件内消化。

接口对照
--------

============================  ============================================  ==========================
维度                          训练仓库 kt-forgetting                         本项目契约
============================  ============================================  ==========================
输出语义                      ``predicted_response_probability``（答对概率）    ``y_pred``（答对概率）
调用形态                      ``predict_next(model, ..., gap_sec)``            ``model.predict(requests)``
知识点编号                    ``meta.json`` 的密集 id（``skill2id``）           框架 ``SkillVocabulary``
时间特征                      历史逐步的 ``delta`` + ``dt_global`` 两个数组     仅目标 ``delta_time_seconds``
目标间隔语义                  「最后一次交互后的额外等待」                       ``delta_time_seconds`` 同义
============================  ============================================  ==========================

**输出语义是一致的**：训练仓库在 ``predict_mastery.py`` 中显式声明
``score_semantics = "predicted_response_probability"``，并打印
「预测答对概率，非真实掌握度」。因此本适配器直接把它作为 ``y_pred``，
不存在「把掌握度当答对概率」的问题。

时间轴推导
----------

带遗忘机制的后端要求历史**每一步**两个数组，且训练仓库
``src/models/inference.py`` 明确警告 ``delta`` 与 ``dt_global`` 不可互相替代：

- ``dt_global[i]`` = 第 i 步距第 i-1 步的时间间隔，**首条必须为 0**；
- ``delta[i]``    = 第 i 步距该步知识点上次出现的时间间隔，首次出现为 0。

本项目契约只提供目标题的 ``delta_time_seconds``，因此历史时间轴由本适配器从
``Interaction.available_at``（答案可见时点）推导，口径与训练仓库
``load_history_json()`` 完全一致。

**仅有 ``event_order`` 的数据无法驱动带遗忘的后端**：没有真实时间轴就推不出
``dt_global``。默认直接失败，不把事件顺序当作秒数使用；确需无时间对照实验时，
必须显式设置 ``allow_order_only_time``，此时所有间隔为 0，模型退化为无时间基线。
BKT 后端不使用时间，因此不受此限制。
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from contracts import ModelValidationError, Prediction, PredictionRequest

#: 多知识点题目的聚合策略。训练仓库未定义聚合口径，因此默认拒绝而不是擅自取第一个。
MULTI_SKILL_POLICIES = ("error", "mean", "first")

#: 后端：auto 按 tag 目录里是否存在 bkt_params.json 自动判断。
BACKENDS = ("auto", "dkt", "bkt")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelValidationError(message)


# --------------------------------------------------------------------------- #
# 纯函数：请求 → 训练仓库所需数组（不依赖 torch，可单独测试）
# --------------------------------------------------------------------------- #
def history_skills_and_corrects(
    history: Sequence, resolve_skill
) -> tuple[list[int], list[int]]:
    """把历史转成模型密集 id 与真实标签；BKT 后端只用这两个。"""
    skills: list[int] = []
    corrects: list[int] = []
    for index, item in enumerate(history):
        _require(
            0 <= int(item.correct) <= 1,
            f"历史第 {index} 条的真实标签必须是 0 或 1，实际为 {item.correct!r}",
        )
        skills.append(resolve_skill(item.skill_tokens))
        corrects.append(int(item.correct))
    return skills, corrects


def build_history_arrays(
    history: Sequence, resolve_skill
) -> tuple[list[int], list[int], list[float], list[float]]:
    """把 ``PredictionRequest.history`` 转成训练仓库要的四个平行数组。

    返回 ``(skills, corrects, deltas, dt_globals)``，与训练仓库
    ``load_history_json()`` 的输出顺序完全一致，口径也一致。
    """
    skills, corrects = history_skills_and_corrects(history, resolve_skill)

    stamps: list[float] = []
    for index, item in enumerate(history):
        stamp = item.available_at if item.available_at is not None else item.timestamp
        if stamp is None or not math.isfinite(float(stamp)):
            raise ModelValidationError(
                f"历史第 {index} 条记录缺少有效时间戳；带遗忘的后端需要真实时间轴推导 "
                "dt_global，不能用事件顺序代替"
            )
        if stamps and float(stamp) < stamps[-1]:
            raise ModelValidationError(
                f"历史必须按时间升序，第 {index} 条早于前一条，违反因果顺序"
            )
        stamps.append(float(stamp))

    dt_globals = [
        0.0 if index == 0 else stamps[index] - stamps[index - 1]
        for index in range(len(stamps))
    ]
    clock = 0.0
    last_seen: dict[int, float] = {}
    deltas: list[float] = []
    for index, gap in enumerate(dt_globals):
        clock += gap
        skill = skills[index]
        deltas.append(clock - last_seen[skill] if skill in last_seen else 0.0)
        last_seen[skill] = clock
    return skills, corrects, deltas, dt_globals


def target_gap_seconds(request: PredictionRequest, *, allow_order_only_time: bool) -> float:
    """目标时点相对历史最后一次作答的额外等待，对应训练仓库的 ``gap_sec``。

    冷启动（无历史）时训练仓库会把 ``elapsed_sec`` 强制为 0，这里保持同样行为。
    有历史却没有真实时间（仅 ``event_order``）时默认失败。
    """
    gap = request.delta_time_seconds
    if gap is None:
        if request.history and not allow_order_only_time:
            raise ModelValidationError(
                "该请求有历史但没有真实时间间隔（输入可能是仅 event_order 的数据）。"
                "带遗忘机制的后端需要真实时间轴，不能用事件顺序代替；"
                "确需无时间对照实验时请显式设置 allow_order_only_time=true"
            )
        return 0.0
    if not math.isfinite(float(gap)) or float(gap) < 0:
        raise ModelValidationError(f"delta_time_seconds 必须是非负有限数，实际为 {gap!r}")
    return float(gap)


# --------------------------------------------------------------------------- #
# 适配器
# --------------------------------------------------------------------------- #
class KtForgettingModel:
    """``kt-forgetting`` 模型的 ``PredictModel`` 适配器。

    ``config`` 字段：

    ==========================  =======  ================================================
    字段                        必填     说明
    ==========================  =======  ================================================
    ``repo``                    是       ``kt-forgetting`` 仓库根目录
    ``tag``                     是       ``exp/<tag>`` 目录名，例如 ``dkt_forget_s2026``
    ``root``                    否       实验根目录，默认 ``exp``
    ``data``                    否       覆盖 metrics.json 记录的数据路径（跨机器迁移）
    ``backend``                 否       ``auto``（默认）/ ``dkt`` / ``bkt``
    ``skill_map``               否       ``{框架知识点标签: 模型密集 id}``，显式映射
    ``n_skills``                否       词表大小；缺 meta.json 时默认从权重推断，填了则校验一致性
    ``multi_skill_policy``      否       ``error``（默认）/ ``mean`` / ``first``
    ``allow_order_only_time``   否       默认 false；true 时把无时间数据当作全零间隔
    ``allow_untrained``         否       默认 false；仅用于没有权重时验证本适配器通路，
                                        启用后 ``is_mock`` 强制为 True
    ``threads``                 否       torch 线程数，默认 2
    ``calibration``             否       温度校准 JSON，须与权重哈希匹配
    ==========================  =======  ================================================
    """

    def __init__(self, config: dict):
        _require(isinstance(config, dict), "模型配置必须是 JSON 对象")
        raw_repo = str(config.get("repo", "")).strip()
        _require(bool(raw_repo), "config 缺少必填字段 repo（kt-forgetting 仓库路径）")
        self.repo = Path(raw_repo).expanduser()
        _require(self.repo.is_dir(), f"repo 目录不存在: {self.repo}")

        self.tag = str(config.get("tag", "")).strip()
        _require(bool(self.tag), "config 缺少必填字段 tag（exp/<tag> 目录名）")
        self.root = str(config.get("root", "exp"))
        self.data_override = config.get("data")
        self.calibration = config.get("calibration")
        self.allow_order_only_time = bool(config.get("allow_order_only_time", False))
        self.allow_untrained = bool(config.get("allow_untrained", False))
        self.threads = int(config.get("threads", 2))
        _require(self.threads >= 1, "threads 必须是正整数")

        self.multi_skill_policy = str(config.get("multi_skill_policy", "error"))
        _require(
            self.multi_skill_policy in MULTI_SKILL_POLICIES,
            f"multi_skill_policy 必须是 {list(MULTI_SKILL_POLICIES)} 之一",
        )
        try:
            self.skill_map = {
                str(key): int(value) for key, value in (config.get("skill_map") or {}).items()
            }
        except (TypeError, ValueError) as exc:
            raise ModelValidationError(f"skill_map 必须是「知识点标签 → 整数 id」的对象: {exc}") from exc

        raw_n_skills = config.get("n_skills")
        if raw_n_skills is None:
            self.n_skills = None
        else:
            try:
                self.n_skills = int(raw_n_skills)
            except (TypeError, ValueError) as exc:
                raise ModelValidationError(f"n_skills 必须是正整数: {exc}") from exc
            _require(self.n_skills >= 1, "n_skills 必须是正整数")
            _require(
                all(0 <= value < self.n_skills for value in self.skill_map.values()),
                f"skill_map 中存在超出 n_skills={self.n_skills} 范围的密集 id",
            )

        self._skill_cache: dict[tuple[str, ...], int] = {}
        self._load_repository()
        self._load_backend(str(config.get("backend", "auto")))

    # ------------------------------------------------------------------ 加载
    def _load_repository(self) -> None:
        """把训练仓库加入导入路径并取出它的推理模块。"""
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        try:
            self._torch = importlib.import_module("torch")
            self._predict_mastery = importlib.import_module("src.eval.predict_mastery")
        except ImportError as exc:
            raise ModelValidationError(
                f"无法从 {self.repo} 导入训练仓库模块（需要 torch）: {exc}"
            ) from exc
        self._torch.set_num_threads(self.threads)

    def _load_backend(self, backend: str) -> None:
        _require(backend in BACKENDS, f"backend 必须是 {list(BACKENDS)} 之一")
        run_dir = self.repo / self.root / self.tag
        _require(run_dir.is_dir(), f"实验目录不存在: {run_dir}")
        if backend == "auto":
            backend = "bkt" if (run_dir / "bkt_params.json").is_file() else "dkt"
        if backend == "bkt":
            self._load_bkt(run_dir)
        else:
            self._load_dkt(run_dir)

    def _load_bkt(self, run_dir: Path) -> None:
        """BKT 的拟合参数是纯 JSON，不需要权重文件，可直接推理。"""
        params_path = run_dir / "bkt_params.json"
        _require(params_path.is_file(), f"缺少 BKT 参数文件: {params_path}")
        bkt_module = importlib.import_module("src.models.bkt")
        self._bkt = bkt_module.BKT()
        raw = json.loads(params_path.read_text(encoding="utf-8"))
        self._bkt.params = {int(key): value for key, value in raw.items()}
        # BKT 参数直接按密集 id 存储，训练仓库的 exp/bkt/ 不带 meta.json；
        # 此时必须由调用方显式提供 skill_map（原始知识点标签 → 密集 id）。
        meta_path = run_dir / "meta.json"
        if meta_path.is_file():
            self._meta = self._read_json(meta_path, required=True)
        elif self.skill_map:
            self._meta = {}
        else:
            raise ModelValidationError(
                f"缺少知识点词表 {meta_path}。BKT 参数按密集 id 存储，需要 meta.json"
                "（原始 skill_id → 密集 id）或显式提供 skill_map 才能把输入数据映射进来"
            )
        self._backend = "bkt"
        self.name = f"kt-forgetting:bkt:{self.tag}"
        # BKT 是训练组在真实数据上用 EM 拟合的真实模型，不是模拟模型。
        self.is_mock = False

    def _load_dkt(self, run_dir: Path) -> None:
        weights = run_dir / "model.pt"
        if not weights.is_file():
            if not self.allow_untrained:
                raise ModelValidationError(
                    f"缺少模型权重 {weights}。该实验目录只有配置与指标，没有权重；"
                    "需要训练组的交接包（src/eval/package_model.py 产出 "
                    "model.pt + metrics.json + meta.json）。"
                    "仅想验证本适配器通路时可显式设置 allow_untrained=true，"
                    "此时模型权重是随机初始化的，is_mock 会被强制为 True。"
                )
            model, run_args, meta = self._build_from_run_dir(run_dir, untrained=True)
        elif (run_dir / "meta.json").is_file() or self.data_override:
            # 词表齐备：走训练仓库自己的加载器，口径与它完全一致。
            try:
                # 训练仓库的 load_model 用 Path(root)/tag，root 是相对路径时会按
                # 当前工作目录解析；这里传绝对路径，避免换目录运行就找不到实验目录。
                model, run_args, meta, _ = self._predict_mastery.load_model(
                    self.tag, str(self.repo / self.root), self.data_override
                )
            except Exception as exc:
                raise ModelValidationError(f"加载 {self.tag} 失败: {exc}") from exc
        else:
            # 少数实验目录只提交了 model.pt 与 metrics.json，没有 meta.json 词表
            # （例如 v1/v2 的 akt_*、dkt_*、synth_*）。权重是真实的，但没有词表就
            # 不知道原始 skill_id 对应哪个密集 id，必须由调用方给出 skill_map；
            # 词表大小可以从权重里的 skill_emb 形状推断，不必手工填写。
            if not self.skill_map:
                raise ModelValidationError(
                    f"{run_dir} 有 model.pt 但缺少 meta.json 词表。请提供 skill_map"
                    "（知识点标签 → 密集 id）；n_skills 可省略，将从权重推断，"
                    "或改用自带 meta.json 的 tag（v3/v4 系列）"
                )
            inferred = self._checkpoint_n_skills(weights)
            if self.n_skills is None:
                self.n_skills = inferred
            else:
                _require(
                    self.n_skills == inferred,
                    f"配置的 n_skills={self.n_skills} 与权重里的词表大小 {inferred} 不一致",
                )
            model, run_args, meta = self._build_from_run_dir(run_dir, untrained=False)

        self._temperature = None
        if self.calibration:
            try:
                self._temperature = self._predict_mastery.load_temperature(
                    self.calibration, weights, self.tag
                )
            except Exception as exc:
                raise ModelValidationError(f"加载温度校准失败: {exc}") from exc
        self._validate_skill_map(int(meta["n_skills"]))
        self._model = model
        self._meta = meta
        self._backend = "dkt"
        suffix = "(untrained)" if self.allow_untrained and not weights.is_file() else ""
        self.name = f"kt-forgetting:{run_args.get('model', 'dkt')}:{self.tag}{suffix}"
        # 只有随机初始化的权重才是模拟模型；训练组的权重是真实训练成果。
        self.is_mock = bool(suffix)

    def _build_from_run_dir(
        self, run_dir: Path, *, untrained: bool
    ) -> tuple[object, dict, dict]:
        """按 metrics.json 记录的配置重建模型结构。

        ``untrained=True`` 时不加载权重（随机初始化，仅供通路验证）；
        ``untrained=False`` 时加载 ``model.pt`` 的真实权重。
        """
        metrics = self._read_json(run_dir / "metrics.json", required=True)
        run_args = metrics.get("args")
        _require(isinstance(run_args, dict), f"{run_dir / 'metrics.json'} 缺少 args 记录")
        meta_path = run_dir / "meta.json"
        if meta_path.is_file():
            meta = self._read_json(meta_path, required=True)
            n_skills = int(meta["n_skills"])
        elif self.skill_map and self.n_skills:
            n_skills = int(self.n_skills)
            meta = {
                "n_skills": n_skills,
                "skill2id": {name: sid for name, sid in self.skill_map.items()},
                "id2skill_name": {str(sid): name for name, sid in self.skill_map.items()},
                "vocabulary_source": "config.skill_map",
            }
        else:
            raise ModelValidationError(
                f"{run_dir} 缺少 meta.json 词表，且配置未提供 skill_map 与 n_skills"
            )

        dkt_module = importlib.import_module("src.models.dkt")
        units = self._predict_mastery.DELTA_UNITS
        delta_unit = run_args.get("delta_unit", "min")
        _require(delta_unit in units, f"未知的 delta_unit: {delta_unit!r}")
        kwargs = dict(
            use_forgetting=not run_args.get("no_forgetting", False),
            dropout=run_args.get("dropout", 0.0),
            delta_scale=units[delta_unit],
            architecture_version=run_args.get("architecture_version", 1),
            slot_dim=run_args.get("slot_dim", 16),
            learn_gain=run_args.get("learn_gain", False),
            time_features=run_args.get("time_features", False),
        )
        model_name = run_args.get("model", "dkt")
        if model_name == "dkt":
            kwargs.update(d_model=run_args.get("d_model", 64), hidden=run_args.get("hidden", 64))
        else:
            kwargs.update(d_model=run_args.get("d_model", 64), n_heads=run_args.get("n_heads", 2))
        model = dkt_module.build_model(model_name, n_skills, **kwargs)
        if untrained:
            model.eval()
        else:
            try:
                state = self._torch.load(run_dir / "model.pt", map_location="cpu",
                                         weights_only=True)
                model.load_state_dict(state)
            except Exception as exc:
                raise ModelValidationError(f"加载 {run_dir / 'model.pt'} 失败: {exc}") from exc
            model.eval()
        return model, run_args, meta

    def _checkpoint_n_skills(self, weights: Path) -> int:
        """从权重里的知识点嵌入形状读出词表大小。

        没有 ``meta.json`` 时这是唯一可靠的信息来源——按配置猜 n_skills 一旦猜错，
        加载权重会直接报形状不匹配，或者更糟：恰好同形状而语义错位。
        """
        try:
            state = self._torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelValidationError(f"无法读取权重 {weights}: {exc}") from exc
        _require(isinstance(state, dict), f"{weights} 不是 state_dict")
        for key in ("skill_emb.weight", "skill_embedding.weight", "skills.weight"):
            tensor = state.get(key)
            if tensor is not None and getattr(tensor, "ndim", 0) == 2:
                return int(tensor.shape[0])
        raise ModelValidationError(
            f"无法从 {weights} 推断词表大小（未找到 skill_emb.weight 等二维嵌入）；"
            "请在配置里显式提供 n_skills"
        )

    def _validate_skill_map(self, n_skills: int) -> None:
        if not self.skill_map:
            return
        illegal = {key: value for key, value in self.skill_map.items()
                   if not 0 <= value < n_skills}
        _require(
            not illegal,
            f"skill_map 中存在超出词表范围 0..{n_skills - 1} 的密集 id: {illegal}",
        )

    @staticmethod
    def _read_json(path: Path, *, required: bool) -> dict:
        if not path.is_file():
            if required:
                raise ModelValidationError(f"缺少必需文件: {path}")
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelValidationError(f"无法解析 {path}: {exc}") from exc
        _require(isinstance(payload, dict), f"{path} 必须是 JSON 对象")
        return payload

    # -------------------------------------------------------------- 词表解析
    def resolve_skill(self, tokens: Sequence[str]) -> int:
        """把本项目的知识点标签解析为模型密集 id；多知识点按策略聚合。"""
        tokens = tuple(str(token) for token in tokens)
        if tokens in self._skill_cache:
            return self._skill_cache[tokens]
        if not tokens:
            raise ModelValidationError("目标题目没有知识点标签，无法解析模型 id")

        resolved = [self._resolve_one(token) for token in tokens]
        if len(resolved) == 1:
            result = resolved[0]
        elif self.multi_skill_policy == "first":
            result = resolved[0]
        elif self.multi_skill_policy == "mean":
            raise ModelValidationError(
                "mean 策略需要模型逐知识点输出后再聚合，当前后端每次只输出一个知识点；"
                "请改用 multi_skill_policy=error，或在输入数据中把多知识点题目拆开"
            )
        else:
            raise ModelValidationError(
                f"目标题目含多个知识点 {tokens}。训练仓库未定义聚合口径，"
                "本适配器拒绝擅自取第一个；请提供 skill_map 并显式设置 "
                "multi_skill_policy=first/mean，或在输入数据中把多知识点题目拆开"
            )
        self._skill_cache[tokens] = result
        return result

    def _resolve_one(self, token: str) -> int:
        if token in self.skill_map:
            return self.skill_map[token]
        if self.skill_map:
            raise ModelValidationError(
                f"知识点 {token!r} 不在 skill_map 中；已提供 skill_map 时不再做模糊匹配"
            )
        if not self._meta:
            raise ModelValidationError(
                f"知识点 {token!r} 无法解析：该模型没有知识点词表（meta.json 缺失），"
                "必须显式提供 skill_map"
            )
        try:
            # 训练仓库的解析器：纯数字按原始 skill_id 优先，'#N' 按密集 id，其余按名字匹配。
            return int(self._predict_mastery.resolve_skill(self._meta, token))
        except SystemExit as exc:  # resolve_skill 用 SystemExit 报错，必须单独捕获
            raise ModelValidationError(
                f"知识点 {token!r} 无法映射到模型词表（{exc}）。"
                "请确认输入数据用的是训练时的原始 skill_id，或显式提供 skill_map"
            ) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise ModelValidationError(f"知识点 {token!r} 解析失败: {exc}") from exc

    # ---------------------------------------------------------------- 推理
    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        return [
            Prediction(request.target.record_id, self._predict_one(request))
            for request in requests
        ]

    def _predict_one(self, request: PredictionRequest) -> float:
        skill = self.resolve_skill(request.target.skill_tokens)
        if self._backend == "bkt":
            # BKT 不使用时间，因此仅 event_order 的数据也能推理。
            skills, corrects = history_skills_and_corrects(request.history, self.resolve_skill)
            probability = self._predict_bkt(skills, corrects, skill)
        else:
            gap = target_gap_seconds(
                request, allow_order_only_time=self.allow_order_only_time
            )
            skills, corrects, deltas, dt_globals = build_history_arrays(
                request.history, self.resolve_skill
            )
            probability = self._predict_dkt(skills, corrects, deltas, dt_globals, skill, gap)

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ModelValidationError(
                f"模型对 {request.target.record_id} 输出了非法概率 {probability!r}；"
                "常见原因是该知识点未出现在训练词表中"
            )
        return float(probability)

    def _predict_dkt(self, skills, corrects, deltas, dt_globals, skill, gap) -> float:
        probability = self._predict_mastery.predict_next(
            self._model, skills, corrects, deltas, dt_globals, next_skill=skill, gap_sec=gap
        )
        if self._temperature is not None:
            probability = float(self._predict_mastery.apply_temperature(probability, self._temperature))
        return float(probability)

    def _predict_bkt(self, skills, corrects, skill) -> float:
        # 把目标作为序列的最后一步，取该步的预测；目标的真实标签不参与计算。
        sequence = {"skill": [*skills, skill], "correct": [*corrects, 0]}
        return float(self._bkt.predict_sequences([sequence])[0][-1])


def create_model(config: dict) -> KtForgettingModel:
    """供 ``main.py --model plugin`` 与 ``evaluate.py --factory`` 加载的工厂函数。"""
    return KtForgettingModel(config)
