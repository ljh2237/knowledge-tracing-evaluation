# 接入 kt-forgetting 训练模型

训练仓库 `kt-forgetting`（模块一知识图谱 + 模块二知识追踪）与本研究测试框架的对接说明。
适配器实现位于 [`examples/kt_forgetting_plugin.py`](../examples/kt_forgetting_plugin.py)，
**不修改训练仓库任何一行代码，也不修改本框架的 `contracts.py`**。

## 1. 结论：输出语义一致，差异只在接口形态

首要的澄清：训练仓库输出的**就是答对概率**，不是知识掌握度。证据：

- `src/eval/predict_mastery.py`：`"score_semantics": "predicted_response_probability"`
- 同文件打印 `[预测答对概率，非真实掌握度]`

变量名叫 `mastery`、子命令名叫 `--mode mastery` 都是历史遗留的误称。因此本框架的
`y_pred` 可以直接接它的输出，**不需要改动任何指标口径**。

差异清单：

| 维度 | kt-forgetting | 本框架 | 处理位置 |
| --- | --- | --- | --- |
| 调用形态 | CLI `-m src.eval.predict_mastery`／函数 `predict_next()` | `model.predict(requests)` | 适配器 |
| 知识点编号 | `meta.json` 的密集 id | 框架 `SkillVocabulary` | `skill_map` / 复用其 `resolve_skill` |
| 时间特征 | 历史逐步的 `delta` **与** `dt_global` 两个数组 | 只有目标 `delta_time_seconds` | 适配器从 `available_at` 推导 |
| 目标间隔 | `gap_sec` = 最后一次交互后的额外等待 | `delta_time_seconds` 同义 | 直接映射 |

## 2. 时间特征：本次接入最需要小心的地方

训练仓库 `src/models/inference.py` 明确写着：

> `delta` = gap since the candidate skill was last practised.
> `dt_global` = gap since the last interaction.
> **Never substitute one for the other.**

本框架的 `PredictionRequest` 只提供目标题的全局间隔，历史里没有这两者。因此适配器
从 `Interaction.available_at`（答案可见时点）推导，口径与训练仓库
`load_history_json()` 完全一致：

```text
dt_global[0] = 0                        （首条必须为 0，训练仓库会校验）
dt_global[i] = available_at[i] - available_at[i-1]
clock[i]     = Σ dt_global
delta[i]     = clock[i] - clock[该知识点上次出现]   首次出现为 0
```

**正确性已逐项验证**：`tests/test_kt_forgetting_plugin.py::test_matches_training_repo_history_loader`
把同一份历史分别喂给本适配器与训练仓库自己的 `load_history_json()`，
断言 `skills / corrects / deltas / dt_globals` 四个数组完全相等。

### 由此产生的硬约束

⚠️ **仅有 `event_order` 的数据无法驱动带遗忘机制的后端。**

没有真实时间轴就推不出 `dt_global`。适配器的默认行为是**直接失败并说明原因**，
不把事件顺序当作秒数使用——这与本框架遗忘测试「不使用事件顺序伪造天数」是同一条规则。

确需无时间对照实验时，必须显式设置 `allow_order_only_time: true`，此时所有间隔为 0，
模型退化为无时间基线，产物中仍会保留该配置以供审计。

BKT 后端不使用时间，因此不受此限制。

## 3. 使用方式

### 3.1 先生成词表与配置

训练仓库每个 `exp/<tag>/meta.json` 都带 `skill2id`（原始知识点 → 密集 id）与 `n_skills`。
本框架要的词表从 **1** 起连续编号，模型要的密集 id 从 **0** 起；两者必须来自同一份
`meta.json`，手工抄写极易错位。用脚本一次生成：

```bash
python scripts/build_kt_forgetting_vocab.py \
  --meta ../kt-forgetting/exp/v4_mooccube_la_dkt_forget_s2026/meta.json \
  --out-dir examples --prefix kt_forgetting_mooccube
```

产出 `examples/kt_forgetting_mooccube_vocab.json`（框架词表）与
`examples/kt_forgetting_mooccube_config.json`（适配器配置，含 `skill_map` 与 `n_skills`）。
脚本会校验密集 id 恰好覆盖 `0..n_skills-1`，不连续时直接报错而不是生成错位词表。

### 3.2 配置文件

```json
{
  "repo": "../kt-forgetting",
  "tag": "v4_mooccube_la_dkt_forget_s2026",
  "root": "exp",
  "backend": "auto",
  "n_skills": 22,
  "skill_map": {"C_1925898::第三章  线性方程组": 0},
  "multi_skill_policy": "error",
  "allow_order_only_time": false,
  "threads": 2
}
```

| 字段 | 必填 | 说明 |
| --- | :-: | --- |
| `repo` | 是 | `kt-forgetting` 仓库根目录；也支持环境变量 `KT_FORGETTING_REPO` |
| `tag` | 是 | `exp/<tag>` 目录名，例如 `v4_mooccube_la_dkt_forget_s2026`、`bkt` |
| `root` | 否 | 实验根目录，默认 `exp`，相对 `repo` 解析 |
| `data` | 否 | 覆盖 `metrics.json` 里记录的数据路径（跨机器迁移用） |
| `backend` | 否 | `auto`（默认）/ `dkt` / `bkt`；`auto` 看目录里有没有 `bkt_params.json` |
| `skill_map` | 否 | `{框架知识点标签: 模型密集 id}`；不提供时复用训练仓库的 `resolve_skill` 做解析 |
| `n_skills` | 否 | 词表大小；**仅当实验目录没有 `meta.json` 时必填**（见 §4.2） |
| `multi_skill_policy` | 否 | `error`（默认）/ `mean` / `first`，见 §6 |
| `allow_order_only_time` | 否 | 默认 `false`，见 §2 |
| `allow_untrained` | 否 | 默认 `false`；**仅用于没有权重时验证适配器通路**，启用后 `is_mock` 强制为 `true` |
| `threads` | 否 | torch 线程数，默认 2 |
| `calibration` | 否 | 温度校准 JSON，须与权重哈希匹配 |

### 3.3 生成预测结果

```bash
python main.py --model plugin \
  --factory examples.kt_forgetting_plugin:create_model \
  --model-config examples/kt_forgetting_mooccube_config.json \
  --vocab examples/kt_forgetting_mooccube_vocab.json \
  --input data/heldout.csv --output-dir outputs/real_model
```

`--vocab` 必须是**训练时的固定词表**：框架用它编码输入，适配器再用 `meta.json`
或 `skill_map` 把它映射到模型密集 id。两者必须指向同一套原始 `skill_id`。

### 3.4 评估

```bash
python evaluate.py all --run-dir outputs/real_model \
    --compare-runs outputs/baseline outputs/real_model --overwrite
```

`is_mock=false` 时报告会自动去掉模拟数据说明与图表水印。

## 4. 权重与词表现状

### 4.1 权重已随仓库交付

训练仓库已把 `exp/**/model.pt` 改为入库（`.gitignore` 第 30 行注明），当前实测：

| 项目 | 实测值 |
| --- | --- |
| `exp/<tag>/model.pt` 数量 | **74 份**（`git ls-files` 同样为 74，确认已纳入版本控制） |
| 合计大小 | **12.90 MB** |
| 仍不入库 | `*.onnx`、`preds_test.npz`、`data.npz`（清洗产物） |

因此只要 `git pull` 就能拿到全部 v3 交付模型与 v4 系列权重，`is_mock=False`。
仍**未随仓库提供**的是 `data/processed/**/data.npz`（`data/` 目录只有 `.gitkeep`），
所以训练仓库自己的 `--split / --seq-index` 取序列路径在本机不可用；
用本框架接入时走的是**你自己的 CSV**，不受影响。

### 4.2 少数实验目录没有 `meta.json`

| 目录 | `model.pt` | `meta.json` | 适配器要求 |
| --- | :-: | :-: | --- |
| `v3_*`、`v4_*`（含 MOOCCube） | 有 | 有 | 直接加载，无需额外配置 |
| `repro_v2_*` | 有 | 有 | 直接加载 |
| `akt_*`、`dkt_*`、`synth_*`（v1/v2 期） | 有 | **无** | 必须同时提供 `skill_map` 与 `n_skills` |

后一类不是缺权重，而是缺**词表 sidecar**：模型结构可以按 `metrics.json` 的 `args`
重建、权重可以加载，但没有 `meta.json` 就不知道原始 `skill_id` 对应哪个密集 id。
适配器此时要求显式给出映射，而不是猜一个。错误信息会点名这两个字段。

## 5. 选哪个模型：优先中文线代（MOOCCube v4）

模块一的图谱是**中文线性代数**。之前可用的数据是 ASSISTments（英文初中代数）与
EdNet（英语），题目域与图谱对不上；MOOCCube 中文线代子集（清华 KEG + 学堂在线）
正是为填这个坑准备的，v4 的 `*_mooccube_la_*` 系列就是在它上面训练的。

| tag | 测试 AUC | 题数 | 说明 |
| --- | ---: | ---: | --- |
| `v4_mooccube_la_dkt_base_s2026` | 0.7867 | 6386 | 无遗忘对照 |
| `v4_mooccube_la_dkt_forget_s2026` | **0.7799** | 6386 | 带遗忘，**推荐默认** |
| `v4_mooccube_la_min_delta_forget_s2026` | 0.7776 | 6386 | 时间单位取 `min` 的变体 |

三者的 `delta_unit` 不同（`day` / `min`）。**这个字段决定时间尺度**：适配器按
`metrics.json` 里的 `delta_unit` 换算，你只需要传真实秒数，不要去改单位。

MOOCCube 模型的 `meta.json` 词表是 22 个中文章节名（`课程ID::章标题`），
可直接按标签解析，例如 `C_1925898::第三章  线性方程组`。用 §3.1 的脚本生成词表即可。

> ⚠️ 上面的 AUC 是训练仓库在**它自己的测试集**上报告的，不是本框架算出来的。
> 本框架只能评估你喂进去的预测，两者不要混为一谈。

## 6. 多知识点题目：默认拒绝而不是擅自聚合

本框架的 `Target.skill_tokens` 是元组，支持一道题挂多个知识点；
训练仓库的模型每次只对一个候选知识点输出。训练组尚未定义聚合口径。

适配器因此**默认直接失败**，而不是悄悄取第一个——这与
`docs/EVALUATION_INTERFACE.md` 的要求一致（「不能未经说明选第一个标签」）。

需要时应显式选择：

- `multi_skill_policy: "first"`：取第一个知识点，简单但会丢失信息；
- `multi_skill_policy: "mean"`：需要模型逐知识点输出后聚合，当前后端不支持，
  会明确报错而不是给出错误结果；
- 推荐做法：在输入数据里把多知识点题目拆开，或与训练组商定聚合口径后再启用。

## 7. `is_mock` 的判定规则

| 情况 | `is_mock` | 说明 |
| --- | :-: | --- |
| 加载到真实 `model.pt` | `false` | 训练组的训练成果 |
| BKT（`bkt_params.json`） | `false` | 训练组在真实数据上 EM 拟合的参数 |
| 缺权重 + `allow_untrained: true` | **`true`** | 权重随机初始化，不是训练成果，**强制标记为模拟** |

最后一条是刻意的：随机初始化的模型不能冒充真实模型结论。它的
`name` 也会带 `(untrained)` 后缀，便于在产物中识别。

另外注意：`is_mock` 描述的是**模型**，不描述数据。用真实 BKT 跑合成数据时
`is_mock` 仍为 `false`，而数据性质由 `run_summary.json` 的 `data` 字段记录。

## 8. 已实测的接入结果

### 8.1 BKT（真实训练参数，无需权重文件）

```bash
python main.py --model plugin \
  --factory examples.kt_forgetting_plugin:create_model \
  --model-config examples/kt_forgetting_bkt_config.json \
  --vocab examples/kt_forgetting_vocab.json \
  --input examples/kt_forgetting_fixture.csv \
  --output-dir outputs/kt_forgetting_bkt --overwrite
```

| 项目 | 结果 |
| --- | --- |
| 退出码 | 0 |
| 预测条数 | 36（3 位学生 × 12 次作答） |
| `model_name` | `kt-forgetting:bkt:bkt` |
| `is_mock` | **False**（真实 EM 拟合参数） |
| `experiment_status` | `external_model_inference` |
| 预测概率 | 0.2840 ~ 0.7664，均值 0.5940，标准差 0.1426 |

### 8.2 DKT 带遗忘（真实权重）

| 项目 | 结果 |
| --- | --- |
| tag | `v4_controlled_dkt_forget_s2026` |
| `model_name` | `kt-forgetting:dkt:v4_controlled_dkt_forget_s2026` |
| `is_mock` | **False** |
| 预测条数 | 36 |
| 预测概率 | 0.2237 ~ 0.8982，均值 0.6895 |
| 间隔敏感性 | 同一历史下 gap 从 0 拉到 365 天，概率 0.8565 → 0.8422（单调不增，30 天后趋稳） |

### 8.3 MOOCCube 中文线代模型（端到端）

用 `scripts/build_kt_forgetting_vocab.py --fixture` 生成的**合成**作答（22 个真实章节标签，
6 名学生、144 条）跑通完整链路：

| 项目 | 结果 |
| --- | --- |
| `model_name` | `kt-forgetting:dkt:v4_mooccube_la_dkt_forget_s2026` |
| `is_mock` | **False** |
| 预测条数 | 144，拒收 0 |
| 预测概率 | 0.1076 ~ 0.9353，均值 0.4583 |
| 间隔敏感性 | gap 0 → 365 天，概率 0.3579 → 0.3387 |
| `evaluate.py all` | 6 个步骤执行（指标 / 遗忘 / 绘图 / 报告等），0 跳过 |

> ⚠️ **这一节的指标数字不可用于精度结论。** 作答标签由固定随机种子生成，
> 与模型的训练分布无关；因此 `evaluate.py` 在该夹具上算出的 AUC（约 0.43）
> 只反映「模型没有见过这份数据」，恰恰说明**合成数据不能用来评价模型**。
> 要得到有意义的精度，必须换真实 held-out CSV。此夹具的唯一用途是验证接线。

### 8.4 尚未完成

- **真实 held-out 数据上的精度与遗忘有效性**：需要清洗后的 MOOCCube 作答 CSV（本机 `data/` 为空）。
- **多知识点题目的聚合口径**：等训练组定义。

## 9. 常见问题

**Q：报「缺少模型权重 …/model.pt」怎么办？**
当前仓库的 v3/v4 模型都已带权重，先确认 `git pull` 到了最新提交、`tag` 拼写正确。
只有确实没有权重的目录才会报这个错；`allow_untrained: true` 仅供验证适配器通路，
产物会被标记 `is_mock=true`，不能用于任何精度结论。

**Q：报「有 model.pt 但缺少 meta.json 词表」？**
见 §4.2。同时提供 `skill_map` 与 `n_skills`，或改用自带 `meta.json` 的 `v3_*` / `v4_*` tag。

**Q：报「知识点 X 无法映射到模型词表」？**
输入数据用的 `skill_id` 与训练时不一致。用 §3.1 的脚本从同一份 `meta.json` 生成词表，
确保 CSV 里的 `skill_id` 就是模型词表里的原始标签（MOOCCube 是 `课程ID::章标题`）。

**Q：报「目标题目含多个知识点」？**
见 §6。默认拒绝是刻意的，不是缺陷。

**Q：报「该请求有历史但没有真实时间间隔」？**
输入是仅 `event_order` 的数据（例如 ASSISTments 的 `order_id`）。带遗忘机制的后端
需要真实时间轴。换用带时间戳的数据，或在本框架里显式设 `allow_order_only_time: true`
做无时间对照实验（结果会标注为退化情形）。

**Q：为什么 `exp/bkt/` 没有 `meta.json`？**
`run_bkt.py` 只写 `bkt_params.json`（按密集 id 索引）与 `metrics.json`，
原始‑密集映射原本放在数据目录的 `meta.json` 里，而数据目录不入库。
所以接 BKT 必须显式提供 `skill_map`。
