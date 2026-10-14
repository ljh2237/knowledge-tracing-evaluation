# 模型评估模块接口说明

交付版本 **1.0.0**。本文档说明负责人 B 刘静涵开发的评估模块的输入格式、校验规则、
指标定义、模型对齐方法与全部输出文件格式。评估模块只消费负责人 A 的
`predictions.csv` 与 `PredictionRequest` / `ModelAdapter` 接口，不修改 v1.0.0 契约。

## 1. 输入：predictions.csv

输入为负责人 A 流水线输出的 `predictions.csv`（UTF-8 或 UTF-8 BOM，逗号分隔）。
也可以直接传入**完整运行目录**，评估程序会自动读取其中的 `predictions.csv` 与
`run_summary.json`。

### 1.1 必需列

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `record_id` | 字符串 | 唯一样本标识，非空、不可重复，是多模型对齐的主键 |
| `y_true` | `0` / `1` | 当前目标题的真实答题结果 |
| `y_pred` | `[0,1]` 有限实数 | 模型预测的**答对概率**；AUC 与 Log Loss 直接使用该列 |

### 1.2 可选列（影响可用功能，缺失即跳过对应分析）

| 列 | 用途 | 缺失后果 |
| --- | --- | --- |
| `cold_start` | 冷启动分组评估 | 只输出 `all` 组，不输出冷启动子分组 |
| `delta_time_seconds` | 遗忘描述性时间分析 | A 部分标记为不可用并说明原因 |
| `time_source` | 区分 `timestamp` / `order` | 描述性分析仍按 `delta_time_seconds` 是否为空判断 |
| `history_length` | 冷启动判定参考 | 不影响指标计算 |
| `model_name` | 模型标识 | 回退到运行目录名或文件名 |
| `is_mock` | 模拟模型标识 | 数据性质标记为未知；图表不加水印 |
| `student_id` / `question_id` / `skill_id` | 多模型样本元信息交叉核对 | 记录为「无法交叉核对」，不默认通过 |

### 1.3 校验规则

校验按以下顺序执行，任何一项不通过都抛出明确错误并以退出码 2 结束，**不会静默丢弃
样本、不会补默认值、不会裁剪概率**：

1. 文件存在且可解析（复用 `data_loader.read_csv_strings` 的 CSV 结构检查）。
2. 文件非空（只有表头视为空文件）。
3. 必需列齐全，缺列时列出缺失列名。
4. `record_id` 非空且唯一，重复时列出重复取值。
5. `y_true` 只允许 `0` / `1`（兼容精确的 `0.0` / `1.0`），空值与其他取值拒绝。
6. `y_pred` 必须可转成数值且在 `[0,1]` 内有限；`NaN`、`Infinity`、负数、大于 1 全部拒绝。
7. `delta_time_seconds` 不允许为负。
8. `cold_start` / `is_mock` 必须是布尔文本（`True/False`，兼容 `1/0`）；
   同一文件内 `is_mock` 不一致视为输入被拼接，直接拒绝。
9. `time_source` 只允许 `timestamp` 或 `order`。
10. `model_name` 在同一文件内必须唯一。

## 2. 指标定义

指标的选择理由、公式推导、适用条件与局限见 **`docs/EVALUATION_METRICS.md`**；
下表只给出计算口径。


| 指标 | 定义 | 实现 |
| --- | --- | --- |
| ROC-AUC | 正样本得分高于负样本的概率；并列计 0.5 | `sklearn.metrics.roc_auc_score` |
| Accuracy | `y_pred >= threshold` 得到的标签与 `y_true` 的一致率，默认 `threshold = 0.5` | `sklearn.metrics.accuracy_score` |
| Log Loss | `-(1/N) Σ [y·ln p + (1-y)·ln(1-p)]`，概率按 `float64` 机器精度裁剪 | `sklearn.metrics.log_loss(labels=[0,1])` |
| Brier | `(1/N) Σ (p - y)²`，作为补充参考 | 直接按定义计算 |

**口径声明**

- AUC 与 Log Loss 使用 `y_pred`（答对概率），**不使用** `predicted_label`。
- `y_pred` 是「当前目标题答对的预测概率」，不是知识掌握概率，报告中不将两者等同。
- 测试集只有单一类别时，AUC **明确标记为不可计算**（`auc_status = undefined_single_class`）
  并记录原因，绝不编造数值；同组的 Accuracy、Log Loss、Brier 仍正常计算。
- `threshold` 只影响 Accuracy 与分类标签，不影响 AUC 与 Log Loss。

### 2.1 样本分组

| 分组 | 含义 |
| --- | --- |
| `all` | 全部有效样本；`--exclude-cold-start` 时排除冷启动样本 |
| `non_cold_start` | `cold_start = False` 的样本 |
| `cold_start` | `cold_start = True` 的样本 |

只要输入含 `cold_start` 列，就始终同时输出三个分组；冷启动样本数可能很少，
其 AUC 单独列出并可能不可计算。

## 3. 多模型对齐方法

以 `record_id` 为唯一对齐主键，**默认严格模式**：

1. 各模型的 `record_id` 集合必须完全相同，不一致时列出各模型的样本数与差异数并失败。
2. 同一 `record_id` 的 `y_true` 必须一致，冲突时列出冲突样本并失败。
3. 同一 `record_id` 的 `student_id` / `question_id` / `skill_id` 必须一致；
   只有所有模型都带该列时才交叉核对，否则记录为「无法交叉核对」。
4. 显式使用 `--missing-policy intersect` 时取交集，此时被丢弃的样本数、涉及模型、
   受影响 `record_id` 样本都会写入 CSV / JSON / Markdown 产物，并在报告中给出警告。
5. 标签冲突**任何策略下都失败**，不会因为取交集而被掩盖。
6. 同时核对各模型 `run_summary.json` 中的 `input_sha256` 与 `configuration`，
   不一致时给出提示（不直接失败，因为样本身份已由 `record_id` 核对）。

## 4. 遗忘测试接口

### 4.1 A 部分：描述性时间间隔分析

- 输入：`predictions.csv` 的 `delta_time_seconds`、`time_source`、`y_pred`。
- 过滤：只保留 `delta_time_seconds` 非空的样本；冷启动样本必然被排除。
- 分桶：默认边界 `0, 60, 300, 3600, 21600, 86400, 604800, 2592000` 秒（左闭右开），
  可通过 `--bucket-edges` 修改。
- 输出：每桶的样本数、占比、时间间隔的最小/最大/均值/中位数、`y_pred` 的
  均值/中位数/标准差、`y_true` 均值、Accuracy。
- 相关系数：Pearson 与 Spearman（整体与非冷启动子集），样本不足或某一维恒定时
  返回「不可计算」。
- **性质声明**：这是观察性分析，未控制学生、知识点、题目难度与历史长度，
  不能作为遗忘的因果证据。仅有 `event_order` 的数据整体跳过并说明原因，
  绝不把事件顺序换算成天数。

### 4.2 B 部分：固定历史受控时间测试

- 输入：原始输入 CSV + 一个或多个模型（`--model mock` 或
  `--factory module:function`，附加模型用 `--controlled-model LABEL=FACTORY`）。
- 候选样本：`target.timestamp` 非空、可见历史非空、`delta_time_seconds` 非空；
  按 `record_id` 排序后用固定 `--seed` 抽样到 `--max-requests` 个，保证可复现。
- 构造方式：对每个候选样本，保持 `history` 元组与目标题目、知识点完全不变，
  只把目标时点改为 `最后一次历史答案可见时点 + interval`，并令
  `delta_time_seconds = interval`。目标时点永远不早于任何历史答案，
  因果顺序始终成立。
- 唯一编号：受控请求的 `record_id` 为 `<原 record_id>#t<间隔秒数>s#r<重复序号>`，
  避免同一批次内 ID 冲突。模型不应依赖 `record_id` 本身。
- 时间间隔：默认 `0,1,3,7,14`，单位由 `--unit` 指定（默认 `days`）；
  0 表示目标时点与最后一次历史作答同时。
- 重复：每个（候选，间隔）组合重复 `--repeats` 次，用于检查推理一致性
  （`max_repeat_spread`）。
- 响应判定：`max |mean(interval) - mean(基线间隔)| > 1e-9` 记为检测到时间响应；
  同时给出是否单调下降，以及间隔与平均概率的 Spearman 相关系数。
- **性质声明**：该测试检验「模型是否对时间间隔作出响应」，不是「模型在真实数据上的
  遗忘效果」。负责人 A 的 `MockModel` 没有遗忘机制，其概率不随时间变化是合理结果。
- 合成夹具：`evaluation/synthetic_time_model.py` 提供 `is_mock=True` 的
  `SyntheticTimeSensitiveModel`，仅用于验证本测试程序能否发现时间响应，
  **不是真实模型，也不是任何训练组的遗忘机制**。

## 5. 性能测试接口

- 输入：原始输入 CSV + 一个模型；请求由 `pipeline.iter_requests` 生成，
  与预测流程使用完全相同的 `PredictionRequest`。
- 请求池：`max_requests` 条（默认 1024），**所有 batch size 共用同一请求池**，
  保证不同批量之间的数字可比。
- 完整性：每个 batch size 只使用能整除的完整批次，舍去尾部不完整批次，
  避免单样本延迟被小批次拉偏。
- 预热：`--warmup` 个批次（默认 1）。
- 重复：`--repeats` 轮（默认 3）。
- 计时：`time.perf_counter()` 包裹每一次 `adapter.predict(batch)`。

| 统计量 | 口径 |
| --- | --- |
| 批次延迟 | 单次 `adapter.predict(batch)` 的墙钟耗时 |
| 单样本延迟 | 批次延迟 ÷ 该批次样本数 |
| P50 / P95 | 对全部被计时的**批次延迟**序列取 `numpy.percentile` 线性插值 |
| 吞吐量 | 被计时样本总数 ÷ 被计时总耗时（样本/秒） |

**边界声明**：`run_summary.json` 中的 `model_seconds` 是整条流水线的累计耗时，
不是模型性能；开发电脑上的数字也不能当作鸿蒙端侧性能。

## 6. 输出文件格式

评估产物根目录默认为 `outputs/evaluation/`（验收证据写入 `evidence/evaluation/`）。

### 6.1 `metrics/metrics.csv` 与 `metrics/metrics.json`

CSV 每行是一个「模型 × 样本分组」的结果，列定义：

```text
label, model_name, is_mock, group, group_description, threshold,
sample_count, positive_count, negative_count, positive_rate,
roc_auc, auc_status, auc_reason, accuracy, log_loss, brier_score,
source_file, source_sha256, data_nature, experiment_status
```

JSON 额外包含 `evaluation_version`、`generated_at_utc`、`scope`、`configuration`、
`inputs`（每个预测文件的来源、SHA256、样本数、数据性质、运行摘要摘要）与 `notes`
（AUC 不可计算的具体原因等）。

`auc_status` 取值：`computed`、`undefined_single_class`、`unavailable`。

### 6.2 `forgetting/`

| 文件 | 内容 |
| --- | --- |
| `interval_buckets.csv` | A 部分的分桶统计，仅在可用时生成 |
| `controlled_time_response.csv` | B 部分的逐条观测（模型、候选、间隔、重复、概率） |
| `forgetting_summary.json` | A 部分的参数/相关系数/结论边界与 B 部分的逐间隔汇总、响应判定、口径说明 |

### 6.3 `comparison/`

| 文件 | 内容 |
| --- | --- |
| `model_comparison.csv` | 每行一个「分组 × 模型」，含 `delta_*` 相对基准的差值 |
| `model_comparison.json` | 新增 `coverage`（样本覆盖与丢弃明细）、`run_summaries`、`notes`、`scope_caveats`、`ranking_by_auc` |
| `model_comparison.md` | 可直接粘贴进报告的中文 Markdown 表格 |

### 6.4 `performance/`

`performance.csv` 每行一个 batch size：

```text
batch_size, batch_count, sample_count, repeats, warmup_batches,
mean_batch_latency_ms, p50_batch_latency_ms, p95_batch_latency_ms,
mean_per_sample_latency_ms, p50_per_sample_latency_ms, p95_per_sample_latency_ms,
throughput_samples_per_second, total_seconds, pool_size, model_name, is_mock
```

`performance.json` 额外记录 `environment`（Python 实现与版本、操作系统、
CPU 逻辑核数、pandas / NumPy / scikit-learn 版本、计时函数）、`settings`、
`input` 与 `caveats`。

### 6.5 `plots/`

九张 PNG（DPI 200）与 `plots_manifest.json`：

```text
auc_comparison.png, accuracy_comparison.png, log_loss_comparison.png, roc_curves.png,
prediction_distribution.png, forgetting_observed.png, forgetting_model_compare.png,
inference_latency.png, throughput_by_batch.png
```

其中 `prediction_distribution.png` 为双面板预测分布图：左侧叠加各模型的 `y_pred`
分布并标出分类阈值，右侧按真实标签分组展示（两组分布重叠越多说明区分能力越弱）。

清单记录每张图的 `status`（`created` / `skipped`）、`reason`、`source` 与
`demo_watermark`。只要数据来自 `is_mock=True` 的模型，图中就有
`DEMO · SYNTHETIC DATA` 水印。

### 6.6 `reports/evaluation_report.md`

十四个章节的 Markdown 报告，全部数值与图片链接由程序从上述产物生成。
未获得真实模型的项目统一标注 **待真实模型接入** 或 **未执行**。

### 6.7 `evaluation_summary.json`

`all` 命令的步骤清单：每一步的 `status`（`executed` / `skipped`）、`detail`
与本次运行的实际复现命令 `reproduce_commands`。

## 7. 复现性与兼容性

- 不依赖绝对路径；所有路径参数都接受相对路径，`--output-dir` 相对当前工作目录解析。
- 无随机性来源使用系统随机数：受控测试的抽样使用显式 `--seed`（默认 20261013）。
- 评估程序只**读取** `evidence/`，从不写入或删除其中任何文件。
- 需要 Python 3.12 及以上（与负责人 A 的 `pyproject.toml` 一致）；
  在不低于 3.10 的版本上同样可运行。
