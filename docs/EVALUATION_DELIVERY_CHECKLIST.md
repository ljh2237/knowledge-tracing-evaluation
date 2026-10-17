# 刘静涵模型评估模块交付核对

交付版本 **1.0.0**。本清单中所有状态、数字与文件指纹均来自实际运行结果，
可在 `evidence/evaluation/acceptance_summary.json` 中逐项核对，未经验证的条目
一律标注「待真实模型接入」而不填估算值。

## 1. 图片任务对应成果

逐项对应任务图片中负责人 B 刘静涵的 7 项任务。

| 图片任务 | 图片指定成果 | 实际对应文件 | 测试证据 | 完成状态 |
| --- | --- | --- | --- | --- |
| 1. 评价指标设计 | 指标说明文档 | `docs/EVALUATION_METRICS.md` | 含选型理由、公式、计算方法、人工核对数值与局限分析 | **已完成** |
| 2. 指标计算程序 | `evaluator.py` | `evaluation/evaluator.py` | `tests/test_evaluator.py` 18 项；验收「AUC / Accuracy / Log Loss 数值校验」与独立手工计算一致（容差 1e-9） | **已完成**（模拟数据） |
| 3. 遗忘机制测试 | `forgetting_test.py` | `evaluation/forgetting_test.py` | `tests/test_forgetting.py` 28 项；验收「遗忘时间描述性分析」与「固定历史受控时间测试」 | **已完成**（模拟数据 + 合成夹具） |
| 4. 模型对比测试 | `model_compare.py` | `evaluation/model_compare.py` | `tests/test_model_compare.py` 15 项；验收「多模型对比测试」 | **已完成**（模拟模型之间） |
| 5. 可视化程序 | `visualizer.py` | `evaluation/visualizer.py` | `tests/test_visualizer.py` 23 项；验收「可视化图表生成」 | **已完成** |
| 6. 性能测试 | `performance_test.py` | `evaluation/performance_test.py` | `tests/test_performance.py` 24 项；验收「Mock 模型推理性能测试」 | **已完成**（开发机 + Mock） |
| 7. 测试报告 | `report_generator.py` | `evaluation/report_generator.py` | `tests/test_report_generator.py` 15 项；验收「自动生成评估报告」 | **已完成** |

### 1.1 图片任务 5 要求的图表逐一核对

| 图片要求 | 实际图表 |
| --- | --- |
| AUC 对比图 | `auc_comparison.png`（并附 `accuracy_comparison.png`、`log_loss_comparison.png`、`roc_curves.png`） |
| **预测分布图** | `prediction_distribution.png`（左：各模型 `y_pred` 分布叠加 + 阈值线；右：按真实标签分组） |
| 随时间变化的预测曲线 | `forgetting_observed.png`（观察性）+ `forgetting_model_compare.png`（受控固定历史） |
| 补充（任务四、六配套） | `inference_latency.png`、`throughput_by_batch.png` |

共 **9 张** PNG，全部由 `evaluation/visualizer.py` 生成，DPI 200，模拟数据带 DEMO 水印。

### 1.2 刘静涵任务的验收标准

> 「给定一份符合约定格式的预测结果 CSV，能够直接计算评价指标、生成图表并导出报告。
> 对于需要真实模型才能运行的实验，程序应明确提示尚未接入模型，而不是生成虚假的实验结论。」

`scripts/acceptance_evaluation.py` 中的「**验收标准：单份预测 CSV 直接出指标+图表+报告**」
一项专门验证这条标准（类别 `acceptance`）：只给 `evidence/mock_run/predictions.csv`
一份文件，依次执行 `metrics → plots → report`，断言

- 三个命令退出码均为 0；
- 指标 JSON 中 `all` 组的 AUC 与 Accuracy 均已算出；
- 生成 AUC / Accuracy / Log Loss 对比图、ROC 曲线与预测分布图；
- 无法计算的图表（遗忘、性能）被跳过且**每条都写明原因**；
- 导出的报告中含模型名、实际 AUC 数值，并明确写出「待真实模型接入」与「未执行推理性能测试」。

实际结果：**通过**。`is_mock=True` 时程序不会给出任何真实模型的精度结论。

## 2. 实际运行结果摘要

以下数字全部由 `python scripts/acceptance_evaluation.py` 本次实际运行产生。

### 2.1 自动化测试

| 项目 | 数量 |
| --- | ---: |
| pytest 用例总数 | 247 |
| 通过 | 247 |
| 失败 / 错误 / 跳过 | 0 / 0 / 0 |
| 其中负责人 A 原有用例 | 93（全部通过） |
| 其中刘静涵新增用例 | 154 |
| `ruff check .` | 全部通过 |
| 集成验收检查项 | 18 项全部通过（含 8 项边界与异常检查、1 项验收标准检查） |

### 2.2 模型评价指标（`evidence/mock_run` 与 `evidence/plugin_run`，288 条预测）

| 模型 | 分组 | 样本数 | ROC-AUC | Accuracy | Log Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| `mock_history_beta_v1` | all | 288 | 0.5233 | 0.5243 | 0.7070 |
| `mock_history_beta_v1` | non_cold_start | 276 | 0.5238 | 0.5290 | 0.7077 |
| `mock_history_beta_v1` | cold_start | 12 | 0.5000 | 0.4167 | 0.6931 |
| `example_constant_plugin` | all | 288 | 0.5000 | 0.5312 | 0.7009 |
| `example_constant_plugin` | non_cold_start | 276 | 0.5000 | 0.5362 | 0.6989 |
| `example_constant_plugin` | cold_start | 12 | 0.5000 | 0.4167 | 0.7473 |

独立手工复核（Mann-Whitney 秩和定义算 AUC、按定义算 Accuracy 与 Log Loss）：
AUC = 0.523336、Accuracy = 0.524306、Log Loss = 0.707047，与程序输出一致。
常量预测器的 AUC 恰为 0.5，符合理论预期。

> 两个模型都是 `is_mock=True` 的模拟模型，上表只能证明评估程序正确，
> **不能作为 DKT/AKT 的精度结论**。

### 2.3 遗忘机制专项测试

- **A 描述性分析**：288 条中 276 条带有效真实时间间隔，12 条冷启动被排除。
  分桶结果：`[1分,5分)` 56 条、平均概率 0.5110；`[5分,1小时)` 115 条、0.4893；
  `[1天,7天)` 105 条、0.5174。性质为观察性，报告中已明确标注不是因果实验。
- **B 受控固定历史测试**：12 个候选样本 × 5 个时间间隔（0/1/3/7/14 天）× 3 次重复。
  - `mock_history_beta_v1`：最大变化 0.0，**未检测到时间响应**（期望结果，
    MockModel 无遗忘机制）。
  - `synthetic_time_sensitive_fixture_v1`：最大变化 0.4484，
    **单调下降**，Spearman = −1.0000，证明测试程序能够发现时间响应。
- 仅有 `event_order` 的数据（`evidence/assistments_run`）被正确标记为不适用，
  未生成时间分桶，未把事件顺序换算成天数。

### 2.4 多模型对比测试

基准 `mock_beta`，288 条样本完全对齐，`is_mock` 均为真：

| 分组 | 模型 | ROC-AUC | ΔAUC | ΔAccuracy | ΔLogLoss |
| --- | --- | ---: | ---: | ---: | ---: |
| all | `mock_beta`（基准） | 0.5233 | 0.0000 | 0.0000 | 0.0000 |
| all | `constant_plugin` | 0.5000 | −0.0233 | +0.0069 | −0.0062 |

> 该对比只说明两个模拟模型的输出差异，**不能证明任何新模型优于基线**。
> 报告中已写入该结论边界。

### 2.5 推理性能测试（Mock 模型，请求池 256 条，每批量 3 轮，预热 1 批）

| 批量大小 | 平均批次延迟(ms) | P50(ms) | P95(ms) | 单样本平均(ms) | 吞吐量(样本/秒) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0096 | 0.0086 | 0.0109 | 0.0096 | 104536.7 |
| 8 | 0.0732 | 0.0607 | 0.1473 | 0.0092 | 109274.1 |
| 32 | 0.2384 | 0.2373 | 0.2570 | 0.0075 | 134221.2 |
| 64 | 0.5058 | 0.4838 | 0.6048 | 0.0079 | 126530.1 |

运行环境：CPython 3.11.9，Windows-10-10.0.26200-SP0，CPU 逻辑核 18，
pandas 2.2.3 / NumPy 1.26.4 / scikit-learn 1.6.1，计时 `time.perf_counter`。

> 这是开发机上 Mock 模型的数字，**既不等于端侧性能，也不能代表真实 DKT/AKT 的开销**。

### 2.6 可视化

九张图表全部生成，无跳过项：`auc_comparison.png`、`accuracy_comparison.png`、
`log_loss_comparison.png`、`roc_curves.png`、`prediction_distribution.png`、
`forgetting_observed.png`、`forgetting_model_compare.png`、`inference_latency.png`、
`throughput_by_batch.png`。
DPI 200，标签语言 `zh`（本机存在 Microsoft YaHei），全部带 `DEMO · SYNTHETIC DATA` 水印。

### 2.7 报告

`docs/EVALUATION_TEST_REPORT.md` 与 `evidence/evaluation/reports/evaluation_report.md`
由程序自动生成（约 11 000 字符，14 个章节），所有图片链接均指向实际存在的文件。

## 3. 新增与修改文件

### 3.1 新增代码

| 文件 | 作用 |
| --- | --- |
| `evaluate.py` | 统一 CLI 入口，七个子命令 |
| `evaluation/__init__.py` | 评估包导出 |
| `evaluation/common.py` | 共享的预测读取、严格校验、元数据与产物写出 |
| `evaluation/evaluator.py` | 模型评价指标 |
| `evaluation/forgetting_test.py` | 遗忘机制专项测试 |
| `evaluation/model_compare.py` | 多模型对比 |
| `evaluation/performance_test.py` | 推理性能测试 |
| `evaluation/visualizer.py` | 八类图表 |
| `evaluation/report_generator.py` | Markdown 报告生成 |
| `evaluation/synthetic_time_model.py` | 明确标注的合成时间敏感夹具（非真实模型） |
| `scripts/acceptance_evaluation.py` | 评估模块集成验收（17 项检查） |
| `tests/test_evaluator.py` 等 7 个测试文件 | 154 项新增自动化测试 |
| `docs/EVALUATION_METRICS.md` | 评价指标说明文档（图片任务 1 的成果） |
| `docs/EVALUATION_INTERFACE.md`、`docs/EVALUATION_GUIDE.md`、`docs/EVALUATION_DELIVERY_CHECKLIST.md` | 接口、使用与交付文档 |
| `requirements-lock-evaluation.txt` | 评估模块新增依赖的精确版本 |
| `scripts/package_evaluation_release.py` | 生成本次集成交付包 |

### 3.2 新增产物

`outputs/evaluation/`（运行结果，已被 `.gitignore` 忽略）与
`evidence/evaluation/`（验收证据，随包交付）。

### 3.3 修改的原有文件

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `tests/test_cli.py` | 第 21 行 `read_text()` 增加 `encoding="utf-8"` | 原写法依赖系统默认编码，在中文 Windows（cp936）上解码 UTF-8 的 `run_summary.json` 失败。修改与原有行为等价，不改变任何断言。 |
| `README.md` | 追加「刘静涵模型评估模块」章节 | 说明新增模块的运行方式与边界 |
| `requirements.txt` | 追加 `scikit-learn`、`matplotlib` 版本范围 | 评估模块的必要依赖 |
| `.gitignore` | 无需修改 | — |

`contracts.py`、`model_adapter.py`、`mock_model.py`、`pipeline.py`、`data_loader.py`、
`io_utils.py`、`main.py`、`dataset_converters.py`、`generate_sample_data.py`
**未做任何修改**，v1.0.0 接口保持不变。

## 4. 原项目回归检查

| 检查项 | 结果 |
| --- | --- |
| `python main.py` 正常运行 | 通过（12 位学生、288 条预测，退出码 0） |
| `predictions.csv` 列定义不变 | 通过（18 列与 `pipeline.RESULT_COLUMNS` 完全一致） |
| 原有自动化测试全部通过 | 通过（93/93） |
| 新增自动化测试全部通过 | 通过（154/154） |
| `ruff check .` | 通过 |
| 评估程序可独立运行 | 通过（`evaluate.py` 七个命令，含从其他工作目录调用） |
| `evidence/` 原有内容未被破坏 | 通过（`tests/test_evaluation_cli.py::test_evaluation_does_not_touch_ruikaining_evidence` 比对运行前后全部文件的 mtime） |
| 交付包解压后可运行 | 通过（`scripts/package_evaluation_release.py` 重新解压并执行 `main.py` 与 `evaluate.py`） |

## 5. 尚未完成的验证事项

以下事项因缺少**真实 held-out 数据**（权重已随 `kt-forgetting` 仓库交付）**无法完成**，
统一标注「待真实模型接入」：

1. **真实 DKT/AKT 的预测精度**：模型权重已可用（`is_mock=False`），但仓库内没有真实
   作答 CSV，尚未在有意义的测试集上算过指标。夹具上的数字不构成精度结论。
2. **真实遗忘机制的有效性**：受控时间测试目前只覆盖 Mock 与明确标注的合成夹具。
   真实权重的间隔敏感性已验证（概率随间隔单调不增），但「遗忘是否与真实遗忘一致」
   需要真实数据。
3. **真实模型与基线的对照实验**：现有对比发生在两个模拟模型之间。
4. **真实模型推理性能与鸿蒙端侧性能**：本次延迟与吞吐量来自开发机上的 Mock 模型，
   端侧性能需要在鸿蒙设备上另行测量。
5. **真实数据的训练/验证/测试划分**：当前使用负责人 A 的合成数据与人工夹具，
   正式评估需要训练组固定的划分或可复现的划分脚本。
6. **多知识点与稀疏知识点的分组评估**：现有合成数据的知识点分布均匀，
   真实数据上的长尾表现需要在真实数据接入后补做。
7. **Python 3.12 实机验证**：负责人 A 的原始验收在 Python 3.12.14 完成；
   本次评估模块在 **Python 3.11.9 / Windows 11** 上实机验证（本机未安装 3.12），
   代码不使用 3.12 专有语法，但 3.12 环境仍需在具备该版本解释器的机器上补测。
