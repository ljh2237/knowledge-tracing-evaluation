# 鲁凯宁知识追踪测试框架

交付版本 **1.0.0**，日期 **2026 年 10 月 13 日**。本工程完成任务图片中负责人 A 的七项工作：Python 环境、模拟数据、预处理、Mock 模型、统一模型接口、主程序和基础功能测试。当前未收到训练组的真实模型或真实数据文件，默认运行使用明确标记的合成数据与模拟预测。

## 快速运行

建议使用 Python 3.12。项目目录为 `knowledge_tracing_test`，先进入此目录。

macOS / Linux：

```bash
bash scripts/setup.sh
source .venv/bin/activate
python main.py
```

本机交付时已安装 `.venv`，可直接执行后两条命令。macOS 也可双击 `运行演示.command`；缺少环境时会安装锁定依赖，需要联网。

Windows：

```bat
scripts\setup.bat
.venv\Scripts\python main.py
```

第二次运行到相同目录时，显式允许替换已有结果：

```bash
python main.py --overwrite
```

成功后生成 `outputs/demo/`：

| 文件 | 用途 |
| --- | --- |
| `predictions.csv` | 统一预测结果，交给刘静涵的评估程序 |
| `skill_mapping.json` | 本次知识点编号，0 预留给未知知识点 |
| `rejected_rows.csv` | 拒收记录和原因，无拒收时保留表头 |
| `run_summary.json` | 模型类型、数据统计、配置、依赖版本、耗时与文件 SHA256 |

默认数据包含 12 位模拟学生、288 条打乱顺序的答题记录、6 个知识点、多知识点题目和缺失的可选答题耗时。主程序按学生和时间恢复序列。

## 常用命令

```bash
# 自己的标准数据
python main.py --input data/my_records.csv --output-dir outputs/my_run

# 数值 Unix 毫秒时间戳，必须明确单位
python main.py --input data/my_records.csv --timestamp-unit ms --output-dir outputs/ms_run

# 拒收异常行并导出清单，其余数据继续运行
python main.py --input data/my_records.csv --invalid-policy drop --output-dir outputs/clean_run

# 调整批量和历史窗口
python main.py --batch-size 32 --max-history 128 --output-dir outputs/b32

# 重建相同模拟数据；不同 seed 可生成另一份数据
python generate_sample_data.py --output data/new_sample.csv --students 20 --events 30 --seed 7

# 检查自动测试与代码规范
bash scripts/verify.sh

# 重现交付时的集成验收，包括两个转换器与异常流程
python scripts/acceptance.py
```

脚本 `verify.sh` 适用于 macOS / Linux；Windows 可运行相同的 `python -m pytest` 与 `ruff check .` 命令。

## 真实数据与模型接入

ASSISTments 支持含 `user_id, problem_id, skill_id, correct, order_id` 的 CSV。旧版同一次作答的多技能行会合并，修正版 `skill1_skill2` 会转换为标准的分号列表。

```bash
python dataset_converters.py --dataset assistments --input raw/skill_builder_data.csv --output data/assistments.csv
python main.py --input data/assistments.csv --output-dir outputs/assistments
```

只有 `order_id` 时保留顺序，时间间隔为空。原始文件若提供无时区的 `start_time`，需确认真实源时区后指定 `--source-timezone`。使用 `start_time` 时还需 `end_time` 或 `ms_first_response` 确认答案可见时间。其他版本的时间列通过 `--time-column` 显式指定。

EdNet KT1 支持单个 `u1.csv` 或包含多个 `u*.csv` 的目录，并要求配套 `questions.csv`：

```bash
python dataset_converters.py --dataset ednet --input raw/KT1 --questions raw/contents/questions.csv --output data/ednet.csv
python main.py --input data/ednet.csv --output-dir outputs/ednet
```

转换器会关联正确答案和标签，识别原始毫秒时间，并隔离同一 `solving_id` 题组。建议先选取学生子集；该版本在内存中处理输入，未验证 EdNet 全量规模。`examples/` 中的两个数据集样例都是人工构造的测试夹具。

训练组实现 `create_model(config)`，返回含 `name`、`is_mock` 和 `predict(requests)` 的对象。接入时必须使用训练组的固定词表：

```bash
python main.py --model plugin --factory my_training_module:create_model \
  --model-config model/config.json --vocab model/skill_mapping.json \
  --input data/heldout.csv --output-dir outputs/real_model
```

可运行的插件示例：

```bash
python main.py --model plugin --factory examples.constant_plugin:create_model \
  --model-config examples/plugin_config.json \
  --vocab evidence/mock_run/skill_mapping.json --output-dir outputs/plugin_demo
```

该插件仍是模拟模型，`is_mock=True`。真实模型模板 `examples/training_model_template.py` 会明确报错“尚未接入真实模型”，不会伪造预测或自动退回 Mock。

## 设计与交付材料

- `docs/RESEARCH.md`：项目理解、数据集与因果预测调研、选型依据和原始来源。
- `docs/DATA_FORMAT.md`：输入/输出字段、缺失值、时间和词表规则。
- `docs/MODEL_INTERFACE.md`：训练组接入契约及刘静涵的评估对接方式。
- `docs/TEST_REPORT.md`：实际验收结果、复现命令、覆盖范围和未验证事项。
- `docs/DELIVERY_CHECKLIST.md`：任务图片中七项任务对应成果和验收证据。
- `evidence/`：本次实际运行生成的数据转换、预测结果与验收记录。
- `qa/`：pytest 日志、JUnit XML、覆盖率 JSON、静态检查日志。
- `requirements-lock.txt`：本次验证的精确依赖版本；`requirements.txt` 提供允许的版本范围。

当前交付证明测试框架可以运行并正确检查接口与数据；模拟概率不能用于证明 DKT/AKT 的精度、遗忘曲线效果或鸿蒙端侧性能。模型训练、评估指标程序和鸿蒙应用分别属于其他工作环节。

---

## 刘静涵模型评估模块 v1.0.0

负责人 B 刘静涵在本工程之上新增模型评估模块，**不修改**负责人 A 的 v1.0.0 契约与代码，
只消费 `predictions.csv` 与 `PredictionRequest` / `ModelAdapter` 接口。

### 模块功能

| 模块 | 代码 | 说明 |
| --- | --- | --- |
| 模型评价指标 | `evaluation/evaluator.py` | ROC-AUC、Accuracy（阈值可配）、Log Loss、Brier；支持冷启动分组 |
| 遗忘机制专项测试 | `evaluation/forgetting_test.py` | ①已有结果的时间间隔描述性分析 ②固定学习历史的受控时间测试 |
| 模型对比测试 | `evaluation/model_compare.py` | 按 `record_id` 严格对齐的多模型指标对比与差值 |
| 推理性能测试 | `evaluation/performance_test.py` | 批次/单样本延迟、P50/P95、吞吐量、多批量比较 |
| 实验结果可视化 | `evaluation/visualizer.py` | 八类图表，模拟数据自动加 DEMO 水印 |
| 报告生成 | `evaluation/report_generator.py` | 十四章节的 Markdown 评估报告 |

### 环境依赖

在负责人 A 的依赖之外，评估模块需要 `scikit-learn` 与 `matplotlib`：

```bash
# macOS / Linux
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -r requirements-lock-evaluation.txt

# Windows
scripts\setup.bat
.venv\Scripts\python -m pip install -r requirements-lock-evaluation.txt
```

`requirements.txt` 已包含这两个依赖的版本范围，`requirements-lock-evaluation.txt`
记录本次验证的精确版本。负责人 A 的 `requirements-lock.txt` 保持不变。

### 运行命令

```bash
# 一条命令完成当前条件下全部可执行的评估流程
python evaluate.py all --overwrite

# 分步执行
python evaluate.py metrics     --predictions evidence/mock_run evidence/plugin_run --overwrite
python evaluate.py forgetting  --predictions evidence/mock_run --input sample_data.csv \
                               --intervals 0,1,3,7,14 --unit days --repeats 3 --overwrite
python evaluate.py compare     --predictions evidence/mock_run evidence/plugin_run \
                               --labels mock_beta constant_plugin --baseline mock_beta --overwrite
python evaluate.py performance --input sample_data.csv --batch-sizes 1,8,32,64 --overwrite
python evaluate.py plots       --evaluation-dir outputs/evaluation \
                               --predictions evidence/mock_run/predictions.csv \
                               evidence/plugin_run/predictions.csv --overwrite
python evaluate.py report      --evaluation-dir outputs/evaluation \
                               --docs-report docs/EVALUATION_TEST_REPORT.md --overwrite

# 自动测试、静态检查与集成验收
python -m pytest -q
python -m ruff check .
python scripts/acceptance_evaluation.py
```

七个命令都支持 `python evaluate.py <命令> --help`。退出码与 `main.py` 一致：成功 0，错误 2。

### 输入输出

- 输入：负责人 A 的 `predictions.csv`，或直接给完整运行目录
  （会自动读取 `predictions.csv` 与 `run_summary.json`）。
  必需列为 `record_id`、`y_true`、`y_pred`；详细校验规则见
  `docs/EVALUATION_INTERFACE.md`。
- 输出：`outputs/evaluation/` 下的 `metrics/`、`forgetting/`、`comparison/`、
  `performance/`、`plots/`、`reports/`，以及 `evaluation_summary.json`。
  验收证据写入 `evidence/evaluation/`（评估程序只读 `evidence/` 原有内容）。

### 自动测试方法

```bash
python -m pytest -q                     # 本次实际结果：243 项全部通过
python -m pytest tests/test_evaluator.py -q   # 只跑评估模块的某一组测试
python scripts/acceptance_evaluation.py # 17 项集成验收检查
```

实际结果见 `docs/EVALUATION_TEST_REPORT.md` 与
`evidence/evaluation/acceptance_summary.json`。

### 真实模型接入方式

评估程序无需修改即可评估真实模型，只要 `predictions.csv` 满足输入格式：

```bash
python main.py --model plugin --factory <模块>:create_model \
    --model-config model/config.json --vocab model/skill_mapping.json \
    --input data/heldout.csv --output-dir outputs/real_model
python evaluate.py all --run-dir outputs/real_model \
    --compare-runs outputs/baseline outputs/real_model --overwrite
```

训练组需要提供的资产清单见 `docs/MODEL_INTERFACE.md` 与
`docs/EVALUATION_TEST_REPORT.md` 第 14 节。

### 当前已验证与未验证的范围

**已验证**（模拟数据与人工夹具，全部标记 `is_mock=True`）：

- 指标计算正确性，含与独立手工计算（Mann-Whitney 秩和 AUC、按定义算 Log Loss）的数值一致。
- 单一类别时 AUC 明确标记为不可计算，不编造数值。
- 冷启动样本可被单独分组评估。
- 严格输入校验：缺列、空文件、重复 `record_id`、非法标签、非法概率、负时间间隔全部失败。
- 多模型严格对齐：样本覆盖不一致、标签冲突、元信息冲突全部失败，不静默删除样本。
- 受控固定历史时间测试能正确区分「无遗忘机制的 Mock」与「明确标注的合成时间敏感夹具」。
- 仅有 `event_order` 的数据不会把事件顺序伪造成天数。
- 八类图表与十四章节报告可自动生成，图片链接有效。
- 原有 `main.py`、原有自动化测试与 `predictions.csv` 列定义未受影响。

**未验证**（统一标注「待真实模型接入」）：

- 真实 DKT/AKT 的预测精度。
- 真实遗忘机制的有效性与遗忘曲线。
- 真实模型与基线的对照实验结论。
- 真实模型推理性能与鸿蒙端侧性能。
- 真实数据的训练/验证/测试划分。
- Python 3.12 实机验证：负责人 A 的原始验收在 Python 3.12.14 完成；本次评估模块在
  **Python 3.11.9 / Windows 11** 上实机验证（本机未安装 3.12），未使用 3.12 专有语法。

### 评估模块文档

- `docs/EVALUATION_INTERFACE.md`：输入格式、校验规则、指标定义、对齐方法、输出格式。
- `docs/EVALUATION_GUIDE.md`：使用指南与常见口径陷阱。
- `docs/EVALUATION_TEST_REPORT.md`：由程序自动生成的实际评估报告。
- `docs/EVALUATION_DELIVERY_CHECKLIST.md`：六项任务对应成果与交付状态。
