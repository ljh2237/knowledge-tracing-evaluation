# 模型评估模块使用指南

面向使用本评估模块的队友与后续维护者。命令全部在工程根目录（含 `evaluate.py`）执行。
输入输出格式与指标口径见 `docs/EVALUATION_INTERFACE.md`。

## 1. 环境准备

评估模块在负责人 A 的依赖之外需要 `scikit-learn` 与 `matplotlib`：

```bash
# macOS / Linux（沿用负责人 A 的环境）
bash scripts/setup.sh
source .venv/bin/activate
python -m pip install -r requirements-lock-evaluation.txt

# Windows
scripts\setup.bat
.venv\Scripts\python -m pip install -r requirements-lock-evaluation.txt
```

`requirements.txt` 已包含两个新增依赖的版本范围，
`requirements-lock-evaluation.txt` 记录本次实际验证的精确版本。
负责人 A 的 `requirements-lock.txt` 保持不变，仍是其原始环境的记录。

## 2. 最省事的方式：一条命令

```bash
python evaluate.py all --overwrite
```

等价于依次执行 `metrics → forgetting → compare → performance → plots → report`，
默认使用负责人 A 的 `evidence/mock_run`、`evidence/plugin_run` 与 `sample_data.csv`。
产物写入 `outputs/evaluation/`。

- 输入缺失的步骤会被**跳过**并在 `evaluation_summary.json` 与报告中写明原因。
- 真正的错误（数据损坏、样本无法对齐）会立即以退出码 2 失败，不会吞掉异常。
- 目录已存在且非空时必须显式加 `--overwrite`。

## 3. 分步执行

### 3.1 模型评价指标

```bash
# 单个模型
python evaluate.py metrics --predictions evidence/mock_run/predictions.csv

# 直接给完整运行目录，多个模型一起算
python evaluate.py metrics --predictions evidence/mock_run evidence/plugin_run

# 改阈值、排除冷启动
python evaluate.py metrics --predictions evidence/mock_run \
    --threshold 0.4 --exclude-cold-start
```

### 3.2 遗忘机制专项测试

```bash
# A 部分（描述性）：只需要已有的预测结果
python evaluate.py forgetting --predictions evidence/mock_run

# A + B（受控固定历史）：需要原始输入 CSV 与模型
python evaluate.py forgetting \
    --predictions evidence/mock_run --input sample_data.csv \
    --intervals 0,1,3,7,14 --unit days --repeats 3 --max-requests 24

# 两个模型的时间响应对比（图表 6 需要至少两个模型）
python evaluate.py forgetting --input sample_data.csv \
    --controlled-model fixture=evaluation.synthetic_time_model:create_model

# 接入真实模型（训练组提供词表与配置）
python evaluate.py forgetting --input data/heldout.csv \
    --factory my_dkt:create_model --model-config model/config.json \
    --vocab model/skill_mapping.json
```

### 3.3 多模型对比

```bash
python evaluate.py compare --predictions evidence/mock_run evidence/plugin_run \
    --labels baseline dkt_forgetting --baseline baseline

# 确实需要取交集时显式声明，丢弃明细会写入产物
python evaluate.py compare --predictions a.csv b.csv --missing-policy intersect
```

### 3.4 推理性能

```bash
python evaluate.py performance --input sample_data.csv \
    --batch-sizes 1,8,32,64 --repeats 3 --warmup 1 --max-requests 256

# 真实模型
python evaluate.py performance --input data/heldout.csv \
    --factory my_dkt:create_model --model-config model/config.json
```

### 3.5 图表与报告

```bash
python evaluate.py plots --evaluation-dir outputs/evaluation \
    --predictions evidence/mock_run/predictions.csv evidence/plugin_run/predictions.csv

python evaluate.py report --evaluation-dir outputs/evaluation \
    --docs-report docs/EVALUATION_TEST_REPORT.md --overwrite
```

`plots` 与 `report` 只读取其他子命令已经写出的 JSON / CSV 产物，
因此可以反复重画、重写报告而不必重新推理。

## 4. 自动测试与验收

```bash
# 全部自动化测试（负责人 A 的 + 刘静涵的）
python -m pytest -q

# 静态检查
python -m ruff check .

# 集成验收：实际执行 17 项检查并写出证据
python scripts/acceptance_evaluation.py
```

验收结果写入 `evidence/evaluation/`，包含 `acceptance.log`、
`acceptance_summary.json`（含每项检查的期望、实际与产物 SHA256）、各模块结果文件、
八张图表、`junit.xml` 与自动生成的报告。

## 5. 结果怎么看

| 想知道什么 | 看哪里 |
| --- | --- |
| 模型 AUC / Accuracy / Log Loss | `metrics/metrics.csv` |
| AUC 为什么是空的 | `metrics/metrics.json` 的 `notes` 与 `auc_status` / `auc_reason` |
| 冷启动会不会拖后腿 | 同上 CSV 的 `cold_start` / `non_cold_start` 两行 |
| 时间间隔与预测概率的关系 | `forgetting/interval_buckets.csv` |
| 模型对时间是否敏感 | `forgetting/forgetting_summary.json` 的 `response` |
| 两个模型差多少 | `comparison/model_comparison.md` |
| 推理快不快 | `performance/performance.csv` |
| 本次跑了什么、怎么复现 | `evaluation_summary.json` 的 `steps` 与 `reproduce_commands` |

## 6. 必须注意的口径陷阱

1. **`y_pred` 不是知识掌握概率**，它是「当前目标题答对的预测概率」。
2. **`predicted_label` 不能当概率用**，AUC 与 Log Loss 一律使用 `y_pred`。
3. **`delta_time_seconds` 是时间间隔，不是掌握度**；仅有 `event_order` 的数据
   该列为空，程序不会把它换算成天数。
4. **`is_mock=True` 的结果不能证明任何模型精度**。模拟模型之间的对比只能验证
   对比程序本身。
5. **`run_summary.json` 的 `model_seconds` 不是模型性能**，它是整条流水线的累计耗时。
6. **开发电脑的延迟不等于鸿蒙端侧性能**。端侧性能需要在目标设备上另行测量。
7. **单一类别时 AUC 无定义**，程序会明确标记而不是给出 0.5 或 1.0。
8. **观察性时间分析不是遗忘的因果证据**，它没有控制任何混淆变量。

## 7. 常见问题

**Q：`metrics` 报「缺少必需列」怎么办？**
预测文件必须含 `record_id`、`y_true`、`y_pred`。请确认输入是负责人 A 的
`predictions.csv` 而不是原始答题 CSV；原始数据要先经过 `main.py` 生成预测。

**Q：`compare` 报「样本覆盖不一致」？**
说明两个模型不是在同一批测试样本上推理。默认严格失败是刻意的设计：静默取交集会
让对比结论失真。确认两次推理用的是同一份输入与同一套 `--invalid-policy`，
确实需要取交集时再显式加 `--missing-policy intersect`。

**Q：`forgetting` 说「未检测到时间响应」，是程序坏了吗？**
不是。负责人 A 的 `MockModel` 没有遗忘机制，概率不随时间变化才是正确结果。
用 `--controlled-model fixture=evaluation.synthetic_time_model:create_model`
跑一次，如果合成夹具被检测出单调下降，说明程序本身是有效的。

**Q：图表标题是英文的？**
说明当前环境没有可用的中文字体。程序会自动退回英文标签（`plots_manifest.json`
的 `font_language` 字段会记录实际使用的语言），不会画出方框。安装任意一款
中文字体（如 Noto Sans CJK SC）后重新生成即可。

**Q：想重新跑一次但不想删除旧结果？**
换一个 `--output-dir`。同一目录重复写入必须显式加 `--overwrite`，
这是为了避免误删已有的验收证据。

## 8. 接入真实模型后要做什么

1. 按 `docs/MODEL_INTERFACE.md` 让训练组提供模型工厂、权重、词表与配置。
2. 用 `python main.py --model plugin --factory ... --vocab ...` 生成
   `predictions.csv`，确认 `run_summary.json` 中 `is_mock` 为 `false`。
3. 重新执行 `python evaluate.py all --run-dir outputs/real_model --overwrite`。
4. 报告会自动去掉模拟数据的说明与图表水印，并把第 12 节的对应条目换成真实结论。
5. 对比真实模型与基线时，必须使用同一份测试集划分——这一步需要训练组提供。
