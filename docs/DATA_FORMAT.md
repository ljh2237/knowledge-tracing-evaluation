# 数据格式与预处理规则

## 标准输入

编码为 UTF-8 或 UTF-8 BOM，逗号分隔，一行代表一次作答。多知识点保存在同一行，使用分号分隔。ID 全部按字符串读取，保留前导零及字面值 `NA`。

| 字段 | 必填 | 内容与处理 |
| --- | --- | --- |
| `record_id` | 可选 | 唯一作答 ID；整列不存在时按原始数据行位置生成，列存在时不得留空或重复 |
| `student_id` | 是 | 非空学生 ID |
| `question_id` | 是 | 非空题目 ID |
| `skill_id` | 是 | 非空知识点，如 `K01` 或 `K01;K02`；去重并按字典序排序，拒绝空标签与 `-1` |
| `correct` | 是 | `0` / `1`，也兼容精确的 `0.0` / `1.0`；缺失或其他值拒收 |
| `timestamp` | 时间模式必填 | 该次作答的预测参照时点；ISO 时间必须含时区，如 `2026-10-01T09:00:00+08:00` |
| `event_order` | 顺序模式必填 | 非负整数，只在没有真实时间时用于顺序；不得换算为秒数 |
| `available_at` | 可选 | 答案已可见的时间，不得早于 timestamp；省略表示输入记录的标签在 timestamp 时已可见 |
| `response_time_ms` | 可选 | 已发生作答的有限非负耗时，单位毫秒；缺失保留 None，不插补为 0 |
| `interaction_group` | 可选 | 同学生的题组/会话 ID；同组答案不会进入彼此的历史 |

每份输入必须全部为时间模式，或全部为顺序模式。缺失时间且缺失有效顺序的记录拒收。默认按学生字符串 ID、时间或顺序、原始记录位置稳定排序。

`timestamp` 的语义必须由数据提供方确认。如果源记录时间为题目给出时间，需提供答案完成时点 `available_at`；若为已有答案的日志时间，可采用默认值。EdNet 转换器自动计算给题时间加耗时，ASSISTments 的 start_time 转换也会记录答案可见时间。

标准样例：

```csv
record_id,student_id,question_id,skill_id,correct,timestamp,response_time_ms
r1,001,Q01,K01,1,2026-10-01T09:00:00+08:00,23000
r2,001,Q02,K01;K02,0,2026-10-01T09:05:00+08:00,
```

## 缺失值与异常策略

默认 `--invalid-policy error`：发现任何无效行即停止，不发布新的运行目录。`--invalid-policy drop`：保留有效行，在 `rejected_rows.csv` 输出原始字段、`source_row` 和原因，摘要记录拒收数量。全部记录无效、结构列缺失、CSV 结构损坏或时间模式混用始终失败。

`source_row` 是 CSV 数据记录序号加 1，表头视为第 1 条；含单元格内换行时不是文本编辑器的物理行号。完全空白行忽略。模型预测数量/ID/概率错误始终失败，不裁剪非法输出，也不换成默认概率。

## 时间与因果历史

- 目标请求不含当前答案，也不含当前耗时。
- 只使用同一学生时间/顺序严格早于目标的记录。
- 时间模式仅使用 `available_at <= 目标 timestamp` 的历史答案。
- 相同时间/顺序的记录互相不可见；同一非空 `interaction_group` 中的答案互相不可见。
- 在上述可见历史中保留最近 `max_history` 条，默认 256。
- `delta_time_seconds` 是目标时点减去历史中最后一次作答的答案可见时点；冷启动或仅顺序数据留空。
- 历史按作答参照时间排序，不按答案完成顺序重排。

Unix 时间戳必须通过 `--timestamp-unit s` 或 `ms` 指定单位，不会猜测。字符串时间必须包含时区，输出统一为 UTC ISO 时间。

## 知识点词表

```json
{
  "schema_version": "1.0",
  "unknown_id": 0,
  "skills": {"K01": 1, "K02": 2}
}
```

编号从 1 连续递增，0 留给未知标签。模拟运行无词表时从输入标签的字典序建立词表，仅用于当前流程。接真实模型必须提供训练词表；不能在测试集上重建编号。未知知识点默认失败，仅在模型确实训练了 UNK 处理时才可使用 `--unknown-skill-policy unk`。

## 预测输出

`predictions.csv` 每条有效输入对应一条预测，按学生和时间顺序输出。

| 字段 | 内容 |
| --- | --- |
| `record_id, student_id, question_id` | 可与原始记录对齐的字符串 ID |
| `skill_id, skill_index` | 分号知识点标签及固定词表编号 |
| `timestamp, event_order, time_source` | UTC 时间、顺序号与 `timestamp` / `order` |
| `y_true` | 真实或合成输入中的 0/1 标签，在模型推理之后关联 |
| `y_pred` | 答对概率，有限实数且处于 [0,1] |
| `predicted_label` | y_pred >= threshold，默认 threshold 为 0.5 |
| `history_length` | 实际传给模型的可见历史条数 |
| `delta_time_seconds` | 时间间隔；缺少真实时间或历史时留空 |
| `cold_start` | 是否没有可见历史 |
| `model_name, is_mock` | 模型名称与是否为模拟模型 |
| `source_row` | 标准输入中的数据记录位置 |
| `schema_version` | 当前为 1.0 |

负责人 B 应按 record_id 对齐预测与标签，使用 y_pred 作为 AUC / Log Loss 的概率输入；不能把 predicted_label 用作概率。默认包含冷启动，正式评估应明确是否保留，并固定同一套筛选规则。

## 运行摘要与可复现性

`run_summary.json` 保存数据清洗统计、推理配置、Python/pandas/NumPy 版本、模型是否为 Mock、工厂名称、配置及训练词表 SHA256、输入 SHA256、输出文件 SHA256、模型调用耗时和导出前流程耗时。耗时仅描述本次运行，不等同于端侧性能。

插件权重的路径与校验由训练组负责；真实模型工厂应记录权重版本并确保推理确定性。CSV 与词表在固定输入下可复现；摘要的生成时间及耗时会变化。
