# 模型接入与评估交接

## 模型工厂

训练组提供一个可导入模块以及工厂函数 `create_model(config: dict)`。入口为 `--model plugin --factory module:create_model`。自定义模块可放在工程目录或通过 `PYTHONPATH` 提供。

工厂应加载权重、训练词表、模型结构和设备设置。真实模型必须声明 `is_mock=False`，开发夹具声明 True。框架不会通过名称自动判断是否真实，不会默默回退到 Mock。工厂加载失败或模型推理异常会返回退出码 2。

```python
from contracts import Prediction

class TrainingModel:
    name = "dkt_checkpoint_实际版本"
    is_mock = False

    def __init__(self, config):
        self.config = config
        # 此处由训练组加载实际模型；例如切到 eval 模式，关闭梯度。

    def predict(self, requests):
        results = []
        for request in requests:
            # 编码 request.history 的 skill_ids/correct，及 target 的题目特征。
            # DKT 的交互编码、AKT 的题目编号、padding 和遮罩取决于训练代码。
            probability = self.infer_one(request)
            results.append(Prediction(request.target.record_id, float(probability)))
        return results

    def infer_one(self, request):
        raise NotImplementedError("由训练组实现真实推理，不得用当前目标答案")

def create_model(config):
    return TrainingModel(config)
```

以上为接入说明，不是已实现的 DKT 模型。`examples/constant_plugin.py` 提供可执行的接口演示，`examples/training_model_template.py` 在未实现时明确失败。

## 请求契约

`predict()` 收到一批不可变 `PredictionRequest`。

| 对象 | 可见字段 |
| --- | --- |
| `request.target` | record_id、student_id、question_id、skill_tokens、skill_ids、timestamp、event_order、time_source |
| `request.history` | 同学生已可见的 Interaction 元组，包含历史 correct 和耗时 |
| `request.delta_time_seconds` | 可用的历史间隔或 None |

目标对象没有当前 correct、当前作答耗时或答案可见时间。框架已经根据时间、答案完成时间和题组隔离规则筛选历史，但训练组仍需确保网络的 padding / attention mask / shift 不泄漏目标。

每个请求代表一个独立的“可见历史 → 当前目标题”任务。模型可以批量编码，不能跨请求累积学生状态，否则批量拆分、重排或重复调用可能改变结果。需要递归状态的模型可从每个请求的历史重建状态。

冷启动历史为空：模型必须提供训练时约定的起始状态/先验并返回一条预测。多知识点以 tuple 提供，不会展开成多条目标。若原模型只支持单知识点，训练组需明确聚合策略，不能未经说明选第一个标签。原始 question_id 保持字符串，题目嵌入索引由训练组的题目词表处理。

## 输出契约与验证

返回 `Sequence[Prediction]`，如 `list[Prediction]` 或 `tuple[Prediction]`。每个目标恰好对应一个 `Prediction(record_id, probability)`。

- 可以乱序返回，框架按 record_id 恢复输入顺序。
- 数量缺失、多余、重复 ID 或未知 ID 全部失败。
- probability 必须为 [0,1] 的有限实数，允许 0 和 1；不允许字符串、布尔值、NaN 或 Infinity。
- PyTorch / MindSpore 的 Tensor 需先转为 Python float，NumPy 的实数标量可直接使用。
- 不自动做 sigmoid / softmax，也不裁剪概率。训练组应明确输出是概率还是 logits，并在插件内转换。

接口不能阻止恶意插件自行读取输入 CSV。真实接入时应代码审查，确认模型只使用请求对象及训练产物。

## 训练组交接清单

1. 可导入模型工厂、完整推理代码和依赖版本。
2. 模型权重、权重版本/校验值、训练配置及设备要求。
3. 与权重匹配的知识点词表；题目词表及交互编码公式。
4. 多知识点策略、未知标签策略、冷启动策略和历史长度要求。
5. 是否需要真实时间及其单位、遗忘机制的参数定义。
6. 固定输入的参考输出，用于验证接入后的数值一致性。

框架已实现统一调用与输出检查；尚未提供的训练资产必须实际接入后才能声称完成真实模型验证。

## 给刘静涵的评估对接

交付输入为标准 `predictions.csv`，最小评估列为 record_id、y_true、y_pred，附加模型类型与时间字段用于筛选和审计。完整示例位于 `evidence/mock_run/predictions.csv`。

```python
import pandas as pd

df = pd.read_csv("evidence/mock_run/predictions.csv", dtype={"record_id": str, "student_id": str})
labels = df["y_true"].to_numpy(dtype=int)
probabilities = df["y_pred"].to_numpy(dtype=float)
# 交给负责人 B 的 AUC / Accuracy / Log Loss 计算程序。
```

本版本只提供可消费的结果与约定，不替代负责人 B 的指标计算、图表、模型对比及遗忘专项实验。Mock 输出均标明 is_mock=True。遗忘实验需要真实模型和可信时间，仅顺序数据的 delta_time_seconds 为空，不能据此形成遗忘结论。真实模型正式评估还需固定训练/验证/测试划分；本框架不会替代训练组划分数据。
