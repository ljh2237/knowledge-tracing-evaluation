"""刘静涵负责的模型评估模块。

六个子模块相互独立，共享 ``evaluation.common`` 中的数据校验与产物写出逻辑：

- ``evaluator``：ROC-AUC / Accuracy / Log Loss 等模型评价指标。
- ``forgetting_test``：遗忘机制专项测试（描述性分析与受控固定历史测试）。
- ``model_compare``：多模型按 record_id 对齐的比较测试。
- ``performance_test``：推理延迟与吞吐量测试。
- ``visualizer``：matplotlib 结果可视化。
- ``report_generator``：Markdown 评估报告自动生成。

模块只消费负责人 A 的 ``predictions.csv`` 与 ``PredictionRequest`` / ``ModelAdapter``
接口，不修改原有契约。所有模拟数据产物均标记 ``is_mock``，不得用于证明真实模型精度。
"""
from __future__ import annotations

from evaluation.common import (
    EVALUATION_VERSION,
    EvaluationError,
    PredictionData,
    load_predictions,
    sha256_of_text,
)

__all__ = [
    "EVALUATION_VERSION",
    "EvaluationError",
    "PredictionData",
    "load_predictions",
    "sha256_of_text",
]
