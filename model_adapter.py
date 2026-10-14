"""One validated predict() entry point for mock or external inference models."""
from __future__ import annotations

import importlib
import math
import numbers
from collections.abc import Sequence

from contracts import ModelValidationError, Prediction, PredictionRequest, PredictModel


class ModelAdapter:
    def __init__(self, model: PredictModel):
        if not callable(getattr(model, "predict", None)):
            raise ModelValidationError("模型必须提供 predict(requests) 方法")
        if not isinstance(getattr(model, "name", None), str) or not model.name.strip():
            raise ModelValidationError("模型必须提供非空字符串 name")
        if type(getattr(model, "is_mock", None)) is not bool:
            raise ModelValidationError("模型必须明确声明布尔值 is_mock")
        self.model = model

    @classmethod
    def from_factory(cls, specification: str, config: dict) -> ModelAdapter:
        try:
            module_name, function_name = specification.split(":")
            if not module_name or not function_name:
                raise ValueError("工厂格式应为 module:function")
            factory = getattr(importlib.import_module(module_name), function_name)
            return cls(factory(config))
        except Exception as exc:
            raise ModelValidationError(f"无法加载模型工厂 {specification!r}: {exc}") from exc

    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        if not requests:
            return []
        expected_ids = [r.target.record_id for r in requests]
        if len(set(expected_ids)) != len(expected_ids):
            raise ModelValidationError("请求 record_id 重复")
        try:
            results = self.model.predict(tuple(requests))
        except Exception as exc:
            raise ModelValidationError(f"模型 {self.model.name} 推理失败: {exc}") from exc
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            raise ModelValidationError("predict 必须返回 Prediction 序列")
        if len(results) != len(requests):
            raise ModelValidationError(f"预测数量不匹配: 需要 {len(requests)}，收到 {len(results)}")
        by_id: dict[str, Prediction] = {}
        for result in results:
            if not isinstance(result, Prediction):
                raise ModelValidationError("预测必须为 contracts.Prediction(record_id, probability)")
            if not isinstance(result.record_id, str) or result.record_id not in expected_ids:
                raise ModelValidationError(f"预测包含未知 record_id: {result.record_id!r}")
            if result.record_id in by_id:
                raise ModelValidationError(f"预测 record_id 重复: {result.record_id}")
            p = result.probability
            if isinstance(p, bool) or not isinstance(p, numbers.Real):
                raise ModelValidationError("预测概率必须是实数，不能是布尔值或字符串")
            if not math.isfinite(float(p)) or not 0 <= p <= 1:
                raise ModelValidationError(f"预测概率必须为 [0,1] 有限数: {p}")
            by_id[result.record_id] = Prediction(result.record_id, float(p))
        # Model may return an arbitrary order, but never associate by array position.
        return [by_id[record_id] for record_id in expected_ids]
