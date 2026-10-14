"""明确标注的合成时间敏感模型夹具，仅用于验证遗忘测试程序是否有效。

**这不是真实的 DKT/AKT，也不是任何训练组的遗忘机制模型。** 它存在的唯一目的，
是让「受控固定历史时间测试」有一个可预期的参照对象：如果测试程序正确，就应当
在这个夹具上检测到预测概率随时间间隔单调下降；而在负责人 A 的 ``MockModel``
（无遗忘机制）上则不应出现下降。

任何报告引用本夹具的结果时，都必须同时标注 ``is_mock=True`` 与
``synthetic_time_sensitive_fixture``，且不得据此声称遗忘机制有效。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contracts import Prediction, PredictionRequest

SECONDS_PER_DAY = 86400.0


class SyntheticTimeSensitiveModel:
    """合成夹具：p = floor + (base - floor) * exp(-decay * 天数)。

    ``base`` 与 ``MockModel`` 一样只使用同知识点可见历史，因此固定历史时 ``base``
    恒定，预测概率随 ``delta_time_seconds`` 单调下降。
    """

    name = "synthetic_time_sensitive_fixture_v1"
    is_mock = True

    def __init__(self, decay_per_day: float = 0.35, floor: float = 0.05):
        if not np.isfinite(decay_per_day) or decay_per_day <= 0:
            raise ValueError("decay_per_day 必须是正的有限数")
        if not 0 <= floor < 1:
            raise ValueError("floor 必须在 [0,1) 内")
        self.decay_per_day = float(decay_per_day)
        self.floor = float(floor)

    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        results = []
        for request in requests:
            relevant = [
                h.correct for h in request.history
                if set(h.skill_tokens).intersection(request.target.skill_tokens)
            ]
            base = (2 + sum(relevant)) / (4 + len(relevant))
            seconds = request.delta_time_seconds or 0.0
            days = max(float(seconds), 0.0) / SECONDS_PER_DAY
            probability = self.floor + (base - self.floor) * float(np.exp(-self.decay_per_day * days))
            results.append(Prediction(request.target.record_id, float(np.clip(probability, 0.0, 1.0))))
        return results


def create_model(config: dict) -> SyntheticTimeSensitiveModel:
    """供 ``main.py --factory`` 与 ``evaluate.py --factory`` 加载的工厂函数。"""
    return SyntheticTimeSensitiveModel(
        decay_per_day=float(config.get("decay_per_day", 0.35)),
        floor=float(config.get("floor", 0.05)),
    )
