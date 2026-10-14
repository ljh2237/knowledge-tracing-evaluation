"""Deterministic history-only mock; not a trained DKT/AKT or forgetting model."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from contracts import Prediction, PredictionRequest


class MockModel:
    name = "mock_history_beta_v1"
    is_mock = True

    def predict(self, requests: Sequence[PredictionRequest]) -> list[Prediction]:
        results = []
        for request in requests:
            relevant = [
                h.correct for h in request.history
                if set(h.skill_tokens).intersection(request.target.skill_tokens)
            ]
            # Beta(2,2) smoothing: cold start 0.5. No access to target label.
            probability = float(np.clip((2 + sum(relevant)) / (4 + len(relevant)), 0.0, 1.0))
            results.append(Prediction(request.target.record_id, probability))
        return results
