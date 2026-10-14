"""Demonstrates plugin loading; explicitly remains a mock model."""
from contracts import Prediction


class ConstantPlugin:
    name = "example_constant_plugin"
    is_mock = True

    def __init__(self, probability=0.6):
        self.probability = probability

    def predict(self, requests):
        # Intentionally reverse the results to exercise ID alignment.
        return [Prediction(r.target.record_id, self.probability) for r in reversed(requests)]


def create_model(config):
    return ConstantPlugin(config.get("probability", 0.6))
