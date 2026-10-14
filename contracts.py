"""Immutable, label-free prediction requests shared with the training team."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"


class DataValidationError(ValueError):
    """Input cannot be interpreted without inventing facts."""


class ModelValidationError(ValueError):
    """A model violated the inference contract."""


@dataclass(frozen=True)
class Interaction:
    record_id: str
    student_id: str
    question_id: str
    skill_tokens: tuple[str, ...]
    skill_ids: tuple[int, ...]
    correct: int
    timestamp: float | None
    available_at: float | None
    event_order: int | None
    response_time_ms: float | None
    interaction_group: str
    source_row: int


@dataclass(frozen=True)
class Target:
    record_id: str
    student_id: str
    question_id: str
    skill_tokens: tuple[str, ...]
    skill_ids: tuple[int, ...]
    timestamp: float | None
    event_order: int | None
    time_source: str


@dataclass(frozen=True)
class PredictionRequest:
    target: Target
    history: tuple[Interaction, ...]
    delta_time_seconds: float | None


@dataclass(frozen=True)
class Prediction:
    record_id: str
    probability: float


class PredictModel(Protocol):
    name: str
    is_mock: bool

    def predict(self, requests: Sequence[PredictionRequest]) -> Sequence[Prediction]:
        """Return exactly one ID-keyed probability per target; do not retain state."""
        ...
