from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MetricResult:
    name: str
    score: float
    details: dict | None = None


class BaseEvaluationMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        pass
