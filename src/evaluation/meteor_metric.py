from evaluate import load

from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult

class MeteorMetric(BaseEvaluationMetric):
    def __init__(self):
        self._metric = load("meteor")

    @property
    def name(self) -> str:
        return "meteor"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        if references is None:
            raise ValueError("METEOR requires reference texts")

        results = self._metric.compute(predictions=generated, references=references)

        return MetricResult(
            name=self.name,
            score=results["meteor"],
            details={},
        )