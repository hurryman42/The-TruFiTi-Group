from detoxify import Detoxify

from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult

class Toxicity(BaseEvaluationMetric):
    def __init__(self, model: str):
        if model not in ["original", "unbiased"]:
            raise ValueError(f"Unsupported toxicity model: {type}")
        self.model = Detoxify(model)

    @property
    def name(self) -> str:
        return "toxicity"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:

        results = self.model.predict(generated)

        return MetricResult(
            name=self.name,
            score=results["toxicity"],
            details=results,
        )