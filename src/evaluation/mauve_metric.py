import mauve

from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult


class MauveMetric(BaseEvaluationMetric):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "mauve"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        if references is None:
            raise ValueError("MAUVE requires reference texts")

        results = mauve.compute_mauve(p_text=references, q_text=generated)

        return MetricResult(
            name=self.name,
            score=results.mauve,
            details={
                "mauve score": results.mauve,
                "mauve star": results.mauve_star,
                "frontier integral": results.frontier_integral,
                "frontier integral star": results.frontier_integral_star,
            },
        )
