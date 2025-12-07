from evaluate import load

from .base_evaluation_metric import BaseEvaluationMetric, MetricResult


class BERTScoreMetric(BaseEvaluationMetric):
    def __init__(self):
        self._metric = load("bertscore")

    @property
    def name(self) -> str:
        return "bertscore"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        if references is None:
            raise ValueError("BERTScore requires reference texts")

        flat_references = [refs[0] for refs in references]

        results = self._metric.compute(
            predictions=generated,
            references=flat_references,
            lang="en",
        )

        precision = sum(results["precision"]) / len(results["precision"])
        recall = sum(results["recall"]) / len(results["recall"])
        f1 = sum(results["f1"]) / len(results["f1"])

        return MetricResult(
            name=self.name,
            score=f1,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "model_type": self._model_type,
            },
        )
