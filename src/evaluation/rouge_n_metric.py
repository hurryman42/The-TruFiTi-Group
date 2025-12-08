from rouge_score import rouge_scorer

from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult


class RougeNMetric(BaseEvaluationMetric):
    def __init__(self, type: str):
        if type not in ["rouge1", "rouge2", "rougeL"]:
            raise ValueError(f"Unsupported ROUGE type: {type}")
        self.type = type
        self.scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)

    @property
    def name(self) -> str:
        return f"{self.type}-score"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        if references is None:
            raise ValueError("references must be provided")

        result_precision, result_recall, result_f1 = 0.0, 0.0, 0.0

        for pred, refs in zip(generated, references, strict=False):
            current_precision, current_recall, current_f1 = 0.0, 0.0, 0.0

            for ref in refs:
                scores = self.scorer.score(ref, pred)
                current_precision = max(current_precision, scores[self.type].precision)
                current_recall = max(current_recall, scores[self.type].recall)
                current_f1 = max(current_f1, scores[self.type].fmeasure)

            result_precision += current_precision
            result_recall += current_recall
            result_f1 += current_f1

        result_precision = result_precision / len(references)
        result_recall = result_recall / len(references)
        result_f1 = result_f1 / len(references)

        return MetricResult(
            name=self.name,
            score=result_f1,
            details={
                "precision": result_precision,
                "recall": result_recall,
                "f1": result_f1,
                "type": self.type,
            },
        )
