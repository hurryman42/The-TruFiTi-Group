from rouge_score import rouge_scorer
from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult

class RougeNMetric(BaseEvaluationMetric):
    def __init__(self, type: str):
        assert type in ["rouge1", "rouge2", "rougeL"]
        self.type = type

    @property
    def name(self) -> str:
        return f"{self.type}-score"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:

        assert references is not None, "references must be provided"

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        result_precision, result_recall, result_f1 = 0.0, 0.0, 0.0

        for i in range(len(generated)):
            sum_precision, sum_recall, sum_f1 = 0.0, 0.0, 0.0

            for j in range(len(references[i])):
                scores = scorer.score(generated[i], references[i][j])
                sum_precision += scores[self.type][0]
                sum_recall += scores[self.type][1]
                sum_f1 += scores[self.type][2]

            result_precision += sum_precision / len(references[i])
            result_recall += sum_recall / len(references[i])
            result_f1 += sum_f1 / len(references[i])

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
            }
        )