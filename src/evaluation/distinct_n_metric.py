from typing import List
from .base_evaluation_metric import BaseEvaluationMetric, MetricResult


def get_ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

class DistinctNMetric(BaseEvaluationMetric):
    def __init__(self, n: int = 1):
        assert n >= 1, "n must be at least 1"
        self.n = n

    @property
    def name(self) -> str:
        return f"distinct-{self.n}"

    def compute(self, generated: List[str], references: List[List[str]] = None) -> MetricResult:
        total_ngrams = 0
        unique_ngrams = set()
        all_ngrams = []

        for sequence in generated:
            tokens = sequence.strip().split()
            tokens = [t for t in tokens if t not in {"<BOS>", "<EOS>"}]
            ngrams = get_ngrams(tokens, self.n)
            total_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)
            all_ngrams.extend(ngrams)

        score = len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "unique_ngrams": len(unique_ngrams),
                "total_ngrams": total_ngrams,
                "distinct_ngrams": unique_ngrams,
                "n": self.n,
            }
        )