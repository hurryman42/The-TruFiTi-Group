from collections import Counter
from typing import List
from .base_evaluation_metric import BaseEvaluationMetric, MetricResult

class DistinctNMetric(BaseEvaluationMetric):
    def __init__(self, n: int = 1):
        assert n >= 1, "n must be at least 1"
        self._n = n

    @property
    def name(self) -> str:
        return f"distinct-{self._n}"

    def compute(
        self,
        generated: List[str],
        references: List[List[str]] = None,
    ) -> MetricResult:
        all_ngrams = []
        total_ngrams = 0

        for sequence in generated:
            # If your sequences are strings of space-separated tokens, split them:
            tokens = sequence.split()
            ngrams = [
                tuple(tokens[i : i + self._n])
                for i in range(len(tokens) - self._n + 1)
            ]
            all_ngrams.extend(ngrams)
            total_ngrams += len(ngrams)

        unique_ngrams = set(all_ngrams)
        score = len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

        details = {
            "unique_ngrams": len(unique_ngrams),
            "total_ngrams": total_ngrams,
            "distinct_ngrams": unique_ngrams,
            "n": self._n,
        }

        return MetricResult(name=self.name, score=score, details=details)