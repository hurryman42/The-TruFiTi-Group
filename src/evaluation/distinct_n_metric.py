from src.enums.types import SpecialTokensEnum
from .base_evaluation_metric import BaseEvaluationMetric, MetricResult


class DistinctNMetric(BaseEvaluationMetric):
    def __init__(self, n: int = 1):
        assert n >= 1, "n must be at least 1"
        self.n = n

    @property
    def name(self) -> str:
        return f"distinct-{self.n}"

    @staticmethod
    def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def compute(self, generated: list[str], references: list[list[str]] | None = None) -> MetricResult:
        total_ngrams = 0
        unique_ngrams: set[tuple[str, ...]] = set()

        for sequence in generated:
            words = sequence.strip().split()
            words = [w for w in words if w not in {SpecialTokensEnum.BOS, SpecialTokensEnum.EOS}]

            ngrams = self._get_ngrams(words, self.n)
            total_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)

        score = len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "unique_ngrams": len(unique_ngrams),
                "total_ngrams": total_ngrams,
                "n": self.n,
            },
        )
