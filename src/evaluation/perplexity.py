import torch

from src.enums.types import SpecialTokensEnum
from src.evaluation.base_evaluation_metric import BaseEvaluationMetric, MetricResult


class PerplexityMetric(BaseEvaluationMetric):
    def __init__(self, model, tokenizer, device, seq_len: int, token_embedding=None):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._seq_len = seq_len
        self._token_embedding = token_embedding

    @property
    def name(self) -> str:
        return "perplexity"

    def compute(
        self,
        generated: list[str],
        references: list[list[str]] | None = None,
    ) -> MetricResult:
        bos_id = self._tokenizer.token_to_id(SpecialTokensEnum.BOS)
        eos_id = self._tokenizer.token_to_id(SpecialTokensEnum.EOS)

        all_tokens = []
        for text in generated:
            encoded = self._tokenizer.encode(text).ids
            all_tokens.append(bos_id)
            all_tokens.extend(encoded)
            all_tokens.append(eos_id)

        data = torch.tensor(all_tokens, dtype=torch.long, device=self._device)

        total_loss = 0.0
        total_tokens = 0

        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(data) - self._seq_len, self._seq_len):
                x = data[i : i + self._seq_len].unsqueeze(0)
                y = data[i + 1 : i + self._seq_len + 1].unsqueeze(0)

                if self._token_embedding is not None:
                    embeddings = self._token_embedding(x)
                    logits, _ = self._model(embeddings, y)
                else:
                    logits = self._model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return MetricResult(
            name=self.name,
            score=perplexity,
            details={
                "avg_loss": avg_loss,
                "total_tokens": total_tokens,
            },
        )
