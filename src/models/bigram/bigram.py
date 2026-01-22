import torch
import torch.nn as nn
import torch.nn.functional as F


class Bigram(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: [batch_size, seq_len]
        embeddings = self.token_embedding(idx)  # [batch_size, seq_len, d_model]

        logits = self.lm_head(embeddings)  # [batch_size, seq_len, vocab_size]

        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        batch_size = idx.size(0)
        is_generating = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits_last = logits[:, -1, :]  # [batch_size, vocab_size]

            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            idx = torch.cat((idx, idx_next), dim=1)

            is_generating = is_generating & (idx_next.squeeze(-1) != eos_token_id)
            if not is_generating.any():
                break

        return idx
