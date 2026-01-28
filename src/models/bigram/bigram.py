import torch
import torch.nn as nn
import torch.nn.functional as F


class Bigram(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [batch_size, seq_len]
        embeddings = self.token_embedding(idx)  # [batch_size, seq_len, d_model]
        logits = self.lm_head(embeddings)  # [batch_size, seq_len, vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        index: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        batch_size = index.size(0)
        is_generating = torch.ones(batch_size, dtype=torch.bool, device=index.device)

        for _ in range(max_new_tokens):
            last_token_emb = self.token_embedding(index[:, -1:])  # [batch_size, 1, d_model]
            logits = self.lm_head(last_token_emb)  # [batch_size, 1, vocab_size]
            logits_last = logits[:, -1, :]  # [batch_size, vocab_size]

            probs = F.softmax(logits_last, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            index = torch.cat((index, index_next), dim=1)

            is_generating = is_generating & (index_next.squeeze(-1) != eos_token_id)
            if not is_generating.any():
                break

        return index
