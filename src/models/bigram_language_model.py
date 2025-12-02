import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim_embedding: int):
        super().__init__()
        self.embedding_to_vocab = nn.Linear(dim_embedding, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # [batch_size, seq_len, dim_embedding] -> [batch_size, seq_len, vocab_size]
        logits = self.embedding_to_vocab(embeddings)

        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size = logits.shape
            # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            logits_flat = logits.view(batch_size * seq_len, vocab_size)

            # [batch_size, seq_len] -> [batch_size * seq_len]
            targets_flat = targets.view(batch_size * seq_len)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(
        self,
        token_embedding,
        pos_encoding,
        idx: torch.Tensor,
        max_new_tokens: int,
        max_context_len: int,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_context = idx[:, -max_context_len:]
            tok_emb = token_embedding(idx_context)
            embeddings = pos_encoding(tok_emb)

            # [batch_size, current_seq_len, dim_embedding] -> [batch_size, current_seq_len, vocab_size]
            logits, _ = self(embeddings)

            # Focus only on last position (what comes next)
            # [batch_size, current_seq_len, vocab_size] -> [batch_size, vocab_size]
            logits_last = logits[:, -1, :]

            # [batch_size, vocab_size] -> [batch_size, vocab_size]
            probs = F.softmax(logits_last, dim=-1)

            # Sample from the distribution
            # [batch_size, vocab_size] -> [batch_size, 1]
            idx_next = torch.multinomial(probs, num_samples=1)

            # [batch_size, current_seq_len] + [batch_size, 1] -> [batch_size, current_seq_len + 1]
            idx = torch.cat((idx, idx_next), dim=1)

            # [batch_size, original_seq_len + max_new_tokens]
        return idx
