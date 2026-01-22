import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.transformer.block import TransformerBlock


class TransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dimension,
        num_blocks,
        num_heads,
        head_dimension,
        max_seq_len,
        ff_hidden_dimension,
        dropout=0.1,
        use_rope=False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dimension, scale=not use_rope)

        if not use_rope:
            self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dimension, dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension,
                    num_heads,
                    head_dimension,
                    max_seq_len,
                    ff_hidden_dimension,
                    dropout,
                    use_rope,
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(embedding_dimension)  # final layer norm
        self.ln_head = nn.Linear(embedding_dimension, vocab_size)  # output projection

        self.block_size = max_seq_len

    def forward(self, index):
        batch_size, seq_length = index.shape
        assert seq_length <= self.block_size, (
            f"Cannot forward sequence of length {seq_length}, block size is only {self.block_size}"
        )

        x = self.token_embedding(index)  # (batch_size, seq_length, embed_dim)
        if not self.use_rope:
            x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)  # (batch_size, seq_length, embedding_dimension)
        logits = self.ln_head(x)  # (batch_size, seq_length, vocab_size)
        return logits

    @torch.no_grad()  # <-- makes sure that no gradient history is built up during generation
    def generate(self, index, max_new_tokens, eos_token_id, temperature=1.0, top_k=None):
        batch_size = index.size(0)
        is_generating = torch.ones(batch_size, dtype=torch.bool, device=index.device)

        for _ in range(max_new_tokens):
            if not is_generating.any():
                break
            # crop to block_size if sequence context is growing too long
            index_conditional = index if index.size(1) <= self.block_size else index[:, -self.block_size :]
            # (batch_size, seq_length, vocab_size)
            logits = self(index_conditional)
            # (batch_size, vocab_size) --- use last time step
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                mask = logits < v[:, [-1]]
                logits[mask] = -float("inf")

            # converts logits to (normalized) probabilities
            probabilities = F.softmax(logits, dim=-1)
            # sample from distribution
            next_token = torch.multinomial(probabilities, num_samples=1)
            # append sampled index to running sequence
            index = torch.cat((index, next_token), dim=1)

            is_generating = is_generating & (next_token.squeeze(-1) != eos_token_id)

            if not is_generating.any():
                break

        return index
