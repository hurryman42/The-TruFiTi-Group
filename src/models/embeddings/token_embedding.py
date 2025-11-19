import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim_embedding: int = 512, scale=True):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, dim_embedding)
        self.dim_embedding = dim_embedding

        if scale:
            # see "Attention Is All You Need"
            self.scale_factor = torch.sqrt(torch.tensor(dim_embedding, dtype=torch.float32))
        else:
            self.scale_factor = 1.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # replacing tokens-ids with vector
        # [batch_size, seq_len] -> [batch_size, seq_len, dim_embedding]
        embeddings = self.token_embedding_table(tokens)

        return embeddings * self.scale_factor
