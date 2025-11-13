import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim_embedding: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.dim_embedding = dim_embedding

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # see Attention Is All You Need paper: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
        scale_factor = torch.sqrt(torch.tensor(self.dim_embedding, dtype=torch.float32))
        return self.embedding(tokens) * scale_factor


#