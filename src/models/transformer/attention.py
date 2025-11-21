import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, embedding_dimension, head_dimension, block_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.query = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.value = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.head_dimension = head_dimension

    def forward(self, x):
        batch_size, seq_length, embedding_dimension = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        attention_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5) # =(QK^T)/(\sqrt{d})

        # causal mask (blocks out the future)
        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)
        return attention_probs @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dimension, head_dimension, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(embedding_dimension, head_dimension, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_dimension, embedding_dimension) # output projection: combine multiple heads

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # runs each head in parallel, then concatenates
        out = self.proj(out)
        return out
