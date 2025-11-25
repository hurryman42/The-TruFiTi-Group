import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dimension, head_dimension, max_seq_len):
        super().__init__()
        self.key = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.query = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.value = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
        self.head_dimension = head_dimension

    def forward(self, x):
        batch_size, seq_len, embedding_dimension = x.shape
        K = self.key(x)     # (batch, seq, head_dim)
        Q = self.query(x)   # (batch, seq, head_dim)
        V = self.value(x)   # (batch, seq, head_dim)

        attention_scores = Q @ K.transpose(-2, -1) / (self.head_dimension ** 0.5)  # =(QK^T)/(\sqrt{d}) (batch, seq, seq)

        # causal mask (blocks out the future)
        attention_scores = attention_scores.masked_fill(self.mask[:seq_len, :seq_len], float('-inf'))    # (batch, seq, seq)

        attention_probabilities = F.softmax(attention_scores, dim=-1)           # (batch, seq, seq)
        return attention_probabilities @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dimension, head_dimension, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dimension, head_dimension, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_dimension, embedding_dimension) # output projection: combine multiple heads

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # runs each head in parallel, then concatenates
        out = self.proj(out)
        return out
