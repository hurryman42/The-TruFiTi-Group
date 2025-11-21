import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward


class Block(nn.Module):
    def __init__(self, embedding_dimension, num_heads, head_dimension, block_size, ff_hidden_dimension, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadAttention(embedding_dimension, num_heads, head_dimension, block_size)
        self.layernorm2 = nn.LayerNorm(embedding_dimension)
        self.feedforward = FeedForward(embedding_dimension, ff_hidden_dimension, dropout)

    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x
