import torch
import torch.nn as nn
from src.models.transformer.attention import MultiHeadAttention
from src.models.transformer.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension, num_heads, head_dimension, max_seq_len, ff_hidden_dimension, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadAttention(num_heads, embedding_dimension, head_dimension, max_seq_len)
        self.layernorm2 = nn.LayerNorm(embedding_dimension)
        self.feedforward = FeedForward(embedding_dimension, ff_hidden_dimension, dropout)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))
        return x
