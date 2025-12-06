import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embedding_dimension,
        head_dimension,
        max_seq_len,
        dropout=0.1,
        use_rope=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.use_rope = use_rope

        total_head_dim = num_heads * head_dimension

        self.W_q = nn.Linear(embedding_dimension, total_head_dim, bias=False)
        self.W_k = nn.Linear(embedding_dimension, total_head_dim, bias=False)
        self.W_v = nn.Linear(embedding_dimension, total_head_dim, bias=False)
        self.W_o = nn.Linear(total_head_dim, embedding_dimension, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionalEmbeddings(dim=head_dimension, max_seq_len=max_seq_len)

        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)  # (batch, seq, num_heads * head_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dimension)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dimension)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dimension)

        if self.use_rope:
            Q = self.rope(Q)  # (batch, seq, num_heads, head_dim)
            K = self.rope(K)

        Q = Q.transpose(1, 2)  # (batch, num_heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attention_scores = Q @ K.transpose(-2, -1)  # (batch, num_heads, seq, seq)
        attention_scores = attention_scores / (self.head_dimension**0.5)
        attention_scores = attention_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        out = attention_weights @ V  # (batch, num_heads, seq, head_dim)
        out = out.transpose(1, 2)  # (batch, seq, num_heads, head_dim)
        out = out.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dimension)
        out = self.W_o(out)  # (batch, seq, embedding_dim)

        return out
