import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implementation of the Sinusoidal Positional Encoding, see "Attention Is All You Need"
    """

    def __init__(
        self,
        dim_embedding: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        position_encoding_table = torch.zeros(
            max_seq_len, dim_embedding
        )  # [max_seq_len, dim_embedding]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(
            dim=1
        )  # [max_seq_len, 1]

        div_term = torch.exp(
            torch.arange(0, dim_embedding, 2).float()
            * (-math.log(10000.0) / dim_embedding)
        )

        # Broadcasting: [max_seq_len, 1] * [dim_embedding/2] -> [max_seq_len, dim_embedding/2]
        position_encoding_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_table[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [max_seq_len, dim_embedding] -> [1, max_seq_len, dim_embedding]
        position_encoding_table = position_encoding_table.unsqueeze(0)

        # Register as buffer so the positional encoding is not passed to the optimizer
        self.register_buffer("position_encoding_table", position_encoding_table)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [1, seq_len, dim_embedding]
        positional_encodings = self.position_encoding_table[:, : x.size(1), :]

        # [batch_size, seq_len, dim_embedding] + [1, seq_len, dim_embedding] -> [batch_size, seq_len, dim_embedding] (broadcasting)
        x = x + positional_encodings

        return self.dropout(x)
