import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension, ff_hidden_dimension=None, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dimension, ff_hidden_dimension),
            nn.GELU(),
            nn.Linear(ff_hidden_dimension, embedding_dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
