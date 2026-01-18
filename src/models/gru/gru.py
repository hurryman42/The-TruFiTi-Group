import torch.nn as nn
import torch
from src.models.embeddings.token_embedding import TokenEmbedding
import torch.nn.functional as F


class GRULanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = TokenEmbedding(vocab_size, input_size, scale=False)
        self.embed_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor, hidden=None):
        embeds = self.embed_dropout(self.embedding(x))
        output, hidden = self.gru(embeds, hidden)
        output = self.layer_norm(output)
        logits = self.fc_out(output)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
    ):
        generated = prompt_ids.clone()
        batch_size = generated.size(0)

        is_generating = torch.ones(batch_size, dtype=torch.bool, device=generated.device)

        logits, hidden = self.forward(generated)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            is_generating = is_generating & (next_token.squeeze(-1) != eos_token_id)

            if not is_generating.any():
                break

            logits, hidden = self.forward(next_token, hidden)

        return generated
