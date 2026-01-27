import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings.token_embedding import TokenEmbedding


class GRU(nn.Module):
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
        self.vocab_size = vocab_size

        self.embedding = TokenEmbedding(vocab_size, input_size, scale=False)
        self.embed_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        embeds = self.embedding(x)  # [batch_size, seq_len, input_size]
        embeds = self.embed_dropout(embeds)

        output, _ = self.gru(embeds, hidden)  # output: [batch_size, seq_len, hidden_size]

        output = self.layer_norm(output)
        logits = self.fc_out(output)  # [batch_size, seq_len, vocab_size]

        return logits

    def forward_with_hidden(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        embeds = self.embedding(x)
        embeds = self.embed_dropout(embeds)
        output, hidden = self.gru(embeds, hidden)
        output = self.layer_norm(output)
        logits = self.fc_out(output)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        index: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        self.eval()

        generated = index.clone()
        batch_size = generated.size(0)
        device = generated.device

        is_generating = torch.ones(batch_size, dtype=torch.bool, device=device)

        logits, hidden = self.forward_with_hidden(generated)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            generated = torch.cat([generated, next_token], dim=1)

            is_generating = is_generating & (next_token.squeeze(-1) != eos_token_id)
            if not is_generating.any():
                break

            embeds = self.embedding(next_token)  # [batch_size, 1, input_size]
            embeds = self.embed_dropout(embeds)

            output, hidden = self.gru(embeds, hidden)  # output: [batch_size, 1, hidden_size]
            output = self.layer_norm(output)
            logits = self.fc_out(output)  # [batch_size, 1, vocab_size]

        return generated
