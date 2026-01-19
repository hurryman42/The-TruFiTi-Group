from pathlib import Path

import torch

from src.dto.config import Config
from src.enums import TransformerCheckpointEnum
from src.models.transformer.transformer import TransformerDecoderOnly


def load_transformer_from_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = Config.from_dict(checkpoint[TransformerCheckpointEnum.CONFIG])
    tokenizer = checkpoint[TransformerCheckpointEnum.TOKENIZER]
    vocab_size = checkpoint[TransformerCheckpointEnum.VOCAB_SIZE]

    print(f"Tokenizer: {config.tokenizer.name}")
    print(f"Vocab size: {vocab_size}, d_model: {config.model.d_model}, seq_len: {config.model.seq_len}")
    print(f"Blocks: {config.model.num_blocks}, Heads: {config.model.num_heads}\n")

    model = TransformerDecoderOnly(
        vocab_size,
        embedding_dimension=config.model.d_model,
        num_blocks=config.model.num_blocks,
        num_heads=config.model.num_heads,
        head_dimension=config.model.head_dim,
        max_seq_len=config.model.seq_len,
        ff_hidden_dimension=config.model.ff_hidden_dim,
        dropout=config.model.dropout,
        use_rope=config.model.use_rope,
    ).to(device)

    model.load_state_dict(checkpoint[TransformerCheckpointEnum.MODEL])
    model.eval()

    return model, tokenizer, config
