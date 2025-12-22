from pathlib import Path

import torch

from src.enums import CheckpointEnum, TokenizerTypeEnum
from src.models.transformer.transformer import TransformerDecoderOnly
from src.utils import load_bpe_hugging_face_tokenizer, load_bpe_custom_tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_model_tokenizer_from_transformer_checkpoint(checkpoint: dict, device: str):
    tokenizer_type = checkpoint[CheckpointEnum.TOKENIZER_TYPE]
    tokenizer_name = checkpoint[CheckpointEnum.TOKENIZER_NAME]

    print(f"Tokenizer: {tokenizer_name}")
    print(
        f"Vocab size: {checkpoint[CheckpointEnum.VOCAB_SIZE]}, "
        f"d_model: {checkpoint[CheckpointEnum.D_MODEL]}, "
        f"seq_len: {checkpoint[CheckpointEnum.SEQ_LEN]}\n"
    )
    print(f"Blocks: {checkpoint[CheckpointEnum.NUM_BLOCKS]}, Heads: {checkpoint[CheckpointEnum.NUM_HEADS]}\n")

    tokenizer_path = BASE_DIR / "tokenizer" / tokenizer_name
    if tokenizer_type == TokenizerTypeEnum.BPE_HUGGING_FACE:
        tokenizer = load_bpe_hugging_face_tokenizer(tokenizer_path)
    elif tokenizer_type == TokenizerTypeEnum.BPE:
        tokenizer = load_bpe_custom_tokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    model = TransformerDecoderOnly(
        checkpoint[CheckpointEnum.VOCAB_SIZE],
        embedding_dimension=checkpoint[CheckpointEnum.D_MODEL],
        num_blocks=checkpoint[CheckpointEnum.NUM_BLOCKS],
        num_heads=checkpoint[CheckpointEnum.NUM_HEADS],
        head_dimension=checkpoint[CheckpointEnum.D_MODEL] // checkpoint[CheckpointEnum.NUM_HEADS],
        max_seq_len=checkpoint[CheckpointEnum.SEQ_LEN],
        ff_hidden_dimension=checkpoint[CheckpointEnum.FF_HIDDEN_DIM],
        dropout=checkpoint[CheckpointEnum.DROPOUT],
        use_rope=checkpoint.get(CheckpointEnum.USE_ROPE, False),
    ).to(device)

    model.load_state_dict(checkpoint[CheckpointEnum.MODEL])
    model.eval()

    return model, tokenizer


def load_checkpoint(model_path: Path, device: str):
    return torch.load(model_path, map_location=device, weights_only=False)
