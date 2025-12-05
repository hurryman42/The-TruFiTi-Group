import argparse
from pathlib import Path

import torch

from src.enums import CheckpointEnum, TokenizerTypeEnum
from src.enums.types import SpecialTokensEnum
from src.models.transformer.transformer import TransformerDecoderOnly
from src.utils.device import get_device
from src.utils.tokenizer_loader import load_bpe_hugging_face_tokenizer, load_char_tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_model(model_path: Path):
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)

    tokenizer_type = TokenizerTypeEnum(checkpoint[CheckpointEnum.TOKENIZER_TYPE])

    print(f"Tokenizer: {tokenizer_type}")
    print(
        f"Vocab size: {checkpoint[CheckpointEnum.VOCAB_SIZE]}, "
        f"d_model: {checkpoint[CheckpointEnum.D_MODEL]}, "
        f"seq_len: {checkpoint[CheckpointEnum.SEQ_LEN]}\n"
    )
    print(f"Blocks: {checkpoint[CheckpointEnum.NUM_BLOCKS]}, Heads: {checkpoint[CheckpointEnum.NUM_HEADS]}\n")

    if tokenizer_type == TokenizerTypeEnum.CHAR:
        tokenizer_path = BASE_DIR / "tokenizer" / "char_tokenizer.json"
        tokenizer = load_char_tokenizer(tokenizer_path)
    else:
        tokenizer_path = BASE_DIR / "tokenizer" / "bpe_hugging_face_tokenizer.json"
        tokenizer = load_bpe_hugging_face_tokenizer(tokenizer_path)

    model = TransformerDecoderOnly(
        checkpoint[CheckpointEnum.VOCAB_SIZE],
        embedding_dimension=checkpoint[CheckpointEnum.D_MODEL],
        num_blocks=checkpoint[CheckpointEnum.NUM_BLOCKS],
        num_heads=checkpoint[CheckpointEnum.NUM_HEADS],
        head_dimension=checkpoint[CheckpointEnum.D_MODEL] // checkpoint[CheckpointEnum.NUM_HEADS],
        max_seq_len=checkpoint[CheckpointEnum.SEQ_LEN],
        ff_hidden_dimension=checkpoint[CheckpointEnum.FF_HIDDEN_DIM],
        dropout=checkpoint[CheckpointEnum.DROPOUT],
    ).to(device)

    model.load_state_dict(checkpoint[CheckpointEnum.MODEL])
    model.eval()

    return model, tokenizer, tokenizer_type, device


def generate(model, tokenizer, tokenizer_type, device, prompt: str = "", length: int = 100) -> str:
    if prompt:
        if tokenizer_type == TokenizerTypeEnum.CHAR:
            idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        else:
            encoded = tokenizer.encode(prompt)
            idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        if tokenizer_type == TokenizerTypeEnum.CHAR:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            bos_id = tokenizer.token_to_id(SpecialTokensEnum.BOS)
            idx = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id(SpecialTokensEnum.EOS)
    generated = model.generate(idx, length, eos_id)

    tokens = generated[0].tolist()
    if tokenizer_type == TokenizerTypeEnum.CHAR:
        return tokenizer.decode(tokens)
    
    if eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Transformer Language Model")
    parser.add_argument("--model", type=str, required=True, help="Model filename")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")

    args = parser.parse_args()

    model_path = BASE_DIR / "models" / args.model
    model, tokenizer, tokenizer_type, device = load_model(model_path)

    print("=" * 80)
    print(
        f"Prompt: '{args.prompt}'"
        if args.prompt
        else f"Unconditional generation: (Starting with {SpecialTokensEnum.BOS}):"
    )
    print("=" * 80)
    print(generate(model, tokenizer, tokenizer_type, device, args.prompt, args.length))
