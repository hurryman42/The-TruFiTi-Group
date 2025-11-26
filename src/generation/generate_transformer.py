import argparse
from pathlib import Path

import torch

from src.enums import TokenizerType
from src.models.transformer.transformer import TransformerDecoderOnly
from src.utils.device import get_device
from src.utils.tokenizer_loader import load_bpe_hugging_face_tokenizer, load_char_tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_model(model_path: Path):
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)

    tokenizer_type = TokenizerType(checkpoint["tokenizer_type"])
    vocab_size = checkpoint["vocab_size"]
    d_model = checkpoint["d_model"]
    seq_len = checkpoint["seq_len"]
    num_heads = checkpoint["num_heads"]
    num_blocks = checkpoint["num_blocks"]
    ff_hidden_dim = checkpoint["ff_hidden_dim"]
    dropout = checkpoint["dropout"]

    print(f"Tokenizer: {tokenizer_type.value}")
    print(f"Vocab size: {vocab_size}, d_model: {d_model}, seq_len: {seq_len}")
    print(f"Blocks: {num_blocks}, Heads: {num_heads}\n")

    if tokenizer_type == TokenizerType.CHAR:
        tokenizer_path = BASE_DIR / "tokenizer" / "char_tokenizer.json"
        tokenizer = load_char_tokenizer(tokenizer_path)
    else:
        tokenizer_path = BASE_DIR / "tokenizer" / "bpe_hugging_face_tokenizer.json"
        tokenizer = load_bpe_hugging_face_tokenizer(tokenizer_path)

    model = TransformerDecoderOnly(
        vocab_size,
        embedding_dimension=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        head_dimension=d_model // num_heads,
        max_seq_len=seq_len,
        ff_hidden_dimension=ff_hidden_dim,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, tokenizer, tokenizer_type, seq_len, device


def generate(model, tokenizer, tokenizer_type, seq_len, device, prompt: str = "", length: int = 200) -> str:
    if prompt:
        if tokenizer_type == TokenizerType.CHAR:
            idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        else:
            encoded = tokenizer.encode(prompt)
            idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(idx, length, max_context_len=seq_len)

    if tokenizer_type == TokenizerType.CHAR:
        return tokenizer.decode(generated[0].tolist())
    return tokenizer.decode(generated[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Transformer Language Model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model filename (e.g. transformer_4.8m.pt)",
    )
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--length", type=int, default=200, help="Number of tokens to generate")

    args = parser.parse_args()

    model_path = BASE_DIR / "models" / args.model
    model, tokenizer, tokenizer_type, seq_len, device = load_model(model_path)

    print("=" * 80)
    print(f"Prompt: '{args.prompt}'" if args.prompt else "Unconditional generation:")
    print("=" * 80)
    print(generate(model, tokenizer, tokenizer_type, seq_len, device, args.prompt, args.length))
