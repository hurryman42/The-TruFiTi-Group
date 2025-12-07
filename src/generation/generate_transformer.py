import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.enums.types import SpecialTokensEnum
from src.models.transformer.transformer import TransformerDecoderOnly
from src.utils.device import get_device
from src.utils.load_transformer import load_checkpoint, load_model_tokenizer_from_transformer_checkpoint

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def generate_single(
    model: TransformerDecoderOnly,
    tokenizer: Tokenizer,
    device: str,
    prompt: str,
    length: int,
) -> str:
    if prompt:
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        bos_id = tokenizer.token_to_id(SpecialTokensEnum.BOS)
        idx = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id(SpecialTokensEnum.EOS)
    generated = model.generate(idx, length, eos_id)

    tokens = generated[0].tolist()
    if eos_id in tokens:
        tokens = tokens[: tokens.index(eos_id)]
    return tokenizer.decode(tokens)


def generate_batch(
    model: TransformerDecoderOnly,
    tokenizer: Tokenizer,
    device: str,
    prompts: list[str],
    length: int,
) -> list[str]:
    if not prompts:
        return []

    bos_id = tokenizer.token_to_id(SpecialTokensEnum.BOS)
    pad_id = tokenizer.token_to_id(SpecialTokensEnum.PAD)
    eos_id = tokenizer.token_to_id(SpecialTokensEnum.EOS)

    encoded = []
    for p in prompts:
        if p:
            encoded.append(tokenizer.encode(p).ids)
        else:
            encoded.append([bos_id])

    max_prompt_len = max(len(e) for e in encoded)
    padded = []
    for e in encoded:
        padding = [pad_id] * (max_prompt_len - len(e))
        padded.append(padding + e)

    idx = torch.tensor(padded, dtype=torch.long, device=device)
    generated = model.generate(idx, length, eos_id)

    results = []
    for tokens in generated.tolist():
        if eos_id in tokens:
            tokens = tokens[: tokens.index(eos_id)]
        results.append(tokenizer.decode(tokens))

    return results


def generate_completions_batch(
    model: TransformerDecoderOnly,
    tokenizer: Tokenizer,
    device: str,
    prompts: list[str],
    length: int,
) -> list[str]:
    full_texts = generate_batch(model, tokenizer, device, prompts, length)

    completions = []
    for text, prompt in zip(full_texts, prompts, strict=False):
        if prompt and text.startswith(prompt):
            completions.append(text[len(prompt) :])
        else:
            completions.append(text)

    return completions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Transformer Language Model")
    parser.add_argument("--model", type=str, required=True, help="Model filename")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model_path = BASE_DIR / "models" / args.model

    checkpoint = load_checkpoint(model_path, device)
    model, tokenizer = load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)

    print("=" * 80)
    if args.prompt:
        print(f"Prompt: '{args.prompt}'")
    else:
        print(f"Unconditional generation (starting with {SpecialTokensEnum.BOS}):")
    print("=" * 80)
    print(generate_single(model, tokenizer, device, args.prompt, args.length))
