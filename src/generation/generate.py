import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from src.enums.types import SpecialTokensEnum, ModelTypeEnum
from src.utils.device import get_device
from src.generation.generate_utils import (
    decode_generated,
    print_generation_header,
    print_results,
    load_model_checkpoint,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def prepare_prompts(
    prompts: list[str],
    tokenizer: Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    if not prompts:
        return torch.empty(0, dtype=torch.long, device=device)

    bos_id = tokenizer.token_to_id(SpecialTokensEnum.BOS)
    pad_id = tokenizer.token_to_id(SpecialTokensEnum.PAD)

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

    return torch.tensor(padded, dtype=torch.long, device=device)


def generate(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    prompts: list[str],
    length: int,
) -> list[str]:
    if not prompts:
        return []

    eos_id = tokenizer.token_to_id(SpecialTokensEnum.EOS)
    idx = prepare_prompts(prompts, tokenizer, device)

    generated = model.generate(index=idx, eos_token_id=eos_id, max_new_tokens=length)
    return decode_generated(generated, tokenizer, eos_id)


def extract_completions(
    full_texts: list[str],
    prompts: list[str],
) -> list[str]:
    completions = []
    for text, prompt in zip(full_texts, prompts, strict=False):
        if prompt and text.startswith(prompt):
            completions.append(text[len(prompt) :])
        else:
            completions.append(text)
    return completions


def generate_completions(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    prompts: list[str],
    length: int,
) -> list[str]:
    full_texts = generate(model, tokenizer, device, prompts, length)
    return extract_completions(full_texts, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with Language Models")
    parser.add_argument("--model", type=str, required=True, help="Model filename")
    parser.add_argument("--type", type=str, required=True, choices=["bigram", "gru", "transformer"], help="Model type")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num", "-n", type=int, default=1, help="Number of generations")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model_path = BASE_DIR / "models" / args.model

    model_type_map = {
        "bigram": ModelTypeEnum.BIGRAM,
        "gru": ModelTypeEnum.GRU,
        "transformer": ModelTypeEnum.TRANSFORMER,
    }
    model_type = model_type_map[args.type]

    model, tokenizer, config = load_model_checkpoint(model_path, device, model_type)

    print_generation_header(args.prompt)

    prompts = [args.prompt] * args.num
    results = generate(model, tokenizer, device, prompts, args.length)

    print_results(results, args.num)
