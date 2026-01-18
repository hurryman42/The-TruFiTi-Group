import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from src.enums.types import SpecialTokensEnum, ModelTypeEnum
from src.utils.device import get_device
from src.generation.generate_utils import (
    prepare_prompts,
    decode_generated,
    extract_completions,
    print_generation_header,
    print_results,
    load_model_checkpoint,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def generate(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    prompts: list[str],
    length: int,
    model_type: ModelTypeEnum,
    token_embedding=None,
    config=None,
) -> list[str]:
    if not prompts:
        return []

    eos_id = tokenizer.token_to_id(SpecialTokensEnum.EOS)
    idx = prepare_prompts(prompts, tokenizer, device)

    match model_type:
        case ModelTypeEnum.BIGRAM:
            generated = model.generate(token_embedding, idx, eos_id, length, max_context_len=config.model.seq_len)
        case ModelTypeEnum.GRU | ModelTypeEnum.TRANSFORMER:
            generated = model.generate(idx, length, eos_id)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    return decode_generated(generated, tokenizer, eos_id)


def generate_completions(
    model,
    tokenizer: Tokenizer,
    device: torch.device,
    prompts: list[str],
    length: int,
    model_type: ModelTypeEnum,
    token_embedding=None,
    config=None,
) -> list[str]:
    full_texts = generate(model, tokenizer, device, prompts, length, model_type, token_embedding, config)
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

    # Load model (returns different things based on type)
    model_data = load_model_checkpoint(model_path, device, model_type)

    print_generation_header(args.prompt)

    prompts = [args.prompt] * args.num

    match model_type:
        case ModelTypeEnum.BIGRAM:
            model, token_embedding, tokenizer, config = model_data
            results = generate(model, tokenizer, device, prompts, args.length, model_type, token_embedding, config)
        case ModelTypeEnum.GRU | ModelTypeEnum.TRANSFORMER:
            model, tokenizer, config = model_data
            results = generate(model, tokenizer, device, prompts, args.length, model_type)

    print_results(results, args.num)
