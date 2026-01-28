import argparse
import random
from pathlib import Path

from src.config import get_data_path
from src.enums.types import ModelTypeEnum, SpecialTokensEnum
from src.evaluation.bert_score import BERTScoreMetric
from src.evaluation.distinct_n_metric import DistinctNMetric
from src.evaluation.perplexity import PerplexityMetric
from src.evaluation.rouge_n_metric import RougeNMetric
from src.generation.generate_utils import load_model_checkpoint
from src.generation.generate import generate, generate_completions
from src.utils import get_device, train_val_test_split
from src.utils.data_loader import read_file_only_reviews, read_file_synopsis_review_pairs

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def split_review_half(review: str) -> tuple[str, str]:
    words = review.split()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def extract_synopsis_and_review(text: str) -> tuple[str, str]:
    parts = text.split(SpecialTokensEnum.REV)
    if len(parts) != 2:
        return "", ""

    synopsis = parts[0].replace(SpecialTokensEnum.SYN, "").strip()
    review = parts[1].strip()

    return synopsis, review


def evaluate(
    model,
    tokenizer,
    device,
    test_texts: list[str],
    seq_len: int,
    num_samples: int = 100,
    gen_length: int = 50,
    seed: int = 42,
    level: int = 2,
) -> dict:
    perplexity_metric = PerplexityMetric(model, tokenizer, device, seq_len)
    ppl_result = perplexity_metric.compute(test_texts)

    random.seed(seed)
    unconditional_prompts = [""] * num_samples
    generated_texts = generate(model, tokenizer, device, unconditional_prompts, gen_length)

    d1_result = DistinctNMetric(n=1).compute(generated_texts)
    d2_result = DistinctNMetric(n=2).compute(generated_texts)

    samples = random.sample(test_texts, min(num_samples, len(test_texts)))

    prompts, references = [], []

    if level == 2:
        for text in samples:
            synopsis, review = extract_synopsis_and_review(text)
            if synopsis and review:
                prompt = f"{SpecialTokensEnum.SYN} {synopsis} {SpecialTokensEnum.REV}"
                prompts.append(prompt)
                references.append(review)
    else:
        for review in samples:
            prompt, reference = split_review_half(review)
            if prompt and reference:
                prompts.append(prompt)
                references.append(reference)

    completions = generate_completions(model, tokenizer, device, prompts, gen_length)
    references_formatted = [[ref] for ref in references]

    bert_result = BERTScoreMetric().compute(completions, references_formatted)

    rouge1_metric = RougeNMetric(type="rouge1")
    rouge1_result = rouge1_metric.compute(completions, references_formatted)

    rouge2_metric = RougeNMetric(type="rouge2")
    rouge2_result = rouge2_metric.compute(completions, references_formatted)

    print(f"Perplexity:  {ppl_result.score:.2f}")
    print(f"Distinct-1:  {d1_result.score:.4f}")
    print(f"Distinct-2:  {d2_result.score:.4f}")
    print(f"BERTScore:   {bert_result.score:.4f}")
    print(f"ROUGE-1:     {rouge1_result.score:.4f}")
    print(f"ROUGE-2:     {rouge2_result.score:.4f}")

    return {
        "perplexity": ppl_result.score,
        "distinct_1": d1_result.score,
        "distinct_2": d2_result.score,
        "bertscore_f1": bert_result.score,
        "rouge1_f1": rouge1_result.score,
        "rouge2_f1": rouge2_result.score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Language Model")
    parser.add_argument("--model", type=str, required=True, help="Model filename")
    parser.add_argument("--type", type=str, required=True, choices=["bigram", "gru", "transformer"], help="Model type")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gen_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_type_map = {
        "bigram": ModelTypeEnum.BIGRAM,
        "gru": ModelTypeEnum.GRU,
        "transformer": ModelTypeEnum.TRANSFORMER,
    }
    model_type = model_type_map[args.type]

    model_path = BASE_DIR / "models" / args.model
    model, tokenizer, config = load_model_checkpoint(model_path, device, model_type)

    data_path = get_data_path(config.data.file)
    if args.level == 2:
        texts = read_file_synopsis_review_pairs(data_path)
    else:
        texts = read_file_only_reviews(get_data_path(config.data.file))

    random.seed(config.data.seed)
    random.shuffle(texts)

    _, _, test_texts = train_val_test_split(
        texts,
        config.data.train_size,
        config.data.val_size,
        config.data.test_size,
    )

    assert test_texts is not None, "Test split configuration resulted in None."

    print(f"Test reviews: {len(test_texts)}")

    if model_type == ModelTypeEnum.BIGRAM:
        seq_len = config.training.seq_len
    else:
        seq_len = config.model.seq_len

    evaluate(
        model,
        tokenizer,
        device,
        test_texts,
        seq_len=seq_len,
        num_samples=args.num_samples,
        gen_length=args.gen_length,
        seed=args.seed,
        level=args.level,
    )
