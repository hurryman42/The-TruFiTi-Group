import argparse
import random
from pathlib import Path

from src.config import get_data_path, load_config
from src.enums import DataConfigEnum, SectionEnum
from src.evaluation.bert_score import BERTScoreMetric
from src.evaluation.distinct_n_metric import DistinctNMetric
from src.evaluation.perplexity import PerplexityMetric
from src.generation.generate_transformer import generate_batch, generate_completions_batch
from src.utils import get_device, train_val_test_split
from src.utils.data_loader import read_file_only_reviews
from src.utils.load_transformer import load_checkpoint, load_model_tokenizer_from_transformer_checkpoint

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def split_review_half(review: str) -> tuple[str, str]:
    words = review.split()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def evaluate(
    model,
    tokenizer,
    device,
    test_texts: list[str],
    seq_len: int,
    num_samples: int = 100,
    gen_length: int = 50,
    seed: int = 42,
) -> dict:
    perplexity_metric = PerplexityMetric(model, tokenizer, device, seq_len)
    ppl_result = perplexity_metric.compute(test_texts)

    random.seed(seed)
    unconditional_prompts = [""] * num_samples
    generated_texts = generate_batch(model, tokenizer, device, unconditional_prompts, gen_length)

    d1_result = DistinctNMetric(n=1).compute(generated_texts)
    d2_result = DistinctNMetric(n=2).compute(generated_texts)

    samples = random.sample(test_texts, min(num_samples, len(test_texts)))

    prompts, references = [], []
    for review in samples:
        prompt, reference = split_review_half(review)
        if prompt and reference:
            prompts.append(prompt)
            references.append(reference)

    completions = generate_completions_batch(model, tokenizer, device, prompts, gen_length)
    bert_result = BERTScoreMetric().compute(completions, [[ref] for ref in references])

    print(f"Perplexity:  {ppl_result.score:.2f}")
    print(f"Distinct-1:  {d1_result.score:.4f}")
    print(f"Distinct-2:  {d2_result.score:.4f}")
    print(f"BERTScore:   {bert_result.score:.4f}")

    return {
        "perplexity": ppl_result.score,
        "distinct_1": d1_result.score,
        "distinct_2": d2_result.score,
        "bertscore_f1": bert_result.score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Transformer Language Model")
    parser.add_argument("--model", type=str, required=True, help="Model filename")
    parser.add_argument("--config", type=str, required=True, help="Config name")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--gen_length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config[SectionEnum.DATA]

    device = get_device()
    print(f"Device: {device}")

    model_path = BASE_DIR / "models" / args.model
    checkpoint = load_checkpoint(model_path, device)
    model, tokenizer = load_model_tokenizer_from_transformer_checkpoint(checkpoint, device)

    data_path = get_data_path(config)
    texts = read_file_only_reviews(data_path)
    random.seed(data_cfg[DataConfigEnum.SEED])
    random.shuffle(texts)

    _, _, test_texts = train_val_test_split(
        texts,
        data_cfg[DataConfigEnum.TRAIN_SIZE],
        data_cfg[DataConfigEnum.VAL_SIZE],
        data_cfg[DataConfigEnum.TEST_SIZE],
    )

    assert test_texts is not None, "Test split configuration resulted in None."

    print(f"Test reviews: {len(test_texts)}")

    evaluate(
        model,
        tokenizer,
        device,
        test_texts,
        seq_len=model.block_size,
        num_samples=args.num_samples,
        gen_length=args.gen_length,
        seed=args.seed,
    )
