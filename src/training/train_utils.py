import json
from pathlib import Path

import torch
import random

from src.config import get_data_path, get_model_save_path
from src.config.config import Config
from src.utils import read_file_synopsis_review_pairs
from src.utils.data_loader import read_file_only_reviews
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.training import train_val_test_split
from src.training.trainer import TrainingMetrics


def save_metrics(metrics: TrainingMetrics, model_save_path: Path):
    metrics_path = model_save_path.with_name(model_save_path.stem + "_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "steps": metrics.steps,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "train_perplexity": metrics.train_perplexity,
                "val_perplexity": metrics.val_perplexity,
            },
            f,
            indent=2,
        )

    print(f"Metrics saved to {metrics_path}")


def print_training_statistics(config: Config, train_data_len: int):
    batch_size = config.training.batch_size
    seq_len = config.model.seq_len
    max_iters = config.training.max_iters

    tokens_per_iter = batch_size * seq_len
    iters_per_epoch = train_data_len // tokens_per_iter
    print(f"Training tokens: {train_data_len:,}".replace(",", "."))
    print(f"Tokens per iteration: {tokens_per_iter:,} (batch_size={batch_size} × seq_len={seq_len})".replace(",", "."))
    print(f"Iterations per epoch: {iters_per_epoch:,}".replace(",", "."))
    print(f"Total epochs: {max_iters / iters_per_epoch:.2f}\n")


def load_and_prepare_data(config: Config):
    tokenizer = load_tokenizer(config.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    data_path = get_data_path(config.data.file)

    print(f"Level: {config.data.level}")
    match config.data.level:
        case 1:
            texts = read_file_only_reviews(data_path)
        case 2:
            texts = read_file_synopsis_review_pairs(data_path)
        case _:
            raise ValueError(f"Invalid level input: {config.data.level}")

    random.seed(config.data.seed)
    random.shuffle(texts)

    train_texts, val_texts, _ = train_val_test_split(
        texts,
        config.data.train_size,
        config.data.val_size,
        config.data.test_size,
    )

    train_data = torch.tensor(encode_texts(train_texts, tokenizer, config.tokenizer.type), dtype=torch.long)
    val_data = torch.tensor(encode_texts(val_texts, tokenizer, config.tokenizer.type), dtype=torch.long)

    return tokenizer, vocab_size, train_data, val_data


def save_model_checkpoint(config: Config, num_params: int, checkpoint_dict: dict) -> Path:
    save_path = get_model_save_path(config, num_params)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint_dict, save_path)
    print(f"Model saved to {save_path}")

    return save_path
