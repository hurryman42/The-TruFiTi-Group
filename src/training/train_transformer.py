import json
from dataclasses import asdict
from pathlib import Path

import torch
import wandb
import random
import argparse

from src.config import get_data_path, load_config, get_model_save_path
from src.dto.config import Config
from src.enums import DataSplitEnum, ModelTypeEnum, TransformerCheckpointEnum
from src.models.transformer.transformer import TransformerDecoderOnly
from src.training.trainer import TrainingMetrics, train_loop
from src.utils import read_file_synopsis_review_pairs
from src.utils.data_loader import read_file_only_reviews
from src.utils.device import get_device
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.training import train_val_test_split
from src.utils.wandb_transfomer_config_override import apply_wandb_overrides


def create_forward_pass():
    def forward_pass(model, x, y):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    return forward_pass


def save_model(model, vocab_size: int, num_params: int, config: Config, tokenizer) -> Path:
    save_path = get_model_save_path(config, num_params)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        TransformerCheckpointEnum.MODEL: model.state_dict(),
        TransformerCheckpointEnum.VOCAB_SIZE: vocab_size,
        TransformerCheckpointEnum.NUM_PARAMS: num_params,
        TransformerCheckpointEnum.CONFIG: asdict(config),
        TransformerCheckpointEnum.TOKENIZER: tokenizer,
    }

    torch.save({str(k): v for k, v in checkpoint.items()}, save_path)
    print(f"Model saved to {save_path}")

    return save_path


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


def main(config: Config):
    wandb.init(
        project="film-critic-lm",
        entity="the-trufiti-group",
        config=asdict(config),
    )

    if wandb.config.get("config"):
        config = Config.from_yaml(wandb.config.config)
        config = apply_wandb_overrides(config)
        wandb.config.update(asdict(config), allow_val_change=True)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Tokenizer: {config.tokenizer.name}\n")

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

    print(f"Total tokens: {len(train_data) + len(val_data):,}".replace(",", "."))

    print_training_statistics(config, len(train_data))

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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1_000_000:.1f}M)\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    forward_pass = create_forward_pass()

    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}
    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        config.model.seq_len,
        config.training.batch_size,
        config.training.max_iters,
        config.training.eval_interval,
        config.training.eval_iters,
        device,
        wandb,
        warmup_iters=config.training.warmup_iters,
    )

    model_save_path = save_model(model, vocab_size, num_params, config, tokenizer)
    save_metrics(metrics, model_save_path)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("--config", type=str, default="transformer_default", help="Config name")

    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loading config: {args.config}\n")

    if config.model.type != ModelTypeEnum.TRANSFORMER:
        raise ValueError(f"Config '{args.config}' is not a transformer config")

    main(config)
