import argparse
import json
import random

import torch

import wandb
from src.config import (
    MODEL_DIR,
    get_data_path,
    get_model_save_path,
    get_model_type,
    get_tokenizer_path,
    get_tokenizer_type,
    load_config,
)
from src.config.utils import recompute_computed_fields
from src.enums import (
    CheckpointEnum,
    DataConfigEnum,
    DataSplitEnum,
    ModelTypeEnum,
    SectionEnum,
    TokenizerTypeEnum,
    TrainingEnum,
    TransformerModelEnum,
)
from src.models.transformer.transformer import TransformerDecoderOnly
from src.training.trainer import TrainingMetrics, train_loop
from src.utils.data_loader import read_file_only_reviews
from src.utils.device import get_device
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_bpe_hugging_face_tokenizer, load_char_tokenizer
from src.utils.training import train_val_test_split
from src.utils.wandb_transfomer_config_override import apply_wandb_overrides


def create_forward_pass():
    def forward_pass(model, x, y):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    return forward_pass


def save_model(model, vocab_size: int, num_params: int, config: dict):
    save_path = get_model_save_path(config, num_params)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = config[SectionEnum.MODEL]
    tokenizer_type = get_tokenizer_type(config)

    checkpoint = {
        CheckpointEnum.MODEL: model.state_dict(),
        CheckpointEnum.NUM_PARAMS: num_params,
        CheckpointEnum.VOCAB_SIZE: vocab_size,
        CheckpointEnum.D_MODEL: model_cfg[TransformerModelEnum.D_MODEL],
        CheckpointEnum.SEQ_LEN: model_cfg[TransformerModelEnum.SEQ_LEN],
        CheckpointEnum.NUM_HEADS: model_cfg[TransformerModelEnum.NUM_HEADS],
        CheckpointEnum.NUM_BLOCKS: model_cfg[TransformerModelEnum.NUM_BLOCKS],
        CheckpointEnum.FF_HIDDEN_DIM: model_cfg[TransformerModelEnum.FF_HIDDEN_DIM],
        CheckpointEnum.DROPOUT: model_cfg[TransformerModelEnum.DROPOUT],
        CheckpointEnum.TOKENIZER_TYPE: str(tokenizer_type),
        CheckpointEnum.DATA_SEED: config[SectionEnum.DATA][DataConfigEnum.SEED],
    }

    torch.save({str(k): v for k, v in checkpoint.items()}, save_path)
    print(f"Model saved to {save_path}")

    return save_path


def save_metrics(metrics: TrainingMetrics, num_params: int):
    params_millions = num_params / 1_000_000
    metrics_path = MODEL_DIR / f"transformer_{params_millions:.1f}M_metrics.json"

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


def print_training_statistics(config: dict, train_data_len: int):
    model_cfg = config[SectionEnum.MODEL]
    training_cfg = config[SectionEnum.TRAINING]

    batch_size = training_cfg[TrainingEnum.BATCH_SIZE]
    seq_len = model_cfg[TransformerModelEnum.SEQ_LEN]
    max_iters = training_cfg[TrainingEnum.MAX_ITERS]

    tokens_per_iter = batch_size * seq_len
    iters_per_epoch = train_data_len // tokens_per_iter
    print(f"Training tokens: {train_data_len:,}".replace(",", "."))
    print(f"Tokens per iteration: {tokens_per_iter:,} (batch_size={batch_size} × seq_len={seq_len})".replace(",", "."))
    print(f"Iterations per epoch: {iters_per_epoch:,}".replace(",", "."))
    print(f"Total epochs: {max_iters / iters_per_epoch:.2f}\n")


def main(config: dict):
    wandb.init(
        project="film-critic-lm",
        entity="the-trufiti-group",
        config=config,
    )

    if wandb.config.get("config"):
        config = load_config(wandb.config.config)
        config = apply_wandb_overrides(config)
        recompute_computed_fields(config)
        wandb.config.update(config, allow_val_change=True)

    device = get_device()
    tokenizer_type = get_tokenizer_type(config)
    tokenizer_path = get_tokenizer_path(config)

    print(f"Using device: {device}")
    print(f"Tokenizer: {tokenizer_type}\n")

    if tokenizer_type == TokenizerTypeEnum.CHAR:
        tokenizer = load_char_tokenizer(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size
    else:
        tokenizer = load_bpe_hugging_face_tokenizer(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

    data_cfg = config[SectionEnum.DATA]

    data_path = get_data_path(config)
    texts = read_file_only_reviews(data_path)
    random.seed(data_cfg[DataConfigEnum.SEED])
    random.shuffle(texts)
    encoded = encode_texts(texts, tokenizer, tokenizer_type)
    print(f"Total tokens: {len(encoded):,}".replace(",", "."))

    train_data, val_data, _ = train_val_test_split(
        encoded,
        data_cfg[DataConfigEnum.TRAIN_SIZE],
        data_cfg[DataConfigEnum.VAL_SIZE],
        data_cfg[DataConfigEnum.TEST_SIZE],
    )

    print_training_statistics(config, len(train_data))

    model_cfg = config[SectionEnum.MODEL]
    model = TransformerDecoderOnly(
        vocab_size,
        embedding_dimension=model_cfg[TransformerModelEnum.D_MODEL],
        num_blocks=model_cfg[TransformerModelEnum.NUM_BLOCKS],
        num_heads=model_cfg[TransformerModelEnum.NUM_HEADS],
        head_dimension=model_cfg[TransformerModelEnum.HEAD_DIM],
        max_seq_len=model_cfg[TransformerModelEnum.SEQ_LEN],
        ff_hidden_dimension=model_cfg[TransformerModelEnum.FF_HIDDEN_DIM],
        dropout=model_cfg[TransformerModelEnum.DROPOUT],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1_000_000:.1f}M)\n".replace(",", "."))

    training_cfg = config[SectionEnum.TRAINING]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg[TrainingEnum.LEARNING_RATE]),
        weight_decay=training_cfg[TrainingEnum.WEIGHT_DECAY],
    )

    forward_pass = create_forward_pass()

    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}
    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        model_cfg[TransformerModelEnum.SEQ_LEN],
        training_cfg[TrainingEnum.BATCH_SIZE],
        training_cfg[TrainingEnum.MAX_ITERS],
        training_cfg[TrainingEnum.EVAL_INTERVAL],
        training_cfg[TrainingEnum.EVAL_ITERS],
        device,
        wandb,
        warmup_iters=training_cfg[TrainingEnum.WARMUP_ITERS],
    )

    save_model(model, vocab_size, num_params, config)
    save_metrics(metrics, num_params)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("config", type=str, default="transformer_default", help="Config file path")

    args = parser.parse_args()

    # TODO change config dict to a DTO
    config = load_config(args.config)
    print(f"Loading config: {args.config}\n")

    if get_model_type(config) != ModelTypeEnum.TRANSFORMER:
        raise ValueError(f"Config '{args.config}' is not a transformer config")

    main(config)
