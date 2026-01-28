import argparse
from dataclasses import asdict

import torch
import wandb

from src.config import load_config
from src.config.config import Config, TransformerModelConfig, TransformerTrainingConfig
from src.enums import ModelTypeEnum, TransformerCheckpointEnum, DataSplitEnum
from src.models.transformer.transformer import TransformerDecoderOnly
from src.utils.device import get_device
from src.utils.wandb_transfomer_config_override import apply_wandb_overrides
from src.training.trainer import train_loop
from src.training.train_utils import (
    load_and_prepare_data,
    print_training_statistics,
    save_model_checkpoint,
    save_metrics,
)


def create_forward_pass():
    def forward_pass(model, x, y):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    return forward_pass


def main(config: Config):
    assert isinstance(config.model, TransformerModelConfig)
    assert isinstance(config.training, TransformerTrainingConfig)
    model_config: TransformerModelConfig = config.model
    train_config: TransformerTrainingConfig = config.training

    if wandb.run is None:
        wandb.init(
            project="film-critic-lm",
            entity="the-trufiti-group",
            config=asdict(config),
        )

    if wandb.config.get("config"):
        config = load_config(wandb.config.config)
        config = apply_wandb_overrides(config)
        wandb.config.update(asdict(config), allow_val_change=True)

    device = get_device()

    print(f"Using device: {device}")
    print(f"Tokenizer: {config.tokenizer.name}\n")

    tokenizer, vocab_size, train_data, val_data = load_and_prepare_data(config)

    print(f"Total tokens: {len(train_data) + len(val_data):,}".replace(",", "."))
    print_training_statistics(train_config.batch_size, model_config.seq_len, train_config.max_iters, train_data.numel())

    model = TransformerDecoderOnly(
        vocab_size,
        embedding_dimension=model_config.d_model,
        num_blocks=model_config.num_blocks,
        num_heads=model_config.num_heads,
        head_dimension=model_config.head_dim,
        max_seq_len=model_config.seq_len,
        ff_hidden_dimension=model_config.ff_hidden_dim,
        dropout=model_config.dropout,
        use_rope=model_config.use_rope,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1_000_000:.1f}M)\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    forward_pass = create_forward_pass()
    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}
    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        model_config.seq_len,
        train_config.batch_size,
        train_config.max_iters,
        train_config.eval_interval,
        train_config.eval_iters,
        device,
        wandb,
        warmup_iters=train_config.warmup_iters,
    )

    checkpoint = {
        str(TransformerCheckpointEnum.MODEL): model.state_dict(),
        str(TransformerCheckpointEnum.VOCAB_SIZE): vocab_size,
        str(TransformerCheckpointEnum.NUM_PARAMS): num_params,
        str(TransformerCheckpointEnum.CONFIG): asdict(config),
        str(TransformerCheckpointEnum.TOKENIZER): tokenizer,
    }

    model_save_path = save_model_checkpoint(config, num_params, checkpoint)
    save_metrics(metrics, model_save_path)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("config", type=str, help="Config name")

    args = parser.parse_args()

    config = load_config(args.config)
    assert isinstance(config.model, TransformerModelConfig)
    print(f"Loading config: {args.config}\n")

    if config.model.type != ModelTypeEnum.TRANSFORMER:
        raise ValueError(f"Config '{args.config}' is not a transformer config")

    main(config)
