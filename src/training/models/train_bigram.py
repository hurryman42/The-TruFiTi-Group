import argparse
import torch
import wandb

from src.config import load_config
from src.config.config import Config, BigramModelConfig, BigramTrainingConfig
from src.enums import BigramCheckpointEnum, ModelTypeEnum, DataSplitEnum
from src.models.bigram.bigram import Bigram
from src.utils.device import get_device
from src.training.trainer import train_loop
from src.training.train_utils import (
    load_and_prepare_data,
    print_training_statistics,
    save_model_checkpoint,
    save_metrics,
)
from dataclasses import asdict

from src.utils.wandb_transfomer_config_override import apply_wandb_overrides


def create_forward_pass():
    def forward_pass(model, x, y):
        _, loss = model(x, y)
        return loss

    return forward_pass


def main(config: Config):
    assert isinstance(config.model, BigramModelConfig)
    assert isinstance(config.training, BigramTrainingConfig)
    model_config: BigramModelConfig = config.model
    train_config: BigramTrainingConfig = config.training

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
    print(f"Tokenizer: {config.tokenizer.type}\n")

    tokenizer, vocab_size, train_data, val_data = load_and_prepare_data(config)

    model = Bigram(vocab_size, model_config.d_model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n".replace(",", "."))

    print_training_statistics(train_config.batch_size, train_config.seq_len, train_config.max_iters, train_data.numel())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
    )

    forward_pass = create_forward_pass()
    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}

    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        train_config.seq_len,
        train_config.batch_size,
        train_config.max_iters,
        train_config.eval_interval,
        train_config.eval_iters,
        device,
        wandb,
    )

    checkpoint = {
        BigramCheckpointEnum.MODEL: model.state_dict(),
        BigramCheckpointEnum.VOCAB_SIZE: vocab_size,
        BigramCheckpointEnum.CONFIG: asdict(config),
        BigramCheckpointEnum.TOKENIZER: tokenizer,
    }

    model_save_path = save_model_checkpoint(config, total_params, checkpoint)
    save_metrics(metrics, model_save_path)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bigram Language Model")
    parser.add_argument("config", type=str, help="Config name")

    args = parser.parse_args()

    config = load_config(args.config)
    assert isinstance(config.model, BigramModelConfig)
    print(f"Loading config: {args.config}\n")

    if config.model.type != ModelTypeEnum.BIGRAM:
        raise ValueError(f"Config '{args.config}' is not a bigram config")

    main(config)
