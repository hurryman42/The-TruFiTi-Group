import sys
from pathlib import Path

import yaml

import wandb

from src.config import load_config
from src.enums import ModelTypeEnum
from src.training.models.train_transformer import main as train_transformer
from src.training.models.train_gru import main as train_gru

SWEEP_CONFIG_DIR = Path("src/config/wandb_sweep")


def train_wrapper():
    with wandb.init():
        config_name = wandb.config.config
        config = load_config(config_name)

        match config.model.type:
            case ModelTypeEnum.TRANSFORMER:
                train_transformer(config)
            case ModelTypeEnum.GRU:
                train_gru(config)
            case _:
                raise ValueError(f"Invalid model type: {config.model.type}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.training.run_sweep <sweep_config>")

    sweep_config_path = SWEEP_CONFIG_DIR / sys.argv[1]
    if not sweep_config_path.exists():
        sys.exit(f"Sweep config not found: {sweep_config_path}")

    sweep_config = yaml.safe_load(sweep_config_path.read_text())
    sweep_id = wandb.sweep(sweep_config, project="film-critic-lm", entity="the-trufiti-group")

    wandb.agent(sweep_id, function=train_wrapper)


if __name__ == "__main__":
    main()
