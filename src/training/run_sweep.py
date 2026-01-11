import sys
from pathlib import Path

import yaml

import wandb

from src.config import load_config
from src.training.train_transformer import main as train

SWEEP_CONFIG_DIR = Path("src/config/wandb_sweep")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m src.training.run_sweep <sweep_config> <base_config>")

    sweep_config_path = SWEEP_CONFIG_DIR / sys.argv[1]
    if not sweep_config_path.exists():
        sys.exit(f"Sweep config not found: {sweep_config_path}")

    base_config = load_config(sys.argv[2])

    sweep_config = yaml.safe_load(sweep_config_path.read_text())
    sweep_id = wandb.sweep(sweep_config, project="film-critic-lm", entity="the-trufiti-group")

    wandb.agent(sweep_id, function=lambda: train(base_config))


if __name__ == "__main__":
    main()
