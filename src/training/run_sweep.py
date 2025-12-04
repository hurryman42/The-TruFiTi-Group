import sys
from pathlib import Path

import yaml

import wandb
from src.training.train_transformer import main as train

SWEEP_CONFIG_DIR = Path("src/config/wandb_sweep")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.training.run_sweep <config_name>")

    config_path = SWEEP_CONFIG_DIR / sys.argv[1]
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    sweep_config = yaml.safe_load(config_path.read_text())
    sweep_id = wandb.sweep(sweep_config, project="film-critic-lm", entity="the-trufiti-group")
    wandb.agent(sweep_id, function=lambda: train({}))


if __name__ == "__main__":
    main()
