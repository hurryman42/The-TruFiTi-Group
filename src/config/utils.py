from datetime import datetime
from pathlib import Path

from src.config.config import Config

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TOKENIZER_DIR = BASE_DIR / "tokenizer"


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)

    if not path.is_absolute():
        path = CONFIG_DIR / path

    if not path.suffix:
        path = path.with_suffix(".yml")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return Config.from_yaml(path)


def get_tokenizer_dir() -> Path:
    return TOKENIZER_DIR


def get_tokenizer_path(tokenizer_name) -> Path:
    return TOKENIZER_DIR / tokenizer_name


def get_data_path(file: str) -> Path:
    return DATA_DIR / file


def get_model_save_path(config: Config, num_params: int) -> Path:
    params_millions = num_params / 1_000_000
    time = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    return MODEL_DIR / f"{config.model.type}_L{config.data.level}_{params_millions:.1f}M_{time}.pt"
