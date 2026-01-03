from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.enums import DataConfigEnum, ModelTypeEnum, SectionEnum, TokenizerTypeEnum, TransformerModelEnum

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TOKENIZER_DIR = BASE_DIR / "tokenizer"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.is_absolute():
        path = CONFIG_DIR / path

    if not path.suffix:
        path = path.with_suffix(".yml")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    _add_computed_fields(config)

    return config


def _add_computed_fields(config: dict[str, Any]) -> None:
    model_type = get_model_type(config)
    model_section = config[SectionEnum.MODEL]

    if model_type == ModelTypeEnum.TRANSFORMER:
        d_model = model_section[TransformerModelEnum.D_MODEL]
        num_heads = model_section[TransformerModelEnum.NUM_HEADS]
        model_section[TransformerModelEnum.FF_HIDDEN_DIM] = 4 * d_model
        model_section[TransformerModelEnum.HEAD_DIM] = d_model // num_heads


def recompute_computed_fields(config: dict[str, Any]) -> None:
    _add_computed_fields(config)


def get_model_type(config: dict[str, Any]) -> ModelTypeEnum:
    return ModelTypeEnum(config[SectionEnum.MODEL][TransformerModelEnum.TYPE])


def get_tokenizer_type(config: dict[str, Any]) -> TokenizerTypeEnum:
    return TokenizerTypeEnum(config[SectionEnum.TOKENIZER][TransformerModelEnum.TYPE])


def get_tokenizer_name(config: dict[str, Any]) -> str:
    tokenizer_type = get_tokenizer_type(config)
    level = get_level(config)
    data_file = get_data_name(config)
    if tokenizer_type == TokenizerTypeEnum.CHAR:
        return "char_tokenizer.json"
    elif tokenizer_type == TokenizerTypeEnum.BPE_HUGGING_FACE:
        return f"bpe_hf_L{level}_{data_file}.json"
    return f"bpe_custom_L{level}_{data_file}.json"


def get_tokenizer_path(config: dict[str, Any]) -> Path:
    tokenizer_name = get_tokenizer_name(config)
    return TOKENIZER_DIR / tokenizer_name


def get_data_path(config: dict[str, Any]) -> Path:
    data_file = config[SectionEnum.DATA][DataConfigEnum.FILE]
    return DATA_DIR / data_file


def get_data_name(config: dict[str, Any]) -> str:
    return Path(config[SectionEnum.DATA][DataConfigEnum.FILE]).stem


def get_level(config: dict[str, Any]) -> str:
    return config[SectionEnum.DATA][DataConfigEnum.LEVEL]


def get_model_save_path(config: dict[str, Any], num_params: int) -> Path:
    model_type = get_model_type(config)
    level = get_level(config)
    params_millions = num_params / 1_000_000

    if model_type == ModelTypeEnum.BIGRAM:
        tokenizer_type = get_tokenizer_type(config)
        if tokenizer_type == TokenizerTypeEnum.BPE_HUGGING_FACE:
            return MODEL_DIR / "bigram_model_bpe_hf.pt"
        return MODEL_DIR / "bigram_model.pt"

    return MODEL_DIR / f"transformer_L{level}_{params_millions:.1f}M_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.pt"
