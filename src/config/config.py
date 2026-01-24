from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml

from src.enums import ModelTypeEnum, TokenizerTypeEnum


# Model Configs


@dataclass
class BaseModelConfig(ABC):
    type: ModelTypeEnum

    @classmethod
    @abstractmethod
    def from_dict(cls, raw: dict) -> Self:
        pass


@dataclass
class BigramModelConfig(BaseModelConfig):
    d_model: int

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            type=ModelTypeEnum(raw["type"]),
            d_model=raw["d_model"],
        )

    def __post_init__(self):
        pass


@dataclass
class GRUModelConfig(BaseModelConfig):
    seq_len: int
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            type=ModelTypeEnum(raw["type"]),
            seq_len=raw["seq_len"],
            input_size=raw["input_size"],
            hidden_size=raw["hidden_size"],
            num_layers=raw["num_layers"],
            dropout=raw["dropout"],
        )

    def __post_init__(self):
        pass


@dataclass
class TransformerModelConfig(BaseModelConfig):
    d_model: int
    seq_len: int
    num_heads: int
    num_blocks: int
    dropout: float
    use_rope: bool
    ff_hidden_dim: int
    head_dim: int

    def __post_init__(self):
        if self.ff_hidden_dim is None:
            self.ff_hidden_dim = self.d_model * 4
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        d_model = raw["d_model"]
        num_heads = raw["num_heads"]
        return cls(
            type=ModelTypeEnum(raw["type"]),
            d_model=d_model,
            seq_len=raw["seq_len"],
            num_heads=num_heads,
            num_blocks=raw["num_blocks"],
            dropout=raw["dropout"],
            use_rope=raw["use_rope"],
            ff_hidden_dim=raw.get("ff_hidden_dim", d_model * 4),
            head_dim=raw.get("head_dim", d_model // num_heads),
        )


type ModelConfig = BigramModelConfig | GRUModelConfig | TransformerModelConfig


def model_config_from_dict(raw: dict) -> ModelConfig:
    model_type = ModelTypeEnum(raw["type"])
    match model_type:
        case ModelTypeEnum.BIGRAM:
            return BigramModelConfig.from_dict(raw)
        case ModelTypeEnum.GRU:
            return GRUModelConfig.from_dict(raw)
        case ModelTypeEnum.TRANSFORMER:
            return TransformerModelConfig.from_dict(raw)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


# Training Configs


@dataclass
class BaseTrainingConfig(ABC):
    batch_size: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    eval_iters: int

    @classmethod
    @abstractmethod
    def from_dict(cls, raw: dict) -> Self:
        pass


@dataclass
class BigramTrainingConfig(BaseTrainingConfig):
    seq_len: int

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            batch_size=raw["batch_size"],
            seq_len=raw["seq_len"],
            learning_rate=float(raw["learning_rate"]),
            max_iters=raw["max_iters"],
            eval_interval=raw["eval_interval"],
            eval_iters=raw["eval_iters"],
        )


@dataclass
class GRUTrainingConfig(BaseTrainingConfig):
    weight_decay: float

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            batch_size=raw["batch_size"],
            learning_rate=float(raw["learning_rate"]),
            max_iters=raw["max_iters"],
            eval_interval=raw["eval_interval"],
            eval_iters=raw["eval_iters"],
            weight_decay=raw["weight_decay"],
        )


@dataclass
class TransformerTrainingConfig(BaseTrainingConfig):
    weight_decay: float
    warmup_iters: int

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            batch_size=raw["batch_size"],
            learning_rate=float(raw["learning_rate"]),
            max_iters=raw["max_iters"],
            eval_interval=raw["eval_interval"],
            eval_iters=raw["eval_iters"],
            weight_decay=raw["weight_decay"],
            warmup_iters=raw["warmup_iters"],
        )


type TrainingConfig = BigramTrainingConfig | TransformerTrainingConfig | GRUTrainingConfig


def training_config_from_dict(raw: dict, model_type: ModelTypeEnum) -> TrainingConfig:
    match model_type:
        case ModelTypeEnum.TRANSFORMER:
            return TransformerTrainingConfig.from_dict(raw)
        case ModelTypeEnum.BIGRAM:
            return BigramTrainingConfig.from_dict(raw)
        case ModelTypeEnum.GRU:
            return GRUTrainingConfig.from_dict(raw)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


# Tokenizer Config


@dataclass
class TokenizerConfig:
    type: TokenizerTypeEnum
    name: str

    @classmethod
    def from_dict(cls, raw: dict, level: int | None, data_file: str) -> Self:
        tokenizer_type = TokenizerTypeEnum(raw["type"])

        match tokenizer_type:
            case TokenizerTypeEnum.CHAR:
                name = "char_tokenizer.json"
            case TokenizerTypeEnum.BPE_HUGGING_FACE:
                name = f"bpe_hf_L{level}_{data_file}.json" if level else f"bpe_hf_{data_file}.json"
            case _:
                name = f"bpe_custom_L{level}_{data_file}.json" if level else f"bpe_custom_{data_file}.json"

        return cls(type=tokenizer_type, name=name)


# Data Config


@dataclass
class DataConfig:
    file: str
    file_name: str
    train_size: float
    val_size: float
    test_size: float
    seed: int
    level: int

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            file=raw["file"],
            file_name=Path(raw["file"]).stem,
            train_size=raw["train_size"],
            val_size=raw["val_size"],
            test_size=raw["test_size"],
            seed=raw["seed"],
            level=raw["level"],
        )


# Main Config


@dataclass
class Config:
    model: ModelConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    data: DataConfig
    config_file_name: str | None

    @classmethod
    def from_dict(cls, raw: dict, config_file_name: str | None = None) -> Self:
        data_config = DataConfig.from_dict(raw["data"])
        model_config = model_config_from_dict(raw["model"])
        return cls(
            model=model_config,
            tokenizer=TokenizerConfig.from_dict(raw["tokenizer"], data_config.level, data_config.file_name),
            training=training_config_from_dict(raw["training"], model_config.type),
            data=data_config,
            config_file_name=config_file_name,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, config_file_name=path.stem)
