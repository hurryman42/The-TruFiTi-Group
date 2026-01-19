from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml

from src.enums import ModelTypeEnum, TokenizerTypeEnum


@dataclass
class ModelConfig:
    type: ModelTypeEnum
    d_model: int
    num_heads: int
    num_blocks: int
    seq_len: int
    dropout: float
    use_rope: bool
    ff_hidden_dim: int | None = None
    head_dim: int | None = None

    def __post_init__(self):
        if self.ff_hidden_dim is None:
            self.ff_hidden_dim = self.d_model * 4
        if self.head_dim is None:
            self.head_dim = self.d_model // self.num_heads

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            type=ModelTypeEnum(raw["type"]),
            d_model=raw["d_model"],
            num_heads=raw["num_heads"],
            num_blocks=raw["num_blocks"],
            seq_len=raw["seq_len"],
            dropout=raw["dropout"],
            use_rope=raw["use_rope"],
            ff_hidden_dim=raw.get("ff_hidden_dim"),
            head_dim=raw.get("head_dim"),
        )


@dataclass
class TokenizerConfig:
    type: TokenizerTypeEnum
    name: str

    @classmethod
    def from_dict(cls, raw: dict, level: int, data_file: str) -> Self:
        tokenizer_type = TokenizerTypeEnum(raw["type"])

        match tokenizer_type:
            case TokenizerTypeEnum.CHAR:
                name = "char_tokenizer.json"
            case TokenizerTypeEnum.BPE_HUGGING_FACE:
                name = f"bpe_hf_L{level}_{data_file}.json"
            case _:
                name = f"bpe_custom_L{level}_{data_file}.json"

        return cls(
            type=tokenizer_type,
            name=name,
        )


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_iters: int
    eval_interval: int
    eval_iters: int
    warmup_iters: int

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            batch_size=raw["batch_size"],
            learning_rate=float(raw["learning_rate"]),
            weight_decay=raw["weight_decay"],
            max_iters=raw["max_iters"],
            eval_interval=raw["eval_interval"],
            eval_iters=raw["eval_iters"],
            warmup_iters=raw["warmup_iters"],
        )


@dataclass
class DataConfig:
    file: str
    file_name: str
    level: int
    seed: int
    train_size: float
    val_size: float
    test_size: float

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        return cls(
            file=raw["file"],
            file_name=Path(raw["file"]).stem,
            level=raw["level"],
            seed=raw["seed"],
            train_size=raw["train_size"],
            val_size=raw["val_size"],
            test_size=raw["test_size"],
        )


@dataclass
class Config:
    model: ModelConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    data: DataConfig

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        data_config = DataConfig.from_dict(raw["data"])
        return cls(
            model=ModelConfig.from_dict(raw["model"]),
            tokenizer=TokenizerConfig.from_dict(raw["tokenizer"], data_config.level, data_config.file_name),
            training=TrainingConfig.from_dict(raw["training"]),
            data=data_config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
