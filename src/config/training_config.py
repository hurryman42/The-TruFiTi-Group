from dataclasses import dataclass, field
from pathlib import Path

from src.enums import TokenizerType

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class BigramConfig:
    tokenizer_type: TokenizerType = TokenizerType.BPE_HUGGING_FACE

    d_model: int = 256
    seq_len: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3

    max_iters: int = 3000
    eval_interval: int = 1000  # interval of printing estimated loss
    eval_iters: int = 50  # number of batches to average for loss estimation

    train_size: float = 0.9
    val_size: float = 0.1

    data_path: Path = BASE_DIR / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
    model_save_path: Path = BASE_DIR / "models" / "bigram_model.pt"

    def __post_init__(self):
        if self.tokenizer_type == TokenizerType.BPE_HUGGING_FACE:
            self.model_save_path = BASE_DIR / "models" / "bigram_model_bpe_hugging_face.pt"
            self.batch_size = 64

    @property
    def tokenizer_path(self) -> Path:
        if self.tokenizer_type == TokenizerType.CHAR:
            return BASE_DIR / "tokenizer" / "char_tokenizer.json"
        return BASE_DIR / "tokenizer" / "bpe_hugging_face_tokenizer.json"


@dataclass
class TransformerConfig:
    tokenizer_type: TokenizerType = TokenizerType.BPE_HUGGING_FACE

    d_model: int = 256
    num_heads: int = 8
    num_blocks: int = 6
    seq_len: int = 128
    ff_hidden_dim: int = field(init=False)
    dropout: float = 0.1

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0  # for AdamW

    max_iters: int = 3000
    eval_interval: int = 1000
    eval_iters: int = 20

    train_size: float = 0.9
    val_size: float = 0.1

    data_path: Path = BASE_DIR / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"

    def __post_init__(self):
        self.ff_hidden_dim = 4 * self.d_model

    @property
    def tokenizer_path(self) -> Path:
        if self.tokenizer_type == TokenizerType.CHAR:
            return BASE_DIR / "tokenizer" / "char_tokenizer.json"
        return BASE_DIR / "tokenizer" / "bpe_hugging_face_tokenizer.json"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def get_model_save_path(self, num_params: int) -> Path:
        params_millions = num_params / 1_000_000
        return BASE_DIR / "models" / f"transformer_{params_millions:.1f}m.pt"
