from enum import StrEnum


class TransformerCheckpointEnum(StrEnum):
    MODEL = "model"
    VOCAB_SIZE = "vocab_size"
    NUM_PARAMS = "num_params"
    CONFIG = "config"
    TOKENIZER = "tokenizer"


class BigramCheckpointEnum(StrEnum):
    MODEL = "model"
    TOKEN_EMBEDDING = "token_embedding"
    POS_ENCODING = "pos_encoding"
    VOCAB_SIZE = "vocab_size"
    CONFIG = "config"
    TOKENIZER = "tokenizer"


class GruCheckpointEnum(StrEnum):
    MODEL = "model"
    VOCAB_SIZE = "vocab_size"
    NUM_PARAMS = "num_params"
    CONFIG = "config"
    TOKENIZER = "tokenizer"
